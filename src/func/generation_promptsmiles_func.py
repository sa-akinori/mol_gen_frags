import argparse
import os
from collections import Counter

import datasets
import numpy as np
import pandas as pd
import torch
from promptsmiles import FragmentLinker, ScaffoldDecorator
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerBase, set_seed

from func.fragmentation import GetNHA
from func.utility import BASEPATH, INVALID_SMILES, LogFile


def log_line(message: str, collected: list[str]) -> None:
    """Print a log line and keep it for the log file written at the end of the run.

    Args:
        message: Line to print.
        collected: List of the lines logged so far, appended to in place.
    """
    print(message)
    collected.append(message)


def is_promptsmiles_expressible(fragment_set: str) -> bool:
    """Tell whether promptsmiles can prompt every fragment of the set.

    ScaffoldDecorator takes a single fragment whatever its attachment points, and FragmentLinker
    takes any number of fragments as long as each carries exactly one, so this condition is
    exactly the range promptsmiles can represent.

    Args:
        fragment_set: Dot-separated fragment SMILES.

    Returns:
        True when the set holds one fragment or every fragment carries exactly one attachment
        point.
    """
    fragments = fragment_set.split(".")
    return len(fragments) == 1 or all(frag.count("*") == 1 for frag in fragments)


def select_prompt_fragments(fragment_set: str) -> tuple[str, list[str]] | None:
    """Route a fragment set to the promptsmiles sampler able to prompt all of its fragments.

    Sets outside the range of :func:`is_promptsmiles_expressible` are not generated at all: no
    part of the set is substituted for the whole, so the row is reported as unsupported by the
    caller instead of being scored on a task of our own making.

    Note: from three fragments on, promptsmiles enables ``scan=True`` automatically, which
    multiplies the number of sampling calls (and therefore the runtime) per molecule.

    Args:
        fragment_set: Dot-separated fragment SMILES.

    Returns:
        Pair ``(sampler_name, fragments)``: ``scaffold`` with the single fragment of the set, or
        ``linking`` with every fragment of the set (largest first). None when promptsmiles cannot
        express the set or when RDKit fails to parse one of its fragments.
    """
    if not is_promptsmiles_expressible(fragment_set):
        return None
    fragments = [(frag, Chem.MolFromSmiles(frag)) for frag in fragment_set.split(".")]
    if any(mol is None for _, mol in fragments):
        return None
    # FragmentLinker raises `IndexError: pop from empty list` on a single fragment.
    if len(fragments) == 1:
        return "scaffold", [fragments[0][0]]
    return "linking", [frag for frag, _ in sorted(fragments, key=lambda pair: GetNHA(pair[1]), reverse=True)]


def to_prediction_row(sampled: list[str], n_samples: int) -> list[str]:
    """Fit sampled SMILES into a fixed-width prediction row.

    The SMILES are kept exactly as sampled (invalid ones included) because validity is a
    metric computed later by the evaluation pipeline.

    Args:
        sampled: SMILES returned by promptsmiles.
        n_samples: Width of the prediction row.

    Returns:
        Exactly ``n_samples`` strings, right-padded with :data:`INVALID_SMILES` when fewer were
        sampled.
    """
    return (list(sampled) + [INVALID_SMILES] * n_samples)[:n_samples]


def format_run_summary(
    ungenerated_counts: Counter[str],
    sampler_counts: Counter[str],
    n_test: int,
) -> str:
    """Render the per-sampler and per-reason statistics of a generation run.

    Args:
        ungenerated_counts: Number of rows left as :data:`INVALID_SMILES` per reason.
        sampler_counts: Number of generated rows per promptsmiles sampler.
        n_test: Number of rows read from the test split.

    Returns:
        Multi-line report holding the totals, one line per sampler and one line per reason a row
        was not generated, each with its share of the test split.
    """
    n_ungenerated = sum(ungenerated_counts.values())
    lines = [
        f"test rows: {n_test}",
        f"generated rows: {sum(sampler_counts.values())}",
        f"ungenerated rows (filled with {INVALID_SMILES}): {n_ungenerated} ({n_ungenerated / n_test:.1%})",
    ]
    lines += [f"  sampler {name}: {count} ({count / n_test:.1%})" for name, count in sorted(sampler_counts.items())]
    lines += [f"  ungenerated ({reason}): {count} ({count / n_test:.1%})" for reason, count in sorted(ungenerated_counts.items())]
    return "\n".join(lines)


class GPT2PromptSampler:
    """Sampling and likelihood callbacks required by the ``promptsmiles`` API.

    The prompts handed over by promptsmiles are *partial* SMILES (unclosed ring closures and
    branches are expected), so no RDKit parsing is done here: the prior simply continues the
    token sequence ``<bos> prompt``.

    ``gen_method`` selects how the continuation is decoded, ``sampling`` (multinomial) or
    ``beam`` (beam search over ``num_beams`` beams, matching RFFMG and SAFE).
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        max_length: int,
        gen_method: str,
        num_beams: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.gen_method = gen_method
        self.num_beams = num_beams

    def sample(self, prompt: str | list[str], batch_size: int) -> list[str]:
        """Sample one SMILES per (partial) SMILES prompt in a single batched forward pass.

        A single ``str`` is replicated ``batch_size`` times, which is exactly what
        ``num_return_sequences=batch_size`` on one input row does internally, so both call
        styles return ``batch_size`` samples of the same prompt (independent ones under
        multinomial sampling).

        Both decoding schemes return exactly one completion per prompt row
        (``num_return_sequences=1`` for beam search), which is the contract promptsmiles relies
        on under ``batch_prompts=True``. Beam search is deterministic, so the ``batch_size``
        rows of a replicated ``str`` prompt all collapse to the same SMILES: scaffold
        decoration then yields largely duplicated samples and a lower uniqueness than
        multinomial sampling. Beam search also expands ``num_beams`` beams for *every* row of
        the batch at once, so drop ``--n_samples`` if the run runs out of GPU memory.

        Prompts of different length are **left**-padded: decoder-only models must see their
        prompt flush against the generated tokens. Left padding also makes every row start
        generating at the same column ``prompt_len``, so the completion of row ``i`` is simply
        ``outputs[i, prompt_len:]``. The returned SMILES is built as ``prompt + completion``
        rather than by decoding the padded row, which guarantees the prefix property
        promptsmiles asserts (``smiles.startswith(prompt)``) without relying on the tokenizer
        round-trip of the prompt.

        Args:
            prompt: Partial SMILES supplied by promptsmiles, or a list of them (one per sample,
                as passed by ``_batch_sample``). May be empty for de novo sampling.
            batch_size: Number of sequences to sample when ``prompt`` is a single string.

        Returns:
            One decoded SMILES per prompt, in prompt order; each starts with its own prompt.
            The length is ``batch_size`` for a ``str`` prompt and ``len(prompt)`` for a list.
        """
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        pad_id = self.tokenizer.pad_token_id
        encoded = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        prompt_ids = [[self.tokenizer.bos_token_id] + ids for ids in encoded]
        prompt_len = max(len(ids) for ids in prompt_ids)
        input_ids = torch.tensor([[pad_id] * (prompt_len - len(ids)) + ids for ids in prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([[0] * (prompt_len - len(ids)) + [1] * len(ids) for ids in prompt_ids], dtype=torch.long, device=self.device)
        # promptsmiles expects one completion per prompt, not one per beam.
        decode_params = ({"do_sample": True} if self.gen_method == "sampling"
                         else {"do_sample": False, "num_beams": self.num_beams, "num_return_sequences": 1, "early_stopping": True})
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **decode_params,
            )
        completions = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        return [prompt_smi + completion for prompt_smi, completion in zip(prompts, completions)]

    def evaluate(self, smiles: list[str]) -> np.ndarray:
        """Compute the negative log-likelihood of complete SMILES under the prior.

        Args:
            smiles: SMILES strings to score.

        Returns:
            Array of shape ``(len(smiles),)`` with the summed NLL (nats) of
            ``<bos> SMILES <eos>`` per sequence.
        """
        pad_id = self.tokenizer.pad_token_id
        sequences = [
            [self.tokenizer.bos_token_id] + self.tokenizer(smi, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
            for smi in smiles
        ]
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.tensor([seq + [pad_id] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([[1] * len(seq) + [0] * (max_len - len(seq)) for seq in sequences], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        # Shift by one: position t predicts token t+1.
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        token_nll = -log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return (token_nll * attention_mask[:, 1:]).sum(dim=1).cpu().numpy()


def build_prompter(
    sampler_name: str,
    fragments: list[str],
    sampler: GPT2PromptSampler,
    n_samples: int,
    random_seed: int,
) -> ScaffoldDecorator | FragmentLinker:
    """Create the promptsmiles sampler the row was routed to.

    Args:
        sampler_name: Either ``scaffold`` (decorate the single prompted fragment) or ``linking``
            (link all prompted fragments), as returned by :func:`select_prompt_fragments`.
        fragments: Fragment SMILES to prompt, as returned by :func:`select_prompt_fragments`.
        sampler: Provider of the ``sample_fn`` / ``evaluate_fn`` callbacks.
        n_samples: Number of SMILES to sample per molecule.
        random_seed: Seed handed to the promptsmiles sampler.

    Both samplers get ``batch_prompts=True``: ``GPT2PromptSampler.sample`` accepts a list of
    prompts, so promptsmiles calls it once per prompting round instead of ``n_samples`` times.

    Returns:
        ``ScaffoldDecorator`` for the scaffold sampler, ``FragmentLinker`` for the linking one.
    """
    if sampler_name == "scaffold":
        return ScaffoldDecorator(
            scaffold=fragments[0],
            batch_size=n_samples,
            sample_fn=sampler.sample,
            evaluate_fn=sampler.evaluate,
            batch_prompts=True,
            random_seed=random_seed,
        )
    return FragmentLinker(
        fragments=fragments,
        batch_size=n_samples,
        sample_fn=sampler.sample,
        evaluate_fn=sampler.evaluate,
        batch_prompts=True,
        random_seed=random_seed,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PromptSMILES generation."""
    parser = argparse.ArgumentParser(description="Generate molecules with PromptSMILES using a plain-SMILES GPT2 prior")
    parser.add_argument("--frag_method", type=str, default="brics", choices=["brics", "rc_cms"], help="Fragmentation method (default: brics)")
    parser.add_argument("--model_ver", type=str, default="finetuning", choices=["finetuning", "from_scratch"], help="Model version (default: finetuning)")
    parser.add_argument("--gen_method", type=str, default="sampling", choices=["sampling", "beam"], help="Decoding scheme: multinomial sampling or beam search (default: sampling)")
    parser.add_argument("--additional_path", type=str, default="normal", help="Additional path segment to append to the output dir")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate per molecule (default: 50)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length (default: 256)")
    parser.add_argument("--num_beams", type=int, default=50, help="Number of beams, used by the beam gen_method (default: 50)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)

    # Setting up the paths
    model_path = f"{BASEPATH}/models/promptsmiles/gpt/{args.model_ver}/{args.frag_method}/best_model"
    output_dir = f"{BASEPATH}/results/promptsmiles/gpt/{args.model_ver}/{args.frag_method}/{args.gen_method}/{args.additional_path}"
    os.makedirs(output_dir, exist_ok=True)

    # Setting up logging.
    log_lines: list[str] = []
    log_line(f"args: {vars(args)}", log_lines)
    log_line(f"gen_method: {args.gen_method}, num_beams: {args.num_beams}", log_lines)
    log_line(f"model_path: {model_path}", log_lines)

    # Loding the test datasets.
    test_dataset = datasets.load_from_disk(f"{BASEPATH}/data/promptsmiles/{args.frag_method}/normal")["test"]
    test_fragment_sets, test_smiles = test_dataset["pass_fragments"], test_dataset["smiles"]
    n_test = len(test_smiles)

    # Generating molecules from the fragments.
    ungenerated_counts: Counter[str] = Counter()
    sampler_names: list[str] = ["unsupported"] * n_test
    prompts: list[tuple[int, str, list[str]]] = []
    for idx, (smiles, fragment_set) in enumerate(tqdm(zip(test_smiles, test_fragment_sets), total=n_test, desc="building prompts")):
        if Chem.MolFromSmiles(smiles) is None:
            sampler_names[idx] = "invalid_target"
            ungenerated_counts["invalid_target"] += 1
            log_lines.append(f"unparsable target SMILES: target={smiles}")
            continue
        selected = select_prompt_fragments(fragment_set)
        if selected is None:
            ungenerated_counts["unsupported"] += 1
            continue
        sampler_name, fragments = selected
        sampler_names[idx] = sampler_name
        prompts.append((idx, sampler_name, fragments))
    log_line(f"test rows: {n_test}, prompted rows: {len(prompts)}", log_lines)

    # Loading the model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    sampler = GPT2PromptSampler(model=model, tokenizer=tokenizer, device=device, max_length=args.max_length, gen_method=args.gen_method, num_beams=args.num_beams)

    # Generating
    sampler_counts: Counter[str] = Counter()
    prompt_fragments: list[str] = [""] * n_test
    predictions: list[list[str]] = [[INVALID_SMILES] * args.n_samples for _ in range(n_test)]
    for idx, sampler_name, fragments in tqdm(prompts, desc="promptsmiles generation"):
        prompter_seed = args.random_seed + idx
        torch.manual_seed(prompter_seed)
        try:
            prompter = build_prompter(sampler_name, fragments, sampler, args.n_samples, prompter_seed)
            sampled = prompter.sample()
        except Exception as err:
            sampler_names[idx] = "generation_error"
            ungenerated_counts["generation_error"] += 1
            log_lines.append(
                f"generation failed: target={test_smiles[idx]}, sampler={sampler_name}, "
                f"fragments={'.'.join(fragments)}, error={type(err).__name__}: {err}"
            )
            continue
        sampler_counts[sampler_name] += 1
        prompt_fragments[idx] = ".".join(fragments)
        predictions[idx] = to_prediction_row(sampled, args.n_samples)

    # The evaluation reads the requested fragment set from the fragment column; the fragments the
    # samplers were really prompted with stay in prompt_fragments.
    assert all(len(column) == n_test for column in (test_fragment_sets, test_smiles, sampler_names, prompt_fragments, predictions)), "prediction rows do not cover the test split"

    columns = [f"prediction_{i + 1}" for i in range(args.n_samples)]
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df.insert(0, "prompt_fragments", prompt_fragments)
    predictions_df.insert(0, "sampler", sampler_names)
    predictions_df.insert(0, "target", test_smiles)
    predictions_df.insert(0, "fragment", test_fragment_sets)
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    log_line(format_run_summary(ungenerated_counts, sampler_counts, n_test), log_lines)
    with LogFile(f"{output_dir}/generation_params.txt") as logfp:
        for line in log_lines:
            logfp.write(line, suppress_std_out=True)
    print(f"Saved SMILES predictions to: {output_dir}/predictions.csv")
    print("Generation completed!")


if __name__ == "__main__":
    main()
