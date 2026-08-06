"""Generate molecules with PromptSMILES on top of a trained plain-SMILES GPT2 prior.

PromptSMILES turns an unconditional SMILES language model into a scaffold decorator /
fragment linker at *inference* time: the scaffold is re-rooted so that an attachment
point sits at the end of the string, the attachment token is dropped, and the model is
asked to continue the (partial) SMILES. The library drives that loop and only needs two
callbacks, both provided by :class:`GPT2PromptSampler`:

    - ``sample_fn(prompt, batch_size) -> list[str]``: sampled completions of a prompt.
    - ``evaluate_fn(smiles) -> negative log-likelihood`` of complete SMILES.

The prompted fragments are read from the ``pass_fragments`` column of the PromptSMILES dataset
(``data/promptsmiles/{frag_method}/normal``, test split), and the reference molecule is the
``smiles`` column of the same row. That dataset is written by ``make_datasets.py`` from the split
SAFE uses, so its rows are identical to SAFE's; it is also the dataset the prior is trained on
(``train_promptsmiles.py`` reads its ``train``/``validation`` splits). RFFMG (``test.source``) and
SAFE (``generation_safe_func.py``) generate from that very column too, so the three methods start
from identical fragment sets. The fragment sets are deliberately *not* recomputed here:
re-fragmenting the test molecules loses part of them and yields a fragment population the other
two methods never saw.

Every test row is a generation target: instead of running one fixed task and skipping the rows
whose fragment set does not fit it, each row is routed to the promptsmiles sampler that can
handle its fragment set (see :func:`select_prompt_fragments`). RFFMG, SAFE and PromptSMILES are
then scored on the same population.

Outputs (read unchanged by ``src/evaluation.py``):
    - ``data/promptsmiles/{frag_method}/normal/test.source``: fragment set per test molecule.
      This is the directory holding the dataset read above; ``save_to_disk`` leaves unrelated
      files untouched (verified 2026-08-02), so the two files live next to the dataset.
    - ``data/promptsmiles/{frag_method}/normal/test.target``: the test molecule itself.
    - ``results/promptsmiles/gpt/{model_ver}/{frag_method}/{gen_method}/normal/predictions.csv``:
      columns ``target``, ``sampler``, ``prediction_1`` .. ``prediction_N`` (the T5Chem layout
      plus the sampler the row was routed to, so the two sampler populations can be told apart
      during evaluation). ``func/evaluation_func.py`` addresses every column it reads by name
      (verified 2026-07-29), so the extra column does not disturb the evaluation.

All three files hold one row per *successfully generated* row of the test split, in the same
order: the evaluation pipeline joins test.source and predictions.csv by row number, so rows
skipped during fragment selection or generation must be absent from all three files.
Skipped rows are logged individually and counted per reason in ``generation_params.txt``.

API notes (verified 2026-07-28 against ``promptsmiles/samplers.py``):
    - ``ScaffoldDecorator(scaffold=..., batch_size=..., sample_fn=..., evaluate_fn=...,
      batch_prompts=..., random_seed=...)`` and ``FragmentLinker(fragments=..., batch_size=...,
      sample_fn=..., evaluate_fn=..., batch_prompts=..., random_seed=...)`` take the keyword
      names used below.
    - ``sample_fn(prompt: str | list[str], batch_size: int)`` must return a *list of SMILES
      strings*.
      The official docstring reads as if a ``(smiles, nlls)`` tuple were expected, but only
      ``DeNovo`` unpacks a tuple; ``ScaffoldDecorator`` / ``FragmentLinker`` index the return
      value as strings (e.g. ``smiles[0].startswith(prompt)``).
    - Each returned SMILES must keep its prompt as *prefix*: ``samplers.py`` asserts
      ``smiles.startswith(prompt)`` and locates the completion in prompt-token space.
      :meth:`GPT2PromptSampler.sample` therefore returns ``prompt + decoded_completion``
      instead of decoding the whole (left-padded) ``model.generate`` output.
    - ``FragmentLinker`` accepts a fragment set only when *every* fragment carries exactly one
      ``*`` (asserted in ``samplers.py``) and when the set holds at least two fragments: with a
      single fragment ``fragments.pop()`` hits an emptied list and raises ``IndexError: pop from
      empty list``, so one attachment point alone is not sufficient. Both constraints are
      enforced by the routing rule of :func:`select_prompt_fragments`.
    - ``evaluate_fn(smiles: list[str])`` must return indexable negative log-likelihoods, sorted
      ascending by the library (smaller NLL = better).
    - ``batch_prompts=True`` is passed below: promptsmiles then calls
      ``sample_fn(prompt=[p_1..p_n], batch_size=n)`` once and expects exactly *one* completion
      per prompt, in the same order. With the default ``False`` it instead calls ``sample_fn``
      with ``batch_size=1`` once per sample, i.e. ``n_samples`` times per molecule, which is
      prohibitively slow on the whole test split.
    - ``--gen_method`` picks the decoding scheme: ``sampling`` (multinomial, the scheme
      PromptSMILES was published with) or ``beam`` (beam search over ``--num_beams`` beams, the
      scheme RFFMG and SAFE use, so that the three methods can be compared without the decoding
      scheme confounding the metrics). Because ``batch_prompts=True`` asks for exactly one
      completion per prompt, beam search keeps ``num_return_sequences=1``.
"""

import argparse
import os
import time
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
from func.utility import BASEPATH, LogFile, save_file

# FragmentLinker needs at least two fragments to have something to link: with a single fragment
# its `fragments.pop()` hits an emptied list and raises `IndexError: pop from empty list`, even
# when that fragment carries exactly one attachment point (measured 2026-07-29).
MIN_LINK_FRAGMENTS = 2


def read_test_split(frag_method: str) -> tuple[list[str], list[str]]:
    """Read the reference molecules and their prompted fragment sets from the PromptSMILES dataset.

    ``data/promptsmiles/{frag_method}/normal`` is written by ``src/make_datasets.py`` from the
    same split SAFE uses (``generation_safe_func.py`` reads the same ``pass_fragments`` column)
    and, in its isotope-stripped form, RFFMG (``test.source``), so the rows are shared across the
    three methods while each method keeps its own copy of the data.

    Args:
        frag_method: Fragmentation method, either ``brics`` or ``rc_cms``.

    Returns:
        Pair ``(smiles, pass_fragments)`` of equally long lists, aligned row by row: the
        reference molecule and its dot-separated fragment set.
    """
    test_dataset = datasets.load_from_disk(f"{BASEPATH}/data/promptsmiles/{frag_method}/normal")["test"]
    return test_dataset["smiles"], test_dataset["pass_fragments"]


def parse_fragments(fragment_set: str) -> list[tuple[str, Chem.Mol]]:
    """Split a fragment set into RDKit-parsable (SMILES, Mol) pairs.

    Args:
        fragment_set: Dot-separated fragment SMILES.

    Returns:
        List of pairs for the fragments RDKit could parse; unparsable fragments are dropped.
    """
    parsed = [(frag, Chem.MolFromSmiles(frag)) for frag in fragment_set.split(".")]
    return [(frag, mol) for frag, mol in parsed if mol is not None]


def select_prompt_fragments(fragment_set: str) -> tuple[str, list[str]] | None:
    """Route a fragment set to the promptsmiles sampler able to prompt it.

    FragmentLinker is preferred because it prompts *every* fragment of the set and therefore
    keeps the whole conditioning information, but it only accepts sets in which each fragment
    carries exactly one attachment point (``samplers.py`` asserts this) and which hold at least
    ``MIN_LINK_FRAGMENTS`` fragments (a single fragment raises ``IndexError: pop from empty
    list``, so one attachment point alone is not sufficient). Every other set -- a set holding a
    multi-attachment fragment or a set of one fragment -- goes to ScaffoldDecorator, which is
    prompted with a single fragment and thus never runs into the linker's index errors. No row
    is dropped for failing a condition: both branches together cover the whole test split.

    Note: from three fragments on, promptsmiles enables ``scan=True`` automatically, which
    multiplies the number of sampling calls (and therefore the runtime) per molecule.

    Args:
        fragment_set: Dot-separated fragment SMILES.

    Returns:
        Pair ``(sampler_name, fragments)``: ``linking`` with every fragment of the set (largest
        first), or ``scaffold`` with the largest fragment carrying an attachment point. None
        when no fragment can be prompted at all, i.e. when RDKit parses none of the fragments or
        none of the parsable ones has an attachment point.
    """
    fragments = parse_fragments(fragment_set)
    # An unparsable fragment would silently shrink the linked set, so such sets are decorated.
    linkable = (len(fragments) == len(fragment_set.split("."))
                and len(fragments) >= MIN_LINK_FRAGMENTS
                and all(frag.count("*") == 1 for frag, _ in fragments))
    if linkable:
        return "linking", [frag for frag, _ in sorted(fragments, key=lambda pair: GetNHA(pair[1]), reverse=True)]
    candidates = [(frag, mol) for frag, mol in fragments if "*" in frag]
    if not candidates:
        return None
    return "scaffold", [max(candidates, key=lambda pair: GetNHA(pair[1]))[0]]


def to_prediction_row(sampled: list[str], n_samples: int) -> list[str]:
    """Fit sampled SMILES into a fixed-width prediction row.

    The SMILES are kept exactly as sampled (invalid ones included) because validity is a
    metric computed later by the evaluation pipeline.

    Args:
        sampled: SMILES returned by promptsmiles.
        n_samples: Width of the prediction row.

    Returns:
        Exactly ``n_samples`` strings, right-padded with ``""`` when fewer were sampled.
    """
    return (list(sampled) + [""] * n_samples)[:n_samples]


def format_run_summary(
    skip_counts: Counter[str],
    sampler_counts: Counter[str],
    n_test: int,
    n_generated: int,
) -> str:
    """Render the per-sampler and per-reason statistics of a generation run.

    Args:
        skip_counts: Number of skipped rows per reason.
        sampler_counts: Number of generated rows per promptsmiles sampler.
        n_test: Number of rows read from the test split.
        n_generated: Number of rows for which SMILES were generated.

    Returns:
        Multi-line report holding the totals, one line per sampler and one line per skip reason.
    """
    lines = [
        f"test rows: {n_test}",
        f"generated rows: {n_generated}",
        f"skipped rows: {sum(skip_counts.values())}",
    ]
    lines += [f"  sampler {name}: {count}" for name, count in sorted(sampler_counts.items())]
    lines += [f"  skipped ({reason}): {count}" for reason, count in sorted(skip_counts.items())]
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
        # The beam search settings mirror generation_rffmg_func.py, except for
        # num_return_sequences: promptsmiles wants one completion per prompt, not one per beam.
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
    parser.add_argument("--frag_method", type=str, default="brics", choices=["brics", "rc_cms"],
                        help="Fragmentation method (default: brics)")
    parser.add_argument("--model_ver", type=str, default="finetuning", choices=["finetuning", "from_scratch"],
                        help="Model version (default: finetuning)")
    parser.add_argument("--gen_method", type=str, default="sampling", choices=["sampling", "beam"],
                        help="Decoding scheme: multinomial sampling or beam search (default: sampling)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to generate per molecule (default: 50)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length (default: 256)")
    parser.add_argument("--num_beams", type=int, default=50,
                        help="Number of beams, used by the beam gen_method (default: 50)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    set_seed(args.random_seed)

    # Results of both decoding schemes live side by side under their own path segment, the one
    # src/evaluation.py selects with its own --gen_method.
    model_path = f"{BASEPATH}/models/promptsmiles/gpt/{args.model_ver}/{args.frag_method}/best_model"
    dataset_dir = f"{BASEPATH}/data/promptsmiles/{args.frag_method}/normal"
    output_dir = f"{BASEPATH}/results/promptsmiles/gpt/{args.model_ver}/{args.frag_method}/{args.gen_method}/normal"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logfp = LogFile(f"{output_dir}/generation_params.txt")
    logfp.write(f"args: {vars(args)}")
    logfp.write(f"gen_method: {args.gen_method}, num_beams: {args.num_beams}")
    logfp.write(f"model_path: {model_path}")

    # Prompts come from the fragment sets RFFMG and SAFE generate from, never from a fresh
    # fragmentation of the test molecules (see module docstring).
    test_smiles, test_fragment_sets = read_test_split(args.frag_method)

    # Every row is routed to the sampler that can prompt it, so no row is dropped for failing a
    # task condition. What remains are the degenerate rows (unparsable reference SMILES, no
    # promptable fragment); each is counted per reason and reported at the end of the run.
    skip_counts: Counter[str] = Counter()
    prompts: list[tuple[str, str, list[str]]] = []
    for smiles, fragment_set in tqdm(zip(test_smiles, test_fragment_sets), total=len(test_smiles), desc="building prompts"):
        if Chem.MolFromSmiles(smiles) is None:
            skip_counts["invalid_target_smiles"] += 1
            logfp.write(f"unparsable target SMILES: target={smiles}", suppress_std_out=True)
            continue
        selected = select_prompt_fragments(fragment_set)
        if selected is None:
            skip_counts["no_promptable_fragment"] += 1
            logfp.write(f"no promptable fragment: target={smiles}, fragments={fragment_set}", suppress_std_out=True)
            continue
        sampler_name, fragments = selected
        prompts.append((smiles, sampler_name, fragments))
    logfp.write(f"test rows: {len(test_smiles)}, prompted rows: {len(prompts)}")

    # Load the trained prior.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    sampler = GPT2PromptSampler(model=model, tokenizer=tokenizer, device=device, max_length=args.max_length,
                                gen_method=args.gen_method, num_beams=args.num_beams)

    start = time.perf_counter()
    sampler_counts: Counter[str] = Counter()
    sources: list[str] = []
    targets: list[str] = []
    sampler_names: list[str] = []
    predictions: list[list[str]] = []
    for idx, (smiles, sampler_name, fragments) in enumerate(tqdm(prompts, desc="promptsmiles generation")):
        # ScaffoldDecorator / FragmentLinker call random.seed(random_seed) in __init__, so a
        # per-row offset is required: a constant seed would reset the global RNG to the same
        # state for every row while still keeping the run reproducible. The offset follows the
        # prompt index, hence skipped rows do not shift the other seeds.
        prompter_seed = args.random_seed + idx
        # Safety net: the routing rule keeps the fragment sets promptsmiles cannot handle away
        # from FragmentLinker, so no failure is expected here, but a single raise (e.g.
        # IndexError / AssertionError) must not abort a run over tens of thousands of rows.
        try:
            prompter = build_prompter(sampler_name, fragments, sampler, args.n_samples, prompter_seed)
            sampled = prompter.sample()
        except Exception as err:
            skip_counts[f"generation_error:{type(err).__name__}"] += 1
            logfp.write(
                f"generation failed: target={smiles}, sampler={sampler_name}, "
                f"fragments={'.'.join(fragments)}, error={type(err).__name__}: {err}",
                suppress_std_out=True,
            )
            continue
        sampler_counts[sampler_name] += 1
        sources.append(".".join(fragments))
        targets.append(smiles)
        sampler_names.append(sampler_name)
        predictions.append(to_prediction_row(sampled, args.n_samples))
    elapsed_sec = round(time.perf_counter() - start, 3)

    # evaluation_func.loadGenSmiles concatenates test.source and predictions.csv by row number,
    # so the three files must hold the successfully generated rows only, in the same order.
    # test.source / test.target are therefore written after generation, not before.
    assert len(sources) == len(targets) == len(sampler_names) == len(predictions), "source/target/prediction rows are misaligned"
    save_file("".join(f"{source}\n" for source in sources), f"{dataset_dir}/test.source")
    save_file("".join(f"{target}\n" for target in targets), f"{dataset_dir}/test.target")

    # Save predictions in the T5Chem column layout (target, prediction_1..N) plus the sampler
    # the row was routed to. evaluation_func.py addresses every column by name, so the extra
    # column is carried along without disturbing the evaluation.
    columns = [f"prediction_{i + 1}" for i in range(args.n_samples)]
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df.insert(0, "sampler", sampler_names)
    predictions_df.insert(0, "target", targets)
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    logfp.write(format_run_summary(skip_counts, sampler_counts, len(test_smiles), len(predictions)))
    logfp.write(f"elapsed_sec: {elapsed_sec}, sec_per_molecule: {round(elapsed_sec / len(predictions), 3) if predictions else None}")
    print(f"Saved SMILES predictions to: {output_dir}/predictions.csv")
    print("Generation completed!")
