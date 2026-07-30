"""Generate molecules with a trained FragGPT (FU-SMILES) GPT2 model via beam search.

FragGPT turns an unconditional FU-SMILES language model into a fragment-conditioned generator
at *inference* time. Every attachment point of the prompted fragment set gets its own fresh
label, the fragments are written in random order and the string is closed with the separator:

    ``<bos> [1*]c1ccccc1.[2*]CCO.``

The trailing ``.`` is the signal "write the next fragment". The model completes the FU-SMILES
string; the completion is split at ``.`` again and the resulting fragments are assembled with
the prompted ones by matching the ``[i*]`` labels (see ``func.fragment_for_fraggpt``).

The prompted fragments are read from the ``pass_fragments`` column of the shared test split
(``data/safe/{frag_method}/normal``, test split), and the reference molecule is the ``smiles``
column of the same row -- exactly what SAFE (``generation_safe_func.py``) and PromptSMILES
(``gen_promptsmiles.py``) prompt with, and the isotope-stripped form of the RFFMG
``test.source``. The fragment sets are deliberately *not* recomputed here.

Outputs (read unchanged by ``src/evaluation.py``):
    - ``data/fraggpt/{frag_method}/normal/test.source``: fragment set per test molecule.
    - ``data/fraggpt/{frag_method}/normal/test.target``: the test molecule itself.
    - ``results/fraggpt/gpt/{model_ver}/{frag_method}/beam/normal/predictions.csv``:
      columns ``target``, ``prediction_1`` .. ``prediction_N`` (T5Chem layout).

All three files hold **one row per row of the test split**, in the test-split order. Rows are
never dropped: a prompt that cannot be built or a candidate that cannot be assembled is
written as an empty string, which the evaluation pipeline counts as invalid. This keeps the
row numbers aligned with RFFMG, SAFE and PromptSMILES, so the four methods can be compared on
identical molecules. Every failure is counted per reason and reported in
``generation_params.txt``.

Note on the exact-match ceiling: generation itself is unaffected by the following -- every
prompted row yields molecules and those molecules contain all of the prompted fragments, so
validity, uniqueness, novelty and the fragment-inclusion rate are not touched. Only the
exact-match metric (top-k accuracy against the reference molecule) has a ceiling. FU-SMILES
expresses a bond by writing the same label twice, so bonding two *prompted* fragments directly
would need a ``[1*][2*]`` fragment, i.e. one made of two dummy atoms alone. A BRICS
decomposition never produces such a fragment, hence it is absent from the training
distribution, and FragGPT therefore always places at least one generated fragment between the
prompted ones. In brics, 82.5% of the test rows have two prompted fragments bonded directly
*in their reference molecule*; on those rows no candidate can equal the reference, which caps
top-k accuracy at 17.5%. The reference molecule is never used during generation -- it only
enters the picture in ``func.evaluation_func.calculateTopKAccuracy``. This is a property of the
method, not a defect of this script.
"""

import argparse
import os
import random
from collections import Counter

import datasets
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

from func.fragment_for_fraggpt import (
    ASSEMBLY_OK,
    FRAGMENT_SEPARATOR,
    assemble_fragments_with_reason,
    renumber_open_attachments,
    split_fragments,
)
from func.utility import BASEPATH, LogFile

# Beam search is the decoding scheme RFFMG and SAFE are evaluated with.
GEN_METHOD = "beam"


def save_file(target: str, save_path: str) -> None:
    """Write a string to a UTF-8 text file with LF newlines.

    Args:
        target: Text to write.
        save_path: Destination file path.
    """
    with open(save_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(target)


def read_test_split(frag_method: str) -> tuple[list[str], list[str]]:
    """Read the reference molecules and their prompted fragment sets from the shared split.

    The split is the one written by ``src/make_datasets.py`` and used by SAFE and PromptSMILES
    (both read the same ``pass_fragments`` column) and, in its isotope-stripped form, by RFFMG
    (``test.source``).

    Args:
        frag_method: Fragmentation method, either ``brics`` or ``rc_cms``.

    Returns:
        Pair ``(smiles, pass_fragments)`` of equally long lists, aligned row by row: the
        reference molecule and its dot-separated fragment set.
    """
    test_dataset = datasets.load_from_disk(f"{BASEPATH}/data/safe/{frag_method}/normal")["test"]
    return test_dataset["smiles"], test_dataset["pass_fragments"]


def build_prompt(fragment_set: str, rng: random.Random) -> tuple[list[str], str]:
    """Turn a fragment set into a FragGPT inference prompt.

    Every bare ``*`` is given its own label ``1..k`` and the fragments are shuffled, so the
    prompt shows the model ``k`` open attachment points in an arbitrary order and an arbitrary
    numbering -- the two invariances the training augmentation taught.

    Args:
        fragment_set: Dot-separated fragment SMILES with unlabeled ``*`` attachment points.
        rng: Random generator seeded per row, so the prompt is reproducible.

    Returns:
        Pair ``(fragments, prompt)``: the labeled fragments in prompt order and the prompt
        string, which ends with the separator that asks for the next fragment.

    Raises:
        ValueError: If RDKit cannot parse one of the fragments.
    """
    fragments = renumber_open_attachments(split_fragments(fragment_set), rng)
    rng.shuffle(fragments)
    return fragments, FRAGMENT_SEPARATOR.join(fragments) + FRAGMENT_SEPARATOR


def format_failure_summary(
    prompt_failures: Counter[str],
    assembly_reasons: Counter[str],
    n_test: int,
    n_candidates: int,
) -> str:
    """Render the per-reason failure statistics of a generation run.

    Args:
        prompt_failures: Number of rows whose prompt could not be built, per reason.
        assembly_reasons: Outcome of every generated candidate, per reason
            (:data:`ASSEMBLY_OK` for the assembled ones).
        n_test: Number of rows read from the test split.
        n_candidates: Number of candidates generated in total (rows x n_samples).

    Returns:
        Multi-line report holding the totals followed by one line per reason.
    """
    lines = [
        f"test rows: {n_test}",
        f"rows without a prompt: {sum(prompt_failures.values())}",
        f"candidates: {n_candidates}",
        f"assembled candidates: {assembly_reasons[ASSEMBLY_OK]}",
    ]
    lines += [f"  prompt failure ({reason}): {count}" for reason, count in sorted(prompt_failures.items())]
    lines += [f"  assembly ({reason}): {count}" for reason, count in sorted(assembly_reasons.items())]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for FragGPT generation."""
    parser = argparse.ArgumentParser(description="Generate molecules using a FragGPT (FU-SMILES) GPT2 model")
    parser.add_argument("--frag_method", type=str, default="brics", choices=["brics", "rc_cms"],
                        help="Fragmentation method (default: brics)")
    parser.add_argument("--model_ver", type=str, default="finetuning", choices=["finetuning", "from_scratch"],
                        help="Model version (default: finetuning)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to generate per molecule (default: 50)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length (default: 256)")
    parser.add_argument("--num_beams", type=int, default=50,
                        help="Number of beams for beam search (default: 50)")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Batch size (default: 24)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)

    model_path = f"{BASEPATH}/models/fraggpt/gpt/{args.model_ver}/{args.frag_method}/best_model"
    dataset_dir = f"{BASEPATH}/data/fraggpt/{args.frag_method}/normal"
    output_dir = f"{BASEPATH}/results/fraggpt/gpt/{args.model_ver}/{args.frag_method}/{GEN_METHOD}/normal"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logfp = LogFile(f"{output_dir}/generation_params.txt")
    logfp.write(f"args: {vars(args)}")
    logfp.write(f"gen_method: {GEN_METHOD}, model_path: {model_path}")

    # Prompts come from the fragment sets RFFMG, SAFE and PromptSMILES generate from, never
    # from a fresh fragmentation of the test molecules (see module docstring).
    test_smiles, test_fragment_sets = read_test_split(args.frag_method)

    # One prompt per test row; None marks a row whose prompt could not be built. Such rows keep
    # their place so that predictions.csv, test.source and test.target stay aligned.
    prompt_failures: Counter[str] = Counter()
    prompt_fragments: list[list[str]] = []
    prompts: list[str | None] = []
    for idx, fragment_set in enumerate(tqdm(test_fragment_sets, desc="building prompts")):
        # A per-row seed keeps the prompt of a row independent of the rows processed before it.
        rng = random.Random(args.random_seed + idx)
        try:
            fragments, prompt = build_prompt(fragment_set, rng)
        except ValueError as err:
            prompt_failures["unparsable_fragment"] += 1
            logfp.write(f"prompt failed: fragments={fragment_set}, error={err}", suppress_std_out=True)
            fragments, prompt = [], None
        prompt_fragments.append(fragments)
        prompts.append(prompt)
    logfp.write(f"test rows: {len(test_smiles)}, prompted rows: {sum(prompt is not None for prompt in prompts)}")

    # Load the trained model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    # Failed rows and failed candidates alike stay in the table as empty strings.
    predictions: list[list[str]] = [[""] * args.n_samples for _ in prompts]
    assembly_reasons: Counter[str] = Counter()
    prompted_rows = [idx for idx, prompt in enumerate(prompts) if prompt is not None]
    for start in tqdm(range(0, len(prompted_rows), args.batch_size), desc="fraggpt generation"):
        batch_rows = prompted_rows[start:start + args.batch_size]

        # Decoder-only batched generation requires **left** padding: every row then starts
        # generating at the same column, so the completion of row i is outputs[i, prompt_len:].
        encoded = tokenizer([prompts[idx] for idx in batch_rows], add_special_tokens=False)["input_ids"]
        prompt_ids = [[bos_id] + ids for ids in encoded]
        prompt_len = max(len(ids) for ids in prompt_ids)
        input_ids = torch.tensor([[pad_id] * (prompt_len - len(ids)) + ids for ids in prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([[0] * (prompt_len - len(ids)) + [1] * len(ids) for ids in prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=args.num_beams,
                num_return_sequences=args.n_samples,
                max_length=args.max_length,
                early_stopping=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for position, row_idx in enumerate(batch_rows):
            group = completions[position * args.n_samples:(position + 1) * args.n_samples]
            for sample_idx, completion in enumerate(group):
                # The prompt ends with the separator, so the completion starts a new fragment.
                candidate = prompt_fragments[row_idx] + split_fragments(completion)
                smiles, reason = assemble_fragments_with_reason(candidate)
                assembly_reasons[reason] += 1
                predictions[row_idx][sample_idx] = smiles or ""

    # evaluation_func.loadGenSmiles concatenates test.source and predictions.csv by row number,
    # so the three files must hold every test row, in the test-split order.
    assert len(test_fragment_sets) == len(test_smiles) == len(predictions), "source/target/prediction rows are misaligned"
    save_file("".join(f"{source}\n" for source in test_fragment_sets), f"{dataset_dir}/test.source")
    save_file("".join(f"{target}\n" for target in test_smiles), f"{dataset_dir}/test.target")

    # Save predictions in the T5Chem column layout: target, prediction_1..N.
    columns = [f"prediction_{i + 1}" for i in range(args.n_samples)]
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df.insert(0, "target", test_smiles)
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    logfp.write(format_failure_summary(prompt_failures, assembly_reasons, len(test_smiles), sum(assembly_reasons.values())))
    print(f"Saved SMILES predictions to: {output_dir}/predictions.csv")
    print("Generation completed!")


if __name__ == "__main__":
    main()
