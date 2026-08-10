import argparse
import os
import random
from collections import Counter

import datasets
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from func.fragment_for_fraggpt import assemble_fragments_with_reason, label_attachment_points
from func.utility import BASEPATH, INVALID_SMILES, LogFile, set_seed

def log_line(message: str, collected: list[str]) -> None:
    """Print a log line and keep it for the log file written at the end of the run.

    Args:
        message: Line to print.
        collected: List of the lines logged so far, appended to in place.
    """
    print(message)
    collected.append(message)


def format_failure_summary(
    assembly_reasons: Counter[str],
    n_test: int,
    n_candidates: int,
) -> str:
    """Render the per-reason failure statistics of a generation run.

    Args:
        assembly_reasons: Outcome of every generated candidate, per reason.
        n_test: Number of rows read from the test split.
        n_candidates: Number of candidates generated in total (rows x n_samples).

    Returns:
        Multi-line report holding the totals followed by one line per reason.
    """
    lines = [
        f"test rows: {n_test}",
        f"candidates: {n_candidates}",
        f"assembled candidates: {assembly_reasons['ok']}",
    ]
    lines += [f"  assembly ({reason}): {count}" for reason, count in sorted(assembly_reasons.items())]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for FragGPT generation."""
    parser = argparse.ArgumentParser(description="Generate molecules using a FragGPT (FU-SMILES) GPT2 model")
    parser.add_argument("--frag_method", type=str, default="brics", choices=["brics", "rc_cms"], help="Fragmentation method (default: brics)")
    parser.add_argument("--model_ver", type=str, default="finetuning", choices=["finetuning", "from_scratch"], help="Model version (default: finetuning)")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate per molecule (default: 50)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length (default: 256)")
    parser.add_argument("--num_beams", type=int, default=50, help="Number of beams, used by the beam gen_method (default: 50)")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size, used by the beam gen_method only; sampling generates one row at a time (default: 24)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--gen_method", type=str, default="beam", choices=["beam", "sampling"], help="Decoding scheme: beam search or multinomial sampling (default: beam)")
    parser.add_argument("--additional_path", type=str, default="normal", help="Additional path segment to append to the output dir (default: empty string)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)
    
    # Setting up paths
    model_path = f"{BASEPATH}/models/fraggpt/gpt/{args.model_ver}/{args.frag_method}/best_model"
    dataset_dir = f"{BASEPATH}/data/fraggpt/{args.frag_method}/{args.additional_path}"
    output_dir = f"{BASEPATH}/results/fraggpt/gpt/{args.model_ver}/{args.frag_method}/{args.gen_method}/{args.additional_path}"
    os.makedirs(output_dir, exist_ok=True)

    # Setting up logging
    log_lines: list[str] = []
    log_line(f"args: {vars(args)}", log_lines)
    log_line(f"gen_method: {args.gen_method}, num_beams: {args.num_beams}", log_lines)
    log_line(f"model_path: {model_path}", log_lines)

    # Loading the test dataset
    test_dataset = datasets.load_from_disk(dataset_dir)["test"]
    test_smiles, test_fragment_sets = test_dataset["smiles"], test_dataset["pass_fragments"]

    # Building prompts from the fragment sets
    prompts: list[str] = []
    for idx, fragment_set in enumerate(tqdm(test_fragment_sets, desc="building prompts")):
        rng = random.Random(args.random_seed + idx)
        fragments = [fragment for fragment in fragment_set.split(".") if fragment]
        fragments = label_attachment_points(fragments, rng)
        rng.shuffle(fragments)
        prompt = '.'.join(fragments) + '.'
        prompts.append(prompt)
    log_line(f"test rows: {len(test_smiles)}, prompted rows: {len(prompts)}", log_lines)

    # Loading the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    predictions: list[list[str]] = [[INVALID_SMILES] * args.n_samples for _ in prompts]
    assembly_reasons: Counter[str] = Counter()
    decode_params = ({"do_sample": True} if args.gen_method == "sampling"
                     else {"do_sample": False, "num_beams": args.num_beams, "early_stopping": True})
    batch_size = 1 if args.gen_method == "sampling" else args.batch_size

    for start in tqdm(range(0, len(prompts), batch_size), desc="fraggpt generation"):
        torch.manual_seed(args.random_seed + start)
        batch_prompts = prompts[start:start + batch_size]
        encoded    = tokenizer(batch_prompts, add_special_tokens=False)["input_ids"]
        prompt_ids = [[bos_id] + ids for ids in encoded]
        prompt_len = max(len(ids) for ids in prompt_ids)
        input_ids  = torch.tensor([[pad_id] * (prompt_len - len(ids)) + ids for ids in prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([[0] * (prompt_len - len(ids)) + [1] * len(ids) for ids in prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=args.n_samples,
                max_length=args.max_length,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **decode_params,
            )

        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for position, batch_prompt in enumerate(batch_prompts):
            group = completions[position * args.n_samples:(position + 1) * args.n_samples]
            for sample_idx, completion in enumerate(group):
                smiles, reason = assemble_fragments_with_reason(batch_prompt + completion)
                assembly_reasons[reason] += 1
                predictions[start + position][sample_idx] = smiles or INVALID_SMILES

    # The evaluation reads the prompt from the fragment column instead of joining by row number.
    assert len(test_fragment_sets) == len(test_smiles) == len(predictions), "fragment/target/prediction rows are misaligned"

    columns = [f"prediction_{i + 1}" for i in range(args.n_samples)]
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df.insert(0, "target", test_smiles)
    predictions_df.insert(0, "fragment", test_fragment_sets)
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    # n_candidates comes from the test split, not from the reasons, so that the totals disagree
    # when a candidate was not accounted for.
    log_line(format_failure_summary(assembly_reasons, len(test_smiles), len(test_smiles) * args.n_samples), log_lines)
    with LogFile(f"{output_dir}/generation_params.txt") as logfp:
        for line in log_lines:
            logfp.write(line, suppress_std_out=True)
    print(f"Saved SMILES predictions to: {output_dir}/predictions.csv")
    print("Generation completed!")


if __name__ == "__main__":
    main()
