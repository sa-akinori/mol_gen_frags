"""Generate molecules from an RFFMG-GPT model via beam search.

Loads a trained GPT2 model and, for every ``source`` fragment in a ``test.source``
file, produces ``n_samples`` candidate molecules with beam search. The predictions
are written to ``predictions.csv`` whose columns match the T5Chem output
(``target``, ``prediction_1`` .. ``prediction_N``) so the shared evaluation pipeline
can read them unchanged.

Each generation prompt is ``<bos> source ">>"``; the predicted molecule is the text
after the first ``">>"`` (special tokens stripped), mirroring the training format.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
from tqdm import tqdm

SEPARATOR = ">>"


def read_lines(path: Path) -> list[str]:
    """Read a newline-separated text file into a list of stripped lines.

    Args:
        path: Path to a ``.source`` / ``.target`` file (one example per line).

    Returns:
        List of lines with trailing whitespace removed.
    """
    with path.open(encoding="utf-8") as f:
        return [line.rstrip() for line in f]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for GPT2 RFFMG generation."""
    parser = argparse.ArgumentParser(description="Generate molecules using an RFFMG-GPT model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained GPT2 model directory")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the test.source file (one fragment per line)")
    parser.add_argument("--target_file", type=str, required=True,
                        help="Path to the test.target file (one molecule per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions.csv")
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

    # Load model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generation requires left padding.
    tokenizer.padding_side = "left"
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    sources = read_lines(Path(args.dataset_file))
    targets = read_lines(Path(args.target_file))

    predictions: list[list[str]] = []
    for start in tqdm(range(0, len(sources), args.batch_size), desc='prediction'):
        batch_sources = sources[start:start + args.batch_size]

        # Build left-padded ``<bos> source ">>"`` prompts.
        encoded = tokenizer([s + SEPARATOR for s in batch_sources], add_special_tokens=False)
        prompt_ids = [[bos_id] + ids for ids in encoded["input_ids"]]
        max_len = max(len(ids) for ids in prompt_ids)
        input_ids = [[pad_id] * (max_len - len(ids)) + ids for ids in prompt_ids]
        attention_mask = [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

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

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(len(batch_sources)):
            group = decoded[i * args.n_samples:(i + 1) * args.n_samples]
            preds = [text.split(SEPARATOR, 1)[1].strip() if SEPARATOR in text else "" for text in group]
            predictions.append(preds)

    # Save predictions in the T5Chem column layout: target, prediction_1..N.
    columns = [f"prediction_{i + 1}" for i in range(args.n_samples)]
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df.insert(0, "target", targets)

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_df.to_csv(f"{args.output_dir}/predictions.csv", index=False)
    print(f"Saved SMILES predictions to: {args.output_dir}/predictions.csv")
    print("Generation completed!")


if __name__ == "__main__":
    main()
