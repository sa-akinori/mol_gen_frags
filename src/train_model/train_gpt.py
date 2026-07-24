"""Train a GPT2 decoder model for RFFMG fragment-to-molecule generation.

The model learns conditional generation ``p(target | source)``. Each training
sequence is formatted as ``<bos> source ">>" target <eos>`` (the original RFFMG
sentence wrapped with bos/eos). The prompt part (``<bos> source ">>"``) is masked
out of the loss with ``-100`` labels so the loss is computed only on the ``target``
tokens and the final ``<eos>``.

Two modes are supported:
    - ``finetuning``: initialize from the pretrained ``entropy/gpt2_zinc_87m`` weights.
    - ``from_scratch``: same config/tokenizer as ``entropy/gpt2_zinc_87m`` but random weights.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase, Trainer, TrainingArguments

from func.utility import BASEPATH, set_seed

def read_lines(path: Path) -> list[str]:
    """Read a newline-separated text file into a list of stripped lines.

    Args:
        path: Path to a ``.source`` / ``.target`` file (one example per line).

    Returns:
        List of lines with trailing whitespace removed.
    """
    with path.open(encoding="utf-8") as f:
        return [line.rstrip() for line in f]


class RFFMGDataset(Dataset):
    """Tokenized ``source>>target`` sequences with prompt-masked labels.

    Each item is a dict with keys ``input_ids`` and ``labels`` (both ``list[int]``).
    The prompt part ``<bos> source ">>"`` is masked with ``-100`` in ``labels`` so the
    loss is computed only on the ``target`` tokens and the final ``<eos>``.

    Sequences are not truncated: if any example exceeds ``max_length`` after adding
    the bos/eos tokens, a ``ValueError`` is raised instead of silently truncating.
    """

    def __init__(
        self,
        sources: list[str],
        targets: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        self.examples: list[dict[str, list[int]]] = []
        for idx, (source, target) in enumerate(zip(sources, targets)):
            prompt_ids = tokenizer(source + ">>", add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            input_ids = [bos_id] + prompt_ids + target_ids + [eos_id]
            if len(input_ids) > max_length:
                raise ValueError(f"Example {idx} has length {len(input_ids)} exceeding max_length {max_length}.")
            labels = [-100] * (1 + len(prompt_ids)) + target_ids + [eos_id]
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


class DataCollatorForCausalLM:
    """Right-pad ``input_ids``/``labels`` to the longest sequence in a batch.

    ``input_ids`` are padded with ``pad_token_id`` and ``labels`` with ``-100`` so the
    padding positions do not contribute to the loss.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            n_pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * n_pad)
            attention_mask.append([1] * len(ids) + [0] * n_pad)
            labels.append(f["labels"] + [-100] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for GPT2 RFFMG training."""
    parser = argparse.ArgumentParser(description="Train a GPT2 model for RFFMG generation")
    parser.add_argument("--frag_method", type=str, default="brics", choices=["brics", "rc_cms"],
                        help="Fragmentation method (default: brics)")
    parser.add_argument("--mode", type=str, default="finetuning", choices=["finetuning", "from_scratch"],
                        help="Training mode (default: finetuning)")
    parser.add_argument("--sampling_num", type=int, default=10, choices=[5, 10],
                        help="Fragment-sampling multiplier N, selecting the data/rffmg/<frag>/<N>times_sampling slice (default: 10)")
    parser.add_argument("--pretrain", type=str, default="entropy/gpt2_zinc_87m",
                        help=f"Pretrained model/tokenizer id (default: entropy/gpt2_zinc_87m)")
    parser.add_argument("--num_train_epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                        help="Per-device train/eval batch size (default: 32)")
    parser.add_argument("--warmup_steps", type=int, default=10000,
                        help="Warmup steps (default: 10000)")
    parser.add_argument("--eval_steps", type=int, default=5000,
                        help="Evaluation interval in steps (default: 5000)")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Checkpoint interval in steps (default: 5000)")
    parser.add_argument("--save_total_limit", type=int, default=5,
                        help="Maximum number of checkpoints to keep (default: 5)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Run wandb offline unless explicitly overridden by the environment.
    os.environ.setdefault("WANDB_MODE", "offline")

    # Seed everything for reproducibility.
    set_seed(args.seed)

    # Tokenizer is shared by both modes (always from the ZINC-pretrained model).
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define both bos_token and eos_token for RFFMG training.")

    # Model: finetune from pretrained weights or reinitialize the same config.
    if args.mode == "finetuning":
        model = GPT2LMHeadModel.from_pretrained(args.pretrain)
    else:  # from_scratch
        config = GPT2Config.from_pretrained(args.pretrain)
        model = GPT2LMHeadModel(config)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data/output locations derived from frag_method, sampling, and mode.
    sampling = f"{args.sampling_num}times_sampling"
    data_dir = Path(f"{BASEPATH}/data/rffmg/{args.frag_method}/{sampling}/normal")
    output_dir = f"{BASEPATH}/models/rffmg/gpt/{args.mode}/{args.frag_method}/{sampling}"

    # Datasets.
    train_dataset = RFFMGDataset(sources=read_lines(data_dir / "train.source"), targets=read_lines(data_dir / "train.target"), tokenizer=tokenizer, max_length=args.max_length)
    val_dataset   = RFFMGDataset(sources=read_lines(data_dir / "val.source"),   targets=read_lines(data_dir / "val.target"),   tokenizer=tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForCausalLM(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    )
    trainer.train()

    best_model_dir = f"{output_dir}/best_model"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
