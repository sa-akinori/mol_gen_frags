"""Train a GPT2 prior on plain SMILES for the PromptSMILES baseline.

The model learns the unconditional language model ``p(SMILES)``. Each training
sequence is formatted as ``<bos> SMILES <eos>`` and **every** token contributes to
the loss (unlike ``train_gpt.py``, no prompt is masked with ``-100``): PromptSMILES
supplies its prompt only at inference time, so the prior must be a plain SMILES
language model.

No data augmentation is applied: one molecule yields exactly one sequence, which
keeps the corpus size comparable with the RFFMG and SAFE datasets. ``--randomize_smiles``
only changes *how* each molecule is written (random root atom instead of the canonical
order); the number of sequences stays the same.

Two modes are supported:
    - ``finetuning``: initialize from the pretrained ``entropy/gpt2_zinc_87m`` weights.
    - ``from_scratch``: same config/tokenizer as ``entropy/gpt2_zinc_87m`` but random weights.
"""

import argparse
import os
import random
from pathlib import Path

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase, Trainer, TrainingArguments

from func.utility import BASEPATH, set_seed

def read_lines(path: Path) -> list[str]:
    """Read a newline-separated text file into a list of stripped lines.

    Args:
        path: Path to a ``.smi`` file (one SMILES per line).

    Returns:
        List of non-empty lines with trailing whitespace removed.
    """
    with path.open(encoding="utf-8") as f:
        return [line.rstrip() for line in f if line.strip()]


def to_training_smiles(smiles: str, randomize: bool, rng: random.Random) -> str:
    """Validate one SMILES and return the string that is fed to the model.

    Args:
        smiles: SMILES read from the corpus file.
        randomize: If True, rewrite the molecule starting from a randomly chosen root
            atom instead of using the canonical atom order. One input SMILES always
            yields one output string, so the corpus size never changes.
        rng: Random generator used to pick the root atom (only used when ``randomize``).

    Returns:
        Canonical SMILES, or a randomly rooted SMILES when ``randomize`` is True.

    Raises:
        ValueError: If RDKit cannot parse ``smiles``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse the SMILES: {smiles!r}")
    root = rng.randrange(mol.GetNumAtoms()) if randomize else -1
    return Chem.MolToSmiles(mol, rootedAtAtom=root, canonical=not randomize)


class PromptSMILESDataset(Dataset):
    """Tokenized ``<bos> SMILES <eos>`` sequences for unconditional LM training.

    Each item is a dict with the single key ``input_ids`` (``list[int]``); the labels are
    built by the collator because every token contributes to the loss.

    Sequences are not truncated: if any example exceeds ``max_length`` after adding
    the bos/eos tokens, a ``ValueError`` is raised instead of silently truncating.
    """

    def __init__(
        self,
        smiles_list: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        self.examples: list[dict[str, list[int]]] = []
        for idx, smiles in enumerate(smiles_list):
            token_ids = tokenizer(smiles, add_special_tokens=False)["input_ids"]
            input_ids = [bos_id] + token_ids + [eos_id]
            if len(input_ids) > max_length:
                raise ValueError(f"Example {idx} has length {len(input_ids)} exceeding max_length {max_length}.")
            self.examples.append({"input_ids": input_ids})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


class DataCollatorForCausalLM:
    """Right-pad ``input_ids`` and derive ``labels`` for unconditional LM training.

    ``input_ids`` are padded with ``pad_token_id``; ``labels`` are a copy of ``input_ids``
    with ``-100`` on the padding positions, so the loss covers every real token
    (including the final ``<eos>``) and ignores the padding.
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
            labels.append(ids + [-100] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for GPT2 PromptSMILES prior training."""
    parser = argparse.ArgumentParser(description="Train a GPT2 prior on plain SMILES for PromptSMILES")
    parser.add_argument("--frag_method", type=str, default="brics", choices=["brics", "rc_cms"],
                        help="Fragmentation method that defined the data split (default: brics)")
    parser.add_argument("--mode", type=str, default="finetuning", choices=["finetuning", "from_scratch"],
                        help="Training mode (default: finetuning)")
    parser.add_argument("--pretrain", type=str, default="entropy/gpt2_zinc_87m",
                        help="Pretrained model/tokenizer id (default: entropy/gpt2_zinc_87m)")
    parser.add_argument("--randomize_smiles", action="store_true",
                        help="Write each molecule from a random root atom instead of the canonical order; the number of sequences is unchanged (default: False)")
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
        raise ValueError("Tokenizer must define both bos_token and eos_token for PromptSMILES prior training.")

    # Model: finetune from pretrained weights or reinitialize the same config.
    if args.mode == "finetuning":
        model = GPT2LMHeadModel.from_pretrained(args.pretrain)
    else:  # from_scratch
        config = GPT2Config.from_pretrained(args.pretrain)
        model = GPT2LMHeadModel(config)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data/output locations derived from frag_method and mode. The corpus is the plain-SMILES
    # view of the split shared with RFFMG and SAFE (written by src/make_datasets.py).
    data_dir = Path(f"{BASEPATH}/data/promptsmiles/{args.frag_method}/normal")
    output_dir = f"{BASEPATH}/models/promptsmiles/gpt/{args.mode}/{args.frag_method}"

    # Dedicated generator so the (optional) SMILES randomization is reproducible
    # without consuming the global random state used by the Trainer.
    rng = random.Random(args.seed)
    train_smiles = [to_training_smiles(smi, args.randomize_smiles, rng) for smi in read_lines(data_dir / "train.smi")]
    val_smiles   = [to_training_smiles(smi, args.randomize_smiles, rng) for smi in read_lines(data_dir / "val.smi")]
    print(f"train molecules: {len(train_smiles)}, val molecules: {len(val_smiles)}, randomize_smiles: {args.randomize_smiles}")

    # Datasets.
    train_dataset = PromptSMILESDataset(smiles_list=train_smiles, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset   = PromptSMILESDataset(smiles_list=val_smiles,   tokenizer=tokenizer, max_length=args.max_length)

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
