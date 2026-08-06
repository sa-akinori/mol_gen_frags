import argparse
import os

import datasets
from rdkit import Chem
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from func.utility import BASEPATH, LogFile, set_seed

RandomizeSMILES = lambda smiles, seed: Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles), numSmiles=1, randomSeed=seed)

def encode_smiles(
    example: dict[str, str],
    idx: int,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
) -> dict[str, list[int]]:
    """Rewrite one molecule from a random root atom and tokenize it as ``<bos> SMILES <eos>``.

    The prompts used at inference time are non-canonical, so the prior is trained on randomized
    writings of the same molecules; the number of sequences is unchanged (one per molecule).

    Args:
        example: Dataset row holding the ``smiles`` column.
        idx: Index of the row within its split.
        tokenizer: Tokenizer of the model being trained.
        seed: Base random seed of the run.

    Returns:
        Dict with the single key ``input_ids`` (``list[int]``).
    """
    # The seed is derived per row instead of from a shared generator: .map(num_proc=N) runs in
    # separate processes, so only a row-derived seed makes the randomization depend on --seed
    # alone and not on num_proc. The +1 keeps it non-zero, which RDKit reads as "unseeded".
    smiles = RandomizeSMILES(example["smiles"], seed + idx + 1)[0]
    token_ids = tokenizer(smiles, add_special_tokens=False)["input_ids"]
    return {"input_ids": [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]}


def build_lm_dataset(
    split: datasets.Dataset,
    split_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    seed: int,
    num_proc: int,
    logfp: LogFile,
) -> datasets.Dataset:
    """Turn one split of the PromptSMILES dataset into tokenized LM sequences.

    Sequences longer than ``max_length`` are dropped rather than truncated, because a truncated
    SMILES string is not a valid molecule. The kept/dropped counts are written to ``logfp``.

    Args:
        split: Split holding the ``smiles`` column.
        split_name: Name of the split, used in the progress bars and the drop count message.
        tokenizer: Tokenizer of the model being trained.
        max_length: Maximum sequence length, bos/eos included.
        seed: Base random seed of the run.
        num_proc: Number of worker processes for the map/filter passes.
        logfp: Log file the kept/dropped counts are written to.

    Returns:
        Dataset with the single column ``input_ids`` (``list[int]``).
    """
    encoded = split.map(
        encode_smiles,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=split.column_names,
        fn_kwargs={"tokenizer": tokenizer, "seed": seed},
        desc=f"tokenizing {split_name}",
    )
    kept = encoded.filter(
        lambda example: len(example["input_ids"]) <= max_length,
        num_proc=num_proc,
        desc=f"filtering {split_name}",
    )
    logfp.write(f"{split_name}: kept {len(kept)} sequences, dropped {len(encoded) - len(kept)} longer than max_length={max_length}")
    return kept


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for GPT2 PromptSMILES prior training."""
    parser = argparse.ArgumentParser(description="Train a GPT2 prior on plain SMILES for PromptSMILES")
    parser.add_argument("--frag_method", type=str, default="brics", choices=["brics", "rc_cms"], help="Fragmentation method that defined the data split (default: brics)")
    parser.add_argument("--mode", type=str, default="finetuning", choices=["finetuning", "from_scratch"], help="Training mode (default: finetuning)")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Per-device train/eval batch size (default: 32)")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps (default: 10000)")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Evaluation interval in steps (default: 5000)")
    parser.add_argument("--save_steps", type=int, default=5000, help="Checkpoint interval in steps (default: 5000)")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum number of checkpoints to keep (default: 5)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length (default: 256)")
    parser.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Worker processes used to tokenize the dataset (default: number of CPUs - 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy (default: steps)")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy (default: steps)")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    set_seed(args.seed)

    # Model: finetune from pretrained weights or reinitialize the same config.
    pretrained_model = "entropy/gpt2_zinc_87m"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if args.mode == "finetuning":
        model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    else:  # from_scratch
        config = GPT2Config.from_pretrained(pretrained_model)
        model = GPT2LMHeadModel(config)
    model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = f"{BASEPATH}/models/promptsmiles/gpt/{args.mode}/{args.frag_method}"
    os.makedirs(output_dir, exist_ok=True)

    logfp = LogFile(f"{output_dir}/training_params.txt")
    logfp.write(f"args: {vars(args)}")
    logfp.write(f"pretrained_model: {pretrained_model}")

    # Datasets
    dataset = datasets.load_from_disk(f"{BASEPATH}/data/promptsmiles/{args.frag_method}/normal")
    train_dataset = build_lm_dataset(dataset["train"], "train", tokenizer, args.max_length, args.seed, args.num_proc, logfp)
    val_dataset   = build_lm_dataset(dataset["validation"], "validation", tokenizer, args.max_length, args.seed, args.num_proc, logfp)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
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
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    )
    trainer.train()

    best_model_dir = f"{output_dir}/best_model"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
