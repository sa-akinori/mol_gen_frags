# Representation for Flexible Fragment-Controlled Molecular Generation (RFFMG): A Framework for Versatile Substructure-Conditioned Molecular Design

[Japanese version (日本語版)](README_ja.md)

```bash
git clone https://github.com/sa-akinori/rffmg_molecular_design.git
cd rffmg_molecular_design
```

## Tutorial

A tutorial for molecular generation is available in [`tutorial.ipynb`](tutorial.ipynb).
It provides step-by-step instructions for extracting fragments from arbitrary SMILES and generating new molecules using pre-trained models.

## Four Conda Environments Required

One environment per method. `pip install -e .` installs the local `func` package and is required in
**every** environment.

### T5Chem (also used by RFFMG-GPT and by dataset construction)
```bash
conda create -n t5chem python=3.12.12
conda activate t5chem
pip install -r requirements/t5chem_requirements.txt
pip install -e .
```
### SAFE (also used by evaluation)
```bash
conda create -n safe python=3.12.12
conda activate safe
pip install -r requirements/safe_requirements.txt
pip install -e .
```
### PromptSMILES
```bash
conda create -n promptsmiles python=3.12.12
conda activate promptsmiles
pip install -r requirements/promptsmiles_requirements.txt
pip install -e .
```
### FragGPT
```bash
conda create -n fraggpt python=3.12.12
conda activate fraggpt
pip install -r requirements/fraggpt_requirements.txt
pip install -e .
```

| Method | Representation | Base model | Environment |
|---|---|---|---|
| RFFMG (T5Chem) | `fragments >> molecule` | T5 (~14.8M) | `t5chem` |
| RFFMG (GPT2) | `fragments >> molecule` | `entropy/gpt2_zinc_87m` (~87M) | `t5chem` |
| SAFE | SAFE string | safe-gpt (~88.8M) | `safe` |
| PromptSMILES | plain SMILES + inference-time prompting | `entropy/gpt2_zinc_87m` | `promptsmiles` |
| FragGPT | FU-SMILES (BRICS fragments with paired `[i*]` labels) | `entropy/gpt2_zinc_87m` | `fraggpt` |

The `run_*.sh` and `gen_fraggpt.sh` scripts activate their environment themselves, so they can be
launched from any shell.

## Modifications to Virtual Environments
## T5Chem
### Speed up training (t5chem/run_trainer.py)
```python
# compute_metrics = AccuracyMetrics
compute_metrics = None
```

### Clarify model save paths (t5chem/run_trainer.py)
```python
# tokenizer.save_vocabulary(args.output_dir)
# trainer.save_model(args.output_dir)
os.makedirs(f'{args.output_dir}/best_model/')
tokenizer.save_vocabulary(f'{args.output_dir}/best_model/')
trainer.save_model(f'{args.output_dir}/best_model/')
```

## SAFE
### Clarify model save paths (safe/trainer/cli.py)
```python
# trainer.save_model()
trainer.save_model(os.path.join(training_args.output_dir, "best_model"))

# tokenizer.save(os.path.join(training_args.output_dir, "tokenizer.json"))
tokenizer.save(os.path.join(training_args.output_dir, "best_model/tokenizer.json"))
```
### Add early stopping for faster training (safe/trainer/cli.py)
```python
trainer = SAFETrainer(
    model=model,
    tokenizer=None,  # we don't deal with the tokenizer at all, https://github.com/huggingface/tokenizers/issues/581 -_-
    train_dataset=train_dataset.shuffle(seed=(training_args.seed or 42)),
    eval_dataset=dataset.get(eval_dataset_key_name, None),
    args=training_args,
    prop_loss_coeff=model_args.prop_loss_coeff,
    compute_metrics=compute_metrics if training_args.do_eval else None,
    data_collator=data_collator,
    preprocess_logits_for_metrics=(
        preprocess_logits_for_metrics if training_args.do_eval else None
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)] #add
)
```
### Fix errors caused by transformers version (safe/trainer/trainer_utils.py & safe/tokenizer.py)
```python
# Error in safe/trainer/trainer_utils.py (line 19)
# def compute_loss(self, model, inputs, return_outputs=False):
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

# Error in safe/tokenizer.py (line 290)
# self.tokenizer.save_pretrained(*args, **kwargs)
self.tokenizer.save(*args, **kwargs)
```

## Preparing Pre-trained/Trained Models and Datasets
### Download trained models from Hugging Face
```bash
$ python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sato-akinori/FFMG', allow_patterns='models/*', local_dir='.')"
$ shopt -s globstar; for zip in models/**/*.zip; do unzip -o "$zip" -d "$(dirname "$zip")"; done
$ find models -name "*.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;
```

### T5Chem
Download and extract the pre-trained model.
```bash
$ mkdir -p models/rffmg/t5chem/pretrained
$ wget -P models/rffmg/t5chem/pretrained https://zenodo.org/records/14280768/files/simple_pretrain.tar.bz2
$ tar -xjvf models/rffmg/t5chem/pretrained/simple_pretrain.tar.bz2 --strip-components=3 -C models/rffmg/t5chem/pretrained/
```

### SAFE
```bash
$ mkdir -p models/safe/gpt/pretrained
$ git clone https://huggingface.co/datamol-io/safe-gpt/ models/safe/gpt/pretrained/
```

### Curated Dataset
Coming soon.

## Building Datasets
### First Step
```bash
$ conda activate t5chem
$ python src/curate_datasets.py
```

### Creating Datasets
```bash
# 1. Create RFFMG fragments
$ conda activate t5chem
$ python src/gen_frags/rffmg_frags.py --frag_method brics # choose brics or rc_cms

# 2. Create SAFE fragments
$ conda activate safe
$ python src/gen_frags/safe_frags.py --frag_method brics # choose brics or rc_cms

# 3. Create train, test, validation datasets
$ conda activate safe
$ python src/make_datasets.py --frag_method brics # choose brics or rc_cms
```

`make_datasets.py` writes the RFFMG dataset for every `--sampling_num`, but the SAFE, PromptSMILES
and FragGPT datasets only when `--sampling_num 5` (the default), because those three have no
`sampling_num` level and a different value would overwrite them with a different molecule split.
Run it once per fragmentation method with the default to obtain all five datasets:

| Dataset | Path | Content |
|---|---|---|
| RFFMG | `data/rffmg/{frag}/{N}times_sampling/normal/` | `train/val/test.source` + `.target` |
| SAFE | `data/safe/{frag}/normal` | HF `DatasetDict` (`smiles`, `full_safe`, `pass_safe`, `full_fragments`, `pass_fragments`) |
| PromptSMILES | `data/promptsmiles/{frag}/normal` | HF `DatasetDict` (`smiles`, `pass_fragments`) |
| FragGPT | `data/fraggpt/{frag}/normal` | HF `DatasetDict`; train/validation hold `full_fragments`, test holds `pass_fragments` |

All five share the same molecule split, and their test splits hold the same 20,000 molecules in the
same row order, so the methods can be compared row by row.

### Model Training
```bash
# 1. Train the RFFMG model (T5Chem or GPT2)
$ conda activate t5chem
$ bash src/train_model/run_rffmg.sh
# Set MODEL_NAME/MODE/FRAG_NAME inside the .sh.
#   MODEL_NAME="t5chem": T5Chem (runs `t5chem train`).
#   MODEL_NAME="gpt":    GPT2 (src/train_model/train_gpt.py). MODE="finetuning" starts from
#                        entropy/gpt2_zinc_87m; MODE="from_scratch" uses the same config with random weights.

# 2. Fine-tune SAFE-GPT
$ conda activate safe
$ bash src/train_model/run_safe.sh
# Due to the large number of arguments, they are specified in the .sh file.
# Adjust the rc_cms part and output_dir in the .sh file as needed. Use --pretrain '' for training without a pre-trained model.
# The from_scratch models in this study were trained with --pretrain ''.

# 3. Train the PromptSMILES prior (plain-SMILES language model)
$ bash src/train_model/run_promptsmiles.sh
# The .sh activates env_promptsmiles itself. Set FRAG_NAME/MODE at the top of the .sh.
# The prior is an unconditional SMILES language model; PromptSMILES supplies its prompt only at
# inference time. Each molecule is always rewritten from a random root atom, because the prompts
# seen at inference are non-canonical and start at an arbitrary atom. No augmentation is applied
# (one molecule = one sequence), and the randomization is drawn once when the dataset is built.

# 4. Train FragGPT (FU-SMILES language model)
$ bash src/train_model/run_fraggpt.sh
# The .sh activates env_fraggpt itself. Set FRAG_NAME/MODE at the top of the .sh.
# The model is an unconditional FU-SMILES language model. The attachment labels are relabeled by a
# random permutation and the fragments are shuffled (on by default; --no-augment disables it), which
# does not change the number of sequences.
```

`MODE="finetuning"` starts from `entropy/gpt2_zinc_87m`; `MODE="from_scratch"` uses the same config
with random weights. All four GPT2-based methods share the same hyperparameters (LR 1e-4, 50 epochs,
batch 32, warmup 10000, eval/save every 5000 steps, early stopping patience 15, seed 42), so the
comparison isolates the representation rather than the training budget.

### Molecular Generation

All four methods are prompted with the **same fragment sets** (the `pass_fragments` of the shared
test split, attachment points written as bare `*` with no connectivity information) and write
`predictions.csv` with the columns `target`, `prediction_1` .. `prediction_N`, so the shared
evaluation pipeline reads them unchanged.

```bash
# RFFMG (T5Chem or GPT2; set MODEL_NAME inside the .sh)
$ conda activate t5chem
$ bash src/gen_mols/gen_rffmg.sh

# SAFE-GPT
$ conda activate safe
$ bash src/gen_mols/gen_safe.sh

# PromptSMILES (scaffold decoration / fragment linking, chosen per row)
$ conda activate env_promptsmiles
$ bash src/gen_mols/gen_promptsmiles.sh
# Set FRAG_NAME/MODEL_VER/GEN_METHOD at the top of the .sh.
# GEN_METHOD="beam" matches RFFMG and SAFE; "sampling" is the multinomial scheme of the paper.

# FragGPT
$ bash src/gen_mols/gen_fraggpt.sh
# The .sh activates env_fraggpt itself. Set FRAG_NAME/MODEL_VER at the top of the .sh.
# Each attachment point of the prompted fragment set is given a fresh label, the model completes the
# FU-SMILES string, and the fragments are reassembled by matching the [i*] labels.
```

Results are written to `results/{repr}/{model}/{model_ver}/{frag}/{gen_method}/{additional_path}/`.

### Evaluation of Generated Molecules
```bash
$ conda activate safe
$ python src/evaluation.py --model_name t5chem --model_ver finetuning --frag_method rc_cms --additional_path normal
# --model_name: t5chem / gpt (RFFMG-GPT) / safe_gpt / promptsmiles / fraggpt
# --gen_method: beam / sampling (defaults to beam, except promptsmiles which defaults to sampling)
```

Run the generation step first: the evaluation joins `test.source` and `predictions.csv` by row
number, and PromptSMILES and FragGPT write their `test.source` / `test.target` at generation time.

If you need the curated ChEMBL dataset used in this study, please feel free to contact us at [sato.akinori@naist.ac.jp] or [miyao@dsc.naist.jp].
