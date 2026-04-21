# Representation for Flexible Fragment-Controlled Molecular Generation (RFFMG): A Framework for Versatile Substructure-Conditioned Molecular Design

[Japanese version (日本語版)](README_ja.md)

```bash
git clone https://github.com/sa-akinori/rffmg_molecular_design.git
cd rffmg_molecular_design
```

## Tutorial

A tutorial for molecular generation is available in [`tutorial.ipynb`](tutorial.ipynb).
It provides step-by-step instructions for extracting fragments from arbitrary SMILES and generating new molecules using pre-trained models.

## Two Conda Environments Required
### T5Chem
```bash
conda create -n t5chem python=3.12
conda activate t5chem
pip install -r requirements/t5chem_requirements.txt
pip install -e .
```
### SAFE
```bash
conda create -n safe python=3.12
conda activate safe
pip install -r requirements/safe_requirements.txt
pip install -e .
```

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

### Disable wandb logging (t5chem/run_trainer.py)
```python
# report_to="wandb",  # enable logging to W&B
report_to='none',
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
$ mkdir -p models/t5chem/pretrained
$ wget -P models/t5chem/pretrained https://zenodo.org/records/14280768/files/simple_pretrain.tar.bz2
$ tar -xjvf models/t5chem/pretrained/simple_pretrain.tar.bz2 --strip-components=3 -C models/t5chem/pretrained/
```

### SAFE
```bash
$ mkdir -p models/safe_gpt/pretrained
$ git clone https://huggingface.co/datamol-io/safe-gpt/ models/safe_gpt/pretrained/
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

### Model Training
```bash
# 1. Train RFFMG model with T5Chem
$ conda activate t5chem
$ t5chem train --data_dir data/rffmg/rc_cms/normal --output_dir models/t5chem/trained/rffmg/rc_cms --pretrain models/t5chem/pretrained --task_type product --num_epoch 50
# Adjust the rc_cms part and output_dir as needed. Use --pretrain '' for training without a pre-trained model.
# The from_scratch models in this study were trained with --pretrain ''.

# 2. Fine-tune SAFE-GPT
$ conda activate safe
$ bash src/train_model/run_safe.sh
# Due to the large number of arguments, they are specified in the .sh file.
# Adjust the rc_cms part and output_dir in the .sh file as needed. Use --pretrain '' for training without a pre-trained model.
# The from_scratch models in this study were trained with --pretrain ''.
```

### Molecular Generation
```bash
# Generate molecules with T5Chem model
$ bash src/gen_mols/gen_t5chem.sh

# Generate molecules with SAFE-GPT model
$ bash src/gen_mols/gen_safe.sh
```

### Evaluation of Generated Molecules
```bash
$ conda activate safe
$ python src/evaluation.py --model_name t5chem --model_ver trained --frag_method rc_cms --additional_path normal
```

If you need the curated ChEMBL dataset used in this study, please feel free to contact us at [sato.akinori@naist.ac.jp] or [miyao@dsc.naist.jp].
