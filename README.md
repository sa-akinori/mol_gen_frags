# Representation for Flexible Fragment-Controlled Molecular Generation (RFFMG): A Framework for Versatile Substructure-Conditioned Molecular Design
```bash
git clone https://github.com/sa-akinori/mol_gen_frags.git
cd mol_gen_frags
```

## 2つの仮想環境が必要
### T5Chem
```bash
conda create -n t5chem_copy python=3.12
conda activate t5chem_copy
pip install -r requirements/t5chem_requirements.txt
pip install -e .
```
### SAFE
```bash
conda create -n safe_copy python=3.12
conda activate safe_copy
pip install -r requirements/safe_requirements.txt
pip install -e .
```

## 仮想環境の変更点
## T5Chem
### 学習速度向上のための変更(t5chem/run_trainer.py)
```python
# compute_metrics = AccuracyMetrics
compute_metrics = None
```

### モデルの保存をわかりやすくするための変更(t5chem/run_trainer.py)
```python
# tokenizer.save_vocabulary(args.output_dir)
# trainer.save_model(args.output_dir)
os.makedirs(f'{args.output_dir}/best_model/')
tokenizer.save_vocabulary(f'{args.output_dir}/best_model/')
trainer.save_model(f'{args.output_dir}/best_model/')
```

### wandbを使わない場合の変更(t5chem/run_trainer.py)
```python
# report_to="wandb",  # enable logging to W&B
report_to='none',
```

## SAFE
### モデルの保存をわかりやすくするための変更(safe/trainer/cli.py)
```python
# trainer.save_model()
trainer.save_model(os.path.join(training_args.output_dir, "best_model"))

# tokenizer.save(os.path.join(training_args.output_dir, "tokenizer.json"))
tokenizer.save(os.path.join(training_args.output_dir, "best_model/tokenizer.json"))
```
### 学習高速化のための追加(safe/trainer/cli.py)
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
### transformersのバージョンによってエラーが出るので修正してください。(safe/trainer/trainer_utils.py & safe/tokenizer.py)
```python
# safe/trainer/trainer_utils.py(19行目)におけるエラー
# def compute_loss(self, model, inputs, return_outputs=False):
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

# safe/tokenizer.py(290行目)におけるエラー
# self.tokenizer.save_pretrained(*args, **kwargs)
self.tokenizer.save(*args, **kwargs)
```

## Pre-trained/trainedモデル・データセットの準備
### 本研究の学習済みモデルをHugging Faceからmodelsフォルダーをダウンロード
```bash
$ python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sato-akinori/FFMG', allow_patterns='models/*', local_dir='.')"
$ shopt -s globstar; for zip in models/**/*.zip; do unzip -o "$zip" -d "$(dirname "$zip")"; done
$ find models -name "*.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;
```

### T5Chem
事前学習モデルをダウンロードして解凍する。
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

### curated datasetの準備
準備中

## データセットの構築
### 最初のステップ
```bash
$ conda activate t5chem_copy
$ python src/curate_datasets.py
```

### データセットの作成
```bash
# 1. rffmgフラグメントの作成
$ conda activate t5chem_copy
$ python src/gen_frags/rffmg_frags.py --frag_method brics # chose brics or rc_cms

# 2. safeフラグメントの作成
$ conda activate safe_copy
$ python src/gen_frags/safe_frags.py --frag_method brics # chose brics or rc_cms

# 3. train, test, validationデータセットの作成
$ conda activate safe_copy
$ python src/make_datasets.py --frag_method brics # chose brics or rc_cms
```

### モデルの学習
```bash
# 1. t5chemを用いたrffmgモデルの学習
$ conda activate t5chem_copy
$ t5chem train --data_dir data/rffmg/rc_cms/normal --output_dir models/t5chem/trained/rffmg/rc_cms --pretrain models/t5chem/pretrained --task_type product --num_epoch 50
# rc_cmsの部分、output_dirは適切に変更してください。また、--pretrain '' とすると事前学習済みモデルなしの学習が行われます。
# 本研究のfrom_scratchモデルは--pretrain ''とした場合の結果です。

# 2. safe-gptのファインチューニング
$ conda activate safe_copy
$ bash src/train_model/run_safe.sh
# 引数が非常に多いため.shファイルに記載済み
# .shファイル中のrc_cmsの部分、output_dirは適切に変更してください。また、--pretrain '' とすると事前学習済みモデルなしの学習が行われます。
# 本研究のfrom_scratchモデルは--pretrain ''とした場合の結果です。
```

### 分子の生成
```bash
# t5chemモデルを用いた分子生成
$ bash src/gen_mols/gen_t5chem.sh

# safe-gptモデルを用いた分子生成
$ bash src/gen_mols/gen_safe.sh
```

### 生成分子の評価
```bash
$ conda activate safe_copy
$ python src/evaluation.py --model_name t5chem --model_ver trained --frag_method rc_cms --additional_path normal
```

If you need the curated ChEMBL dataset used in this study, please feel free to contact us at [sato.akinori@naist.ac.jp] or [miyao@dsc.naist.jp].
