# Representation for Flexible Fragment-Controlled Molecular Generation (RFFMG): A Framework for Versatile Substructure-Conditioned Molecular Design
```bash
git clone https://github.com/sa-akinori/rffmg_molecular_design.git
cd rffmg_molecular_design
```

## チュートリアル

分子生成のチュートリアルは [`tutorial.ipynb`](tutorial.ipynb) に用意されています。
学習済みモデルを用いて、任意のSMILESからフラグメントを抽出し、新しい分子を生成する手順をステップごとに解説しています。

## 4つの仮想環境が必要

手法ごとに1つの環境を用意します。`pip install -e .`（ローカルの `func` パッケージ）は
**すべての環境で必要**です。

### T5Chem（RFFMG-GPT の学習とデータセット構築にも使用）
```bash
conda create -n t5chem python=3.12.12
conda activate t5chem
pip install -r requirements/t5chem_requirements.txt
pip install -e .
```
### SAFE（生成分子の評価にも使用）
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/promptsmiles_requirements.txt
pip install -e .
```
### FragGPT
```bash
conda create -n fraggpt python=3.12.12
conda activate fraggpt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/fraggpt_requirements.txt
pip install -e .
```

| 手法 | 表現 | ベースモデル | 環境 |
|---|---|---|---|
| RFFMG (T5Chem) | `断片集合 >> 分子` | T5 (~14.8M) | `t5chem` |
| RFFMG (GPT2) | `断片集合 >> 分子` | `entropy/gpt2_zinc_87m` (~87M) | `t5chem` |
| SAFE | SAFE 文字列 | safe-gpt (~88.8M) | `safe` |
| PromptSMILES | プレーンな SMILES + 推論時プロンプト | `entropy/gpt2_zinc_87m` | `promptsmiles` |
| FragGPT | FU-SMILES（BRICS 断片に対の `[i*]` ラベル） | `entropy/gpt2_zinc_87m` | `fraggpt` |

`run_*.sh` と `gen_*.sh` はすべてスクリプト内で環境を activate するため、どのシェルからでも実行できます。
以下の `python src/...` のコマンドは、併記した環境で実行してください。

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
$ mkdir -p models/rffmg/t5chem/pretrained
$ wget -P models/rffmg/t5chem/pretrained https://zenodo.org/records/14280768/files/simple_pretrain.tar.bz2
$ tar -xjvf models/rffmg/t5chem/pretrained/simple_pretrain.tar.bz2 --strip-components=3 -C models/rffmg/t5chem/pretrained/
```

### SAFE
```bash
$ mkdir -p models/safe/gpt/pretrained
$ git clone https://huggingface.co/datamol-io/safe-gpt/ models/safe/gpt/pretrained/
```

### curated datasetの準備
準備中

## データセットの構築
### 最初のステップ
```bash
$ conda activate t5chem
$ python src/curate_datasets.py
```

### データセットの作成
```bash
# 1. rffmgフラグメントの作成
$ conda activate t5chem
$ python src/gen_frags/rffmg_frags.py --frag_method brics # chose brics or rc_cms

# 2. safeフラグメントの作成
$ conda activate safe
$ python src/gen_frags/safe_frags.py --frag_method brics # chose brics or rc_cms

# 3. train, test, validationデータセットの作成
$ conda activate safe
$ python src/make_datasets.py --frag_method brics # chose brics or rc_cms
```

`make_datasets.py` は RFFMG データセットをどの `--sampling_num` でも出力しますが、
SAFE・PromptSMILES・FragGPT は **`--sampling_num 5`（既定値）のときだけ**出力します。
この3つは `sampling_num` の階層を持たないため、別の値で実行すると異なる分子分割で上書きされてしまうからです。
フラグメント化手法ごとに既定値で1回実行すれば、5種類すべてのデータセットが揃います。

| データセット | パス | 内容 |
|---|---|---|
| RFFMG | `data/rffmg/{frag}/{N}times_sampling/normal/` | `train/val/test.source` + `.target` |
| SAFE | `data/safe/{frag}/normal` | HF `DatasetDict`（`smiles`, `full_safe`, `pass_safe`, `full_fragments`, `pass_fragments`） |
| PromptSMILES | `data/promptsmiles/{frag}/normal` | HF `DatasetDict`（`smiles`, `pass_fragments`） |
| FragGPT | `data/fraggpt/{frag}/normal` | HF `DatasetDict`。train/validation は `full_fragments`、test は `pass_fragments` |

5つとも同一の分子分割を共有し、test split は同じ20,000分子が同じ行順で並ぶため、
手法間を行単位で直接比較できます。

### モデルの学習
```bash
# 1. rffmgモデルの学習（T5ChemまたはGPT2）
$ bash src/train_model/run_rffmg.sh
# .shファイル内の MODEL_NAME/MODE/FRAG_NAME を設定してください。
#   MODEL_NAME="t5chem": T5Chem（`t5chem train` を実行）。
#   MODEL_NAME="gpt":    GPT2（src/train_model/train_gpt.py）。MODE="finetuning" は entropy/gpt2_zinc_87m を初期値に、
#                        MODE="from_scratch" は同一configをランダム初期化で学習します。

# 2. safe-gptのファインチューニング
$ bash src/train_model/run_safe.sh
# 引数が非常に多いため.shファイルに記載済み
# .shファイル中のrc_cmsの部分、output_dirは適切に変更してください。また、--pretrain '' とすると事前学習済みモデルなしの学習が行われます。
# 本研究のfrom_scratchモデルは--pretrain ''とした場合の結果です。

# 3. PromptSMILES の prior（プレーンSMILESの言語モデル）の学習
$ bash src/train_model/run_promptsmiles.sh
# FRAG_NAME/MODE は .sh 上部で設定してください。
# prior は無条件のSMILES言語モデルで、PromptSMILES は推論時にのみプロンプトを与えます。
# 推論時のプロンプトは非カノニカルで任意の原子から始まるため、各分子は常にランダムな根原子から
# 書き直します。データ水増しは行わず（1分子=1系列）、ランダム化はデータセット構築時に1回だけ行います。

# 4. FragGPT（FU-SMILESの言語モデル）の学習
$ bash src/train_model/run_fraggpt.sh
# FRAG_NAME/MODE は .sh 上部で設定してください。
# 無条件のFU-SMILES言語モデルです。結合点の付番をランダムに置換し断片順をシャッフルします。
# 系列数は変わりません。
```

`MODE="finetuning"` は `entropy/gpt2_zinc_87m` を初期値に、`MODE="from_scratch"` は同一configを
ランダム初期化で学習します。GPT2ベースの4手法はハイパーパラメータを統一しており
（LR 1e-4 / 50エポック / batch 32 / warmup 10000 / eval・save 5000ステップごと /
EarlyStopping patience 15 / seed 42）、学習量ではなく表現の違いを比較できるようにしています。

### 分子の生成

4手法とも**同一の断片集合**（共有 test split の `pass_fragments`。結合点は素の `*` で、
どの断片同士がつながるかの情報は含まない）をプロンプトとし、
`target`, `prediction_1` .. `prediction_N` の列を持つ `predictions.csv` を出力します。
共通の評価パイプラインがそのまま読めます。

```bash
# rffmgモデル（T5ChemまたはGPT2。.sh内のMODEL_NAMEで切替）
$ bash src/gen_mols/gen_rffmg.sh

# safe-gptモデル
$ bash src/gen_mols/gen_safe.sh

# PromptSMILES（行ごとに scaffold decoration / fragment linking を自動で振り分け）
$ bash src/gen_mols/gen_promptsmiles.sh
# FRAG_NAME/MODEL_VER/GEN_METHOD は .sh 上部で設定してください。
# GEN_METHOD="beam" は RFFMG・SAFE と同条件、"sampling" は論文の多項サンプリングです。
# どちらを通ったかは predictions.csv の `sampler` 列に記録されます。
# PromptSMILES が表現できない断片集合では生成を行わず、その行は `unsupported` として
# INVALID_SMILES のまま残ります。

# FragGPT
$ bash src/gen_mols/gen_fraggpt.sh
# FRAG_NAME/MODEL_VER は .sh 上部で設定してください。
# 与えた断片集合の各結合点に新しい番号を振り、モデルが FU-SMILES の続きを生成したあと、
# [i*] の番号を照合して断片を組み立てます。
```

生成結果は `results/{表現}/{モデル}/{model_ver}/{frag}/{gen_method}/{additional_path}/` に出力されます。

### 生成分子の評価
```bash
$ conda activate safe
$ python src/evaluation.py --repr_name rffmg --model_name gpt --model_ver finetuning --frag_method rc_cms --additional_path normal
# --repr_name:  rffmg / safe / promptsmiles / fraggpt（表現。パスの第1階層）
# --model_name: t5chem / gpt（表現を学習させたモデル。パスの第2階層）
# --gen_method: beam / sampling（既定は beam。promptsmiles のみ既定が sampling）
```

有効な `--repr_name` / `--model_name` の組み合わせ（これ以外はパーサが弾きます）:

| `--repr_name` | `--model_name` | 結果のパス |
|---------------|----------------|-----------|
| rffmg         | t5chem         | `results/rffmg/t5chem/` |
| rffmg         | gpt            | `results/rffmg/gpt/` |
| safe          | gpt            | `results/safe/gpt/` |
| promptsmiles  | gpt            | `results/promptsmiles/gpt/` |
| fraggpt       | gpt            | `results/fraggpt/gpt/` |

先に生成を実行してください。評価は `test.source` と `predictions.csv` を行番号で結合しており、
PromptSMILES と FragGPT は `test.source` / `test.target` を生成時に書き出すためです。

`src/evaluation.py` が出力する `stats.csv` は test の全行を対象とし、4手法へ同じ断片集合を要求します。
PromptSMILES は表現できない断片集合に対しては生成を行わず、その行は `INVALID_SMILES` として0点で
集計されます（FragGPT の組み立て失敗・SAFE のデコード失敗と同じ扱いです）。したがってこの数値は
カバレッジを含んだものになります。

どの行が生成に回らなかったかは `predictions.csv` の `sampler` 列に記録されます
（`scaffold` / `linking` / `unsupported` / `invalid_target` / `generation_error`）。
理由別の件数は `generation_params.txt` にあります。

If you need the curated ChEMBL dataset used in this study, please feel free to contact us at [sato.akinori@naist.ac.jp] or [miyao@dsc.naist.jp].
