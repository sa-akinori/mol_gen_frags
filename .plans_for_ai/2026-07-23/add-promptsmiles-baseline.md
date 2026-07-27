# Plan: PromptSMILES をベースラインモデルとして追加

- **Date**: 2026-07-23
- **Status**: pending-approval

## Overview

PromptSMILES (Thomas et al., *J. Cheminform.* 2024) をベースラインモデルとして本プロジェクトに組み込む。
PromptSMILES 自体は「制約付き生成（scaffold decoration / fragment linking）を再学習なしで行う推論時プロンプト手法」だが、
公平な比較のためには**ベースとなる化学言語モデル（prior）を既存モデルと同一の ChEMBL31 コーパスで再学習する必要がある**。

- **ベースCLMアーキ**: `GTr`（StableTransformerEncoder。RFFMG-GPT(GPT2) と系統を揃え比較しやすくするため）
- **評価タスク**: Scaffold decoration + Fragment linking
- **実行環境**: 新規 conda 環境 `env_promptsmiles`（既存 env との依存衝突回避、CLAUDE.md の「新規ライブラリは事前確認」を本計画で充足）
- 実装は外部パッケージ（`smiles-rnn` / `promptsmiles`, いずれも MIT）を利用し、本リポジトリには既存構成（`src/train_model/`, `src/gen_mols/`, `src/evaluation.py`）に合わせた薄いラッパーのみ追加する。

### タスク入力のマッピング方針
既存の RFFMG/SAFE 比較と条件を揃えるため、評価用スキャフォールド／フラグメントは
テストセット分子から既存の `src/func/fragmentation.py` を用いて生成し、PromptSMILES の記法
（結合点を `*`、フラグメント区切りを `.`）に変換する。これにより同一テスト分子集合上で
再構成率・妥当性・一意性・新規性を横並び比較する。

## Plan

### Step 1: 新規 conda 環境のセットアップ手順を追加

- **Target file**: `src/train_model/setup_promptsmiles_env.sh`（新規）
- **Changes**: 環境構築コマンドをスクリプト化。
  `conda create -n env_promptsmiles python=3.11` →
  `pip install smiles-rnn promptsmiles molscore` →
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` →
  `pip install rdkit pandas numpy`。冒頭にコメントで用途・GPU前提を明記。
  併せて CLAUDE.md の環境節に追記するかは実装時に確認（src 直変更を避けるため本スクリプトを正とする）。
- **Dependencies**: none

### Step 2: 学習コーパスの書き出しスクリプト

- **Target file**: `src/make_promptsmiles_corpus.py`（新規）
- **Changes**: `data/curated/passed_filters_rdkit_canonical_smiles.tsv` の `rdkit_washed_smiles` 列を読み込み、
  `Chem.MolFromSmiles` の None チェック（無効SMILES除外）→重複除去→`random_state=42` で train/val/test 分割し、
  1行1SMILES のプレーンテキスト `data/promptsmiles/{train,val,test}.smi` を出力。
  件数・除外数を `data/promptsmiles/corpus_log.txt` に記録。パスは全て `pathlib.Path` / `BASEPATH` 経由。
  型ヒント・Google style docstring 付与。
- **Dependencies**: none

### Step 3: prior 学習ラッパースクリプト

- **Target file**: `src/train_model/run_promptsmiles.sh`（新規）
- **Changes**: `run_safe.sh` の作法に倣う（リポジトリルートへ cd、`conda activate env_promptsmiles`、
  `WANDB_MODE=offline`）。`train_prior.py GTr -i data/promptsmiles/train.smi
  --valid_smiles data/promptsmiles/val.smi -o models/promptsmiles/gtr -s chembl31
  --grammar SMILES --randomize --n_epochs <N> --batch_size <B> --device gpu` を実行。
  出力は `models/promptsmiles/gtr/`。エポック等ハイパラは変数で上部に定義しログ化。
- **Dependencies**: after Step 1, Step 2

### Step 4: 生成スクリプト（scaffold decoration / fragment linking）

- **Target file**: `src/gen_mols/gen_promptsmiles.py` + `src/gen_mols/gen_promptsmiles.sh`（新規）
- **Changes**: 学習済み prior をロードし、`promptsmiles` の `ScaffoldDecorator` / `FragmentLinker`
  に `sample_fn` / `evaluate_fn`（尤度）を渡して生成。`--task {scaffold,linking}`、
  `--frag_method {brics,rc_cms}`、`--n_samples`、`--random_seed 42` を引数化。
  評価用スキャフォールド／フラグメントは Overview の方針でテストセットから生成。
  生成結果は evaluation.py が読める形式・パス（`results/promptsmiles/...`）で保存。
- **Dependencies**: after Step 3

### Step 5: 評価パイプラインへの統合

- **Target file**: `src/evaluation.py`（+ 必要なら `src/func/evaluation_func.py`）
- **Changes**: `--model_name` に `promptsmiles` を追加し、`str_name/model_dir/arc_name` の分岐と
  出力フォーマット読み取りを追加。既存指標（validity / uniqueness / novelty / reconstruction）を
  RFFMG・SAFE と同一関数で算出できるよう配線。
- **Dependencies**: after Step 4

### Step 6: 再現性・ログの担保（横断）

- **Target file**: Step 2–5 の各スクリプト
- **Changes**: `set_seed(42)` の使用、乱数シード明示、ハイパラ・メトリクスのログ保存、
  wandb offline ディレクトリ分離、データパスの `Path` 化を各所で徹底。
- **Dependencies**: Step 2–5 に内包

## 補足・確認事項（実装前に要検討）

1. **学習エポック/バッチ**: 既存 SAFE は 50 epoch。PromptSMILES prior の妥当な epoch 数を決める必要がある（暫定 10〜50）。
2. **GPU/学習時間**: 約188万件コーパスでの Transformer prior 学習は数時間〜規模。実行タイミング要相談。
3. **評価スキャフォールド集合の定義**: 論文の 17 スキャフォールド等を使うか、本プロジェクトのテスト分子由来にするか。本計画は後者（比較整合性重視）を採用。
