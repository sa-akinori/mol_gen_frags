# Plan: RFFMG-GPT バックエンド追加

- **Date**: 2026-07-09
- **Status**: approved

## Overview

RFFMG のフラグメント表現（`フラグメント >> 分子` の source→target）を、既存の T5Chem に加えて
**GPT2 系デコーダ専用モデル**でも学習・生成・評価できるようにする。事前学習済み
`entropy/gpt2_zinc_87m`（GPT2, 12層/768/約87M, vocab 2707, n_positions 256）を利用し、
SAFE-GPT(88.8M) と等容量での比較を可能にする。T5Chem と並ぶ `MODEL_NAME=gpt` バックエンドとして追加する。

### 確定した決定事項
- **学習モード**: finetuning（ZINC 重み初期化）+ from_scratch（ZINC と同一 config・同一トークナイザ、重み random）の両対応
- **区切り**: `>>`（元の sentence 形式そのまま）
- **スコープ**: 学習・生成・評価までのフルパイプライン
- **生成スクリプトの統合**: `gen_t5chem.{py,sh}`（gen_mols 版）を `gen_rffmg.{py,sh}` にリネームし、`--model_name {t5chem,gpt}` で分岐する統合スクリプトにする。GPT2 生成本体は `src/func/generation_rffmg_func.py`
- **HP**: run_safe.sh 準拠、LR=1e-4 統一（両モード）、50 epoch、warmup 10000、batch 32、eval/save 5000 step、save_total_limit 5、load_best_model_at_end、wandb offline、seed 固定
- **実行環境**: 既存 t5chem env を流用（transformers 4.45.2 / torch 2.10）。**t5chem パッケージは変更しない**（素の transformers を使用）

### データ表現（デコーダ専用化）
- 学習系列: `<bos>` + `source + ">>" + target` + `<eos>`（= 元 sentence に bos/eos）
- labels: プロンプト部（`<bos>source>>`）を -100 マスクし、target 部のみ loss（条件付き生成 p(target|source)）
- 生成プロンプト: `<bos>` + `source + ">>"` → 続きを生成し `>>` 以降・eos までを予測分子とする
- 最大長: 256（n_positions）。train と gen で formatting を厳密一致させる

## Plan

### Step 1: GPT2 学習スクリプト新規作成

- **Target file**: `src/train_model/train_gpt.py`（新規）
- **Changes**:
  - argparse: `--frag_method {brics,rc_cms}`, `--mode {finetuning,from_scratch}`, `--pretrain`(既定 `entropy/gpt2_zinc_87m`), `--data_dir`, `--output_dir`, epoch/lr/batch/warmup 等
  - トークナイザ: `AutoTokenizer.from_pretrained("entropy/gpt2_zinc_87m")`（両モード共通）
  - モデル: finetuning=`GPT2LMHeadModel.from_pretrained(pretrain)` / from_scratch=`GPT2LMHeadModel(GPT2Config(ZINC 同一: n_layer=12, n_embd=768, n_head=12, n_positions=256, vocab_size=2707, bos/eos id 一致))`
  - データ: `data/rffmg/{frag}/normal/{train,val}.{source,target}` を読み、`source + ">>" + target` を組立 → tokenize（bos/eos 手動付与, max_length=256, truncation）→ labels のプロンプト部を -100 マスク
  - `Trainer` + `TrainingArguments`: 上記 HP、`report_to=["wandb"]`、`load_best_model_at_end=True`
  - `set_seed` で乱数固定（再現性）
  - 学習後 `trainer.save_model(f"{output_dir}/best_model")` とトークナイザ保存
  - コードスタイル: 型ヒント必須・Google style docstring・import 順（標準→サードパーティ→ローカル）
- **Dependencies**: none（既存 `.source/.target` を利用、make_datasets 変更不要）

### Step 2: 学習ラッパー .sh の MODEL_NAME 分岐

- **Target file**: `src/train_model/run_rffmg.sh`（改修）
- **Changes**: `MODEL_NAME` を `t5chem|gpt` で分岐。`gpt` の場合
  `python src/train_model/train_gpt.py --frag_method $FRAG_NAME --mode $MODE --output_dir models/rffmg/gpt/${MODE}/${FRAG_NAME}` を実行。
  wandb dir を `wandb/rffmg/gpt/${MODE}/${FRAG_NAME}` に。既存 t5chem 分岐は温存。
- **Dependencies**: after Step 1

### Step 3: GPT2 生成本体を func/ に新規作成

- **Target file**: `src/func/generation_rffmg_func.py`（新規、`generation_safe_func.py` と同じ subprocess パターン）
- **Changes**:
  - argparse: `--model_path`, `--dataset_file`(test.source), `--target_file`(test.target), `--output_dir`, `--n_samples`, `--max_length`(既定 256), `--num_beams`, `--batch_size`, `--random_seed`
  - `GPT2LMHeadModel.from_pretrained` + `AutoTokenizer.from_pretrained` をロード、GPU、eval
  - 各 source を `<bos>source>>` にして beam search（num_beams, num_return_sequences=n_samples, max_length=256）。生成列から `>>` 以降・eos までを取り出し予測分子 SMILES とする。バッチ処理・seed 固定
  - **predictions.csv を T5Chem と同一列**（`target`(=test.target), `prediction_1..N`）で書き出し
- **Dependencies**: after Step 1

### Step 4: 生成ラッパーを gen_rffmg に統合（リネーム + model_name 分岐）

- **Target file**: `src/gen_mols/gen_t5chem.py` → `src/gen_mols/gen_rffmg.py`（リネーム+改修）、`src/gen_mols/gen_t5chem.sh` → `src/gen_mols/gen_rffmg.sh`（リネーム+改修）
- **Changes**:
  - `--model_name {t5chem,gpt}`（既定 t5chem）を追加。model_path=`models/rffmg/{model_name}/{model_ver}/{frag}/best_model`、出力=`results/rffmg/{model_name}/{model_ver}/{frag}/beam/{additional_path}/predictions.csv`
  - 分岐: `t5chem` → 既存 `t5chem predict`（subprocess）／`gpt` → `python src/func/generation_rffmg_func.py ...`（subprocess）
  - 既存引数（`--frag_method`, `--model_ver`, `--additional_path`, `--n_samples`, `--max_length`(gpt 用に既定 256 を扱えるように), `--num_beams`, `--batch_size`, `--random_seed`）は維持
  - gen_rffmg.sh に `MODEL_NAME` 変数を追加し `--model_name` を渡す
- **Dependencies**: after Step 3

### Step 5: 評価スクリプトの GPT 対応

- **Target file**: `src/evaluation.py`（改修）
- **Changes**: `--model_name` に `gpt`(RFFMG-GPT) を追加（choices と help）。`str_name`/`model_dir` の 2 値分岐を 3 値化 →
  gpt は `str_name='rffmg'`, `model_dir='gpt'`, `tr_file_name=data/rffmg/{frag}/normal/train.target`,
  `testInputfile=data/rffmg/{frag}/{additional_path}/test.source`。
  `loadTrainSmiles`/`loadGenSmiles` へは `arc_name='t5chem'` 相当を渡し既存分岐を再利用（gen_rffmg(gpt) が同形式 CSV のため `evaluation_func.py` は無改修）
- **Dependencies**: after Step 4（CSV 形式確定後）

### Step 6: 作図スクリプトの GPT 対応

- **Target file**: `src/figure.py`（改修）
- **Changes**: `arc_name`（現状 `'t5chem'` ハードコード）に rffmg+gpt の組合せを扱えるよう
  `str_name='rffmg'`/`model_dir='gpt'` のパス生成を対応。`results/rffmg/gpt/...`・`figures/.../rffmg/gpt/...` を参照可能に
- **Dependencies**: after Step 5

### Step 7: 動作確認（debug データでスモーク）+ ドキュメント更新

- **Target file**: `README.md` / `README_ja.md` / `.claude/structure.md`（改修）
- **Changes**:
  - `data/rffmg/{frag}/normal/debug` の少量データで train_gpt.py → gen_rffmg.py(gpt) → evaluation.py を 1 回通し、パス・CSV 形式・系列長(256 内)を確認
  - README/structure に GPT バックエンド（`MODEL_NAME=gpt`、train_gpt.py/generation_rffmg_func.py/gen_rffmg.py、`models/rffmg/gpt/...` パス、gen_rffmg へのリネーム）を追記
- **Dependencies**: after Step 6

## 変更しないもの（再利用）

- `src/make_datasets.py`（既存 `.source/.target` をそのまま利用）
- `src/func/evaluation_func.py`（gen_rffmg(gpt) が t5chem 形式で出力 → t5chem 分岐を再利用）
- `src/gen_frags/rffmg_frags.py`（表現は共通）
- `site-packages/t5chem`（**パッケージ改変なし**、素の transformers を使用）
- `requirements/*`（transformers 既存、新規追加なし）

## リスク / 留意

- n_positions=256: normal データは concat max 217 で収まる。将来の長い評価スライス生成時は要再確認
- from_scratch はランダム初期化のため 50 epoch で収束するか要観察（HP 探索はしない方針）
- 生成の formatting（bos / `>>` / eos）を train と gen で厳密一致させること
- gen_t5chem → gen_rffmg リネームに伴い、これらを参照する箇所（README 等）があれば追随更新
