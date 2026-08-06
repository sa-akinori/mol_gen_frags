# Plan: FragGPT / PromptSMILES に from_scratch モードを追加する

- **Date**: 2026-08-04
- **Status**: pending-approval

## Overview

`.plans_for_ai/2026-07-29/add_fraggpt_baseline.md` および
`.plans_for_ai/2026-07-28/add_promptsmiles_baseline.md` で
「**finetuning と from_scratch の両方を学習する**（既存2手法と同じ構成に揃えるため）」と
決めていたが、現在の `train_fraggpt.py` / `train_promptsmiles.py` には
**`--mode` 引数が存在せず、`from_scratch` が実装されていない**。

### 現状（実測）

| | RFFMG (`train_gpt.py`) | FragGPT | PromptSMILES |
|---|---|---|---|
| `--mode` 引数 | あり | **なし** | **なし** |
| `from_scratch` | `GPT2Config` からランダム初期化 | **未実装**（常に `from_pretrained`） | **未実装** |
| 出力先 | `models/rffmg/gpt/{mode}/...` | **`models/fraggpt/gpt/finetuning/...` 固定** | **`models/promptsmiles/gpt/finetuning/...` 固定** |

SAFE と RFFMG は両モードを持つため、このままではベースライン比較表に穴が空く。

### `run_fraggpt.sh` の破損

```bash
export WANDB_DIR="wandb/fraggpt/gpt/${MODE}/${FRAG_NAME}"
```

`MODE` 変数が定義されておらず、`wandb/fraggpt/gpt//brics` と空文字が挟まる。
`FRAG_NAME` の行はあるが `MODE=` の行が失われている。

### 両 .sh から失われている共通項目（要確認）

`run_fraggpt.sh` / `run_promptsmiles.sh` とも、以前は持っていた以下が現在は無い。

- `export CUDA_VISIBLE_DEVICES=0`
- リポジトリルートへの `cd`
- `source ~/miniconda3/etc/profile.d/conda.sh` + `conda activate env_fraggpt` / `env_promptsmiles`

`gen_fraggpt.sh` は現在これらを持っており（2026-08-02 にユーザー判断で追加）、
`run_*.sh` だけ無い状態になっている。**復元するかはユーザーに確認する。**

## Plan

### Step 1: `train_fraggpt.py` に `--mode` を追加する

- **Target file**: `src/train_model/train_fraggpt.py`
- **Changes**: `src/train_model/train_gpt.py:107-108, 142-148, 153` を手本にする。
  - `--mode` を追加: `type=str, default="finetuning", choices=["finetuning", "from_scratch"]`
  - モデル初期化を分岐させる。
    ```python
    if args.mode == "finetuning":
        model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    else:  # from_scratch
        config = GPT2Config.from_pretrained(pretrained_model)
        model = GPT2LMHeadModel(config)
    ```
    **tokenizer は両モードとも `from_pretrained` のまま**（語彙は共通、重みだけ変える）。
  - `from transformers import ...` に `GPT2Config` を追加する。
  - `output_dir`（113行）を `models/fraggpt/gpt/{args.mode}/{args.frag_method}` にする。
  - 学習ハイパラは一切変更しない。
- **Dependencies**: none

### Step 2: `train_promptsmiles.py` に `--mode` を追加する

- **Target file**: `src/train_model/train_promptsmiles.py`
- **Changes**: Step 1 と完全に同一の変更を行う。
  `output_dir`（120行）を `models/promptsmiles/gpt/{args.mode}/{args.frag_method}` にする。
  学習ハイパラおよび SMILES ランダム化の挙動は一切変更しない。
- **Dependencies**: none

### Step 3: `run_fraggpt.sh` を修正する

- **Target file**: `src/train_model/run_fraggpt.sh`
- **Changes**:
  - 上部に `MODE="finetuning"` を追加する（コメントで `"finetuning"` / `"from_scratch"` を明示）。
    `FRAG_NAME` と同じ書式に揃えること。
  - `train_fraggpt.py` の実行に `--mode "${MODE}"` を渡す。
  - `WANDB_DIR` の `${MODE}` が解決されることを確認する（変数定義により自動的に解決される）。
  - **以下は「要確認」項目。ユーザーの承認があった場合のみ追加すること**:
    - `export CUDA_VISIBLE_DEVICES=0`
    - リポジトリルートへの `cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1`
    - `source ~/miniconda3/etc/profile.d/conda.sh` + `conda activate env_fraggpt`
- **Dependencies**: after Step 1

### Step 4: `run_promptsmiles.sh` を修正する

- **Target file**: `src/train_model/run_promptsmiles.sh`
- **Changes**: Step 3 と同一。
  - `MODE="finetuning"` を追加し `--mode "${MODE}"` を渡す。
  - `WANDB_DIR` の `finetuning` 固定を `${MODE}` に変更する。
  - 「要確認」項目（`conda activate env_promptsmiles` など）の扱いは Step 3 と揃える。
- **Dependencies**: after Step 2

### Step 5: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 全変更ファイルの構文チェック（py / bash）
  - `--mode from_scratch` で**事前学習重みが読まれていない**こと（重みが `from_pretrained` と異なること）を確認
  - `--mode finetuning` で事前学習重みと一致すること
  - 出力先が `models/{手法}/gpt/{mode}/{frag_method}` になること（両モードで別ディレクトリ）
  - 小さいデータで両モードが数ステップ回ること
  - `train_gpt.py`（RFFMG）が影響を受けていないこと
- **Dependencies**: after Step 4

## スコープ外

- `train_gpt.py`（RFFMG）と SAFE の学習スクリプト
- 学習ハイパラの変更
- `README.md` / `README_ja.md` の更新（必要なら別途）

## 注意

`models/fraggpt/` と `models/promptsmiles/` は未作成のため、出力先の変更による
既存モデルへの影響は無い。
