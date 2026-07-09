# Plan: 学習スクリプトの出力構造を新レイアウトに移行し、wandb を識別可能に保存

- **Date**: 2026-07-08
- **Status**: pending-approval

## Overview

学習の保存先を新レイアウト `models/{gpt|t5chem}/{safe|rffmg}/{mode}/{slice}/` に移行し、
wandb の run を「どのモデル/表現/モード/slice か」で識別・フィルタできるようにする。

**モデル/出力レイアウト（新）**

| model | repr | mode | 出力先 |
|-------|------|------|--------|
| gpt (safe) | safe | finetuning | `models/gpt/safe/finetuning/${FRAG_NAME}/` |
| gpt (safe) | safe | from_scratch | `models/gpt/safe/from_scratch/${FRAG_NAME}/` |
| t5chem | rffmg | finetuning | `models/t5chem/rffmg/finetuning/${FRAG_NAME}/` |
| t5chem | rffmg | from_scratch | `models/t5chem/rffmg/from_scratch/${FRAG_NAME}/` |

- MODE の値を dir 名と一致させ `finetuning` / `from_scratch` に統一（従来の `finetune` から改名）。
- pretrained（学習済みモデル）は本タスクでは移動しない。**ソース参照は既存の
  `models/safe_gpt/pretrained` / `models/t5chem/pretrained` のまま**（実ファイル移動・生成側パス更新は別途）。

**wandb マッピング**

| 項目 | safe (gpt) | t5chem |
|------|-----------|--------|
| project | `gpt_safe`（`--wandb_project`） | `t5chem_rffmg`（env + パッケージ setdefault 化） |
| 識別名 | `WANDB_RUN_GROUP=${MODE}_${FRAG_NAME}`（run名は safe-train が uuid 固定のため group で識別） | `--run_name "${MODE}_${FRAG_NAME}"` |
| tags | `WANDB_TAGS=gpt,safe,${MODE},${FRAG_NAME}` | `WANDB_TAGS=t5chem,rffmg,${MODE},${FRAG_NAME}` |

学習率など他のハイパラは現状維持（統一方針）。

## Plan

### Step 1: run_safe.sh を新構造 + wandb 対応に更新

- **Target file**: `src/train_model/run_safe.sh`
- **Changes**:
  - `MODE` の値を `finetuning` / `from_scratch` に変更（既定 `finetuning`）。
  - 分岐の `OUTPUT_DIR` を新構造に:
    - finetuning: `models/gpt/safe/finetuning/${FRAG_NAME}/`（`--model_path ${PRETRAINED_DIR}` あり）
    - from_scratch: `models/gpt/safe/from_scratch/${FRAG_NAME}/`（`--model_path` なし）
  - `PRETRAINED_DIR="models/safe_gpt/pretrained"` は据え置き（config/tokenizer/model_path の参照元）。
  - wandb 用に分岐後へ追記:
    - `export WANDB_RUN_GROUP="${MODE}_${FRAG_NAME}"`
    - `export WANDB_TAGS="gpt,safe,${MODE},${FRAG_NAME}"`
  - `safe-train` 引数に `--wandb_project gpt_safe` を追加（project 指定 + report_to=wandb を保証）。
  - 他の引数は現状維持。
- **Dependencies**: none

### Step 2: run_t5chem.sh を新構造 + wandb 対応に更新

- **Target file**: `src/train_model/run_t5chem.sh`
- **Changes**:
  - `MODE` の値を `finetuning` / `from_scratch` に変更（既定 `finetuning`）。
  - 分岐の `OUTPUT_DIR` を新構造に:
    - finetuning: `models/t5chem/rffmg/finetuning/${FRAG_NAME}`（`--pretrain models/t5chem/pretrained`）
    - from_scratch: `models/t5chem/rffmg/from_scratch/${FRAG_NAME}`（`--tokenizer simple`）
  - wandb 用に分岐後へ追記:
    - `export WANDB_PROJECT="t5chem_rffmg"`
    - `export WANDB_TAGS="t5chem,rffmg,${MODE},${FRAG_NAME}"`
  - `t5chem train` 引数に `--run_name "${MODE}_${FRAG_NAME}"` を追加。
  - 他の引数（`--data_dir`, `--task_type product`, `--num_epoch 50`）は現状維持。
- **Dependencies**: none

### Step 3: t5chem の run_trainer.py（インストール済みパッケージ）を wandb 有効化 + project を env 尊重に

- **Target file**: `/home/sato/miniconda3/envs/t5chem/lib/python3.12/site-packages/t5chem/run_trainer.py`
- **Changes**:
  - 114行目 `os.environ["WANDB_PROJECT"]="T5Chem"` → `os.environ.setdefault("WANDB_PROJECT", "T5Chem")`
    （run_t5chem.sh の `export WANDB_PROJECT=t5chem_rffmg` が優先されるようにする）
  - 265行目 `report_to="none",` → `report_to="wandb",`（wandb 記録を有効化）
- **Dependencies**: Step 2（run_t5chem.sh が env/引数を渡す前提）
- **注意**: これは site-packages（リポジトリ外）の編集で、**t5chem 再インストール時に失われる**。
  再現性のため README_ja に手順を追記/整合させることを推奨（本計画では任意）。

## Notes / 確認事項

- 既定は両スクリプトとも `MODE="finetuning"`, `FRAG_NAME="rc_cms"`。
- **pretrained ソースは旧パス（`models/safe_gpt/pretrained` 等）のまま**。出力のみ新構造。
  出力が `models/gpt/safe/...` に出るため、生成側（`gen_safe.py` / `gen_safe_denovo.py` / `gen_t5chem.py`）が
  新パスを読むには別途更新が必要（本計画外）。
- 既存モデルディレクトリ（`models/safe_gpt/*`, `models/t5chem/{trained,from_scratch,finetuning}/*`）の
  物理移動は本計画外（別途）。
- safe-train は run 名を `safe-model-<uuid>` で固定するため、safe 側の識別は project + group + tags + config に依存する。
- wandb がオフライン運用の場合も group/tags/run名は `.wandb` に記録され、`wandb sync` や `src/figure.py` で参照可能。
