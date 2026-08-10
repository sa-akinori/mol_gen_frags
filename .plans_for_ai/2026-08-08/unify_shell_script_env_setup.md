# Plan: 全 .sh で必要な conda 環境を activate する

- **Date**: 2026-08-08
- **Status**: approved

## Overview

学習・生成の `.sh` 8本のうち、環境設定を持たないものが4本ある。実行前に手動で activate する
運用にしていたが、方針を変えて **すべての `.sh` が自分で必要な環境を activate する**形に統一する。

### 現状（実測）

| ファイル | shebang | cd | conda activate | CUDA_VISIBLE_DEVICES |
|---|---|---|---|---|
| `run_fraggpt.sh` | **なし** | あり | **なし** | **なし** |
| `run_promptsmiles.sh` | あり | あり | promptsmiles | あり |
| `run_rffmg.sh` | あり | あり | t5chem | あり |
| `run_safe.sh` | あり | あり | safe | あり |
| `gen_fraggpt.sh` | あり | あり | fraggpt | あり |
| `gen_promptsmiles.sh` | あり | **なし** | **なし** | **なし** |
| `gen_rffmg.sh` | あり | **なし** | **なし** | **なし** |
| `gen_safe.sh` | あり | **なし** | **なし** | あり |

### `gen_*.sh` に `cd` を入れない理由

`gen_promptsmiles.sh` / `gen_rffmg.sh` / `gen_safe.sh` は
`SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` を定義して `${SCRIPT_DIR}/gen_*.py` を呼ぶ方式で、
**cwd に依存しない**。加えて4環境すべてに `__editable__.func-0.1.0.pth` が入っており
`func` パッケージが editable install されているため、`from func.utility import ...` も
cwd に関係なく解決する。実測でも `/tmp` から `python .../gen_rffmg.py --help` が通る。

したがって **`SCRIPT_DIR` 方式を維持し、`cd` は追加しない**（ユーザー判断）。

### スコープ外

- `conda activate` の失敗検査（`|| exit 1`）— ユーザー判断で追加しない
- `gen_fraggpt.sh` の `cd` 方式を `SCRIPT_DIR` 方式に変える作業（現状で動いているため触らない）
- `requirements/promptsmiles_requirements.txt` への `joblib` 追加 — ユーザーが別途対応

## Plan

### Step 1: `run_fraggpt.sh` を他の `run_*.sh` と同形にする

- **Target file**: `src/train_model/run_fraggpt.sh`
- **Changes**:
  - 先頭に以下を追加する。`run_promptsmiles.sh` / `run_rffmg.sh` / `run_safe.sh` と
    **同じ順序・同じコメント文言**に揃えること。
    ```bash
    #!/usr/bin/env bash
    export CUDA_VISIBLE_DEVICES=0

    # リポジトリルートに移動（どこから実行しても相対パスが解決できるようにする）
    cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

    # conda setup
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate fraggpt
    ```
  - `cd` の行は既にあるので重複させないこと。位置だけ他3本に合わせる。
  - `FRAG_NAME` / `MODE` / `WANDB_*` / python 実行部は変更しない。
- **Dependencies**: none

### Step 2: `gen_*.sh` 3本に環境設定を追加する

- **Target file**: `src/gen_mols/gen_promptsmiles.sh`, `src/gen_mols/gen_rffmg.sh`,
  `src/gen_mols/gen_safe.sh`
- **Changes**:
  - 各ファイルの `SCRIPT_DIR` 定義の**前**に以下を追加する。
    ```bash
    export CUDA_VISIBLE_DEVICES=0

    # conda setup
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate <env>
    ```
  - `<env>` は `gen_promptsmiles.sh` → `promptsmiles`、`gen_rffmg.sh` → `t5chem`、
    `gen_safe.sh` → `safe`。**実在する環境名**（`conda env list` で確認済み）。
  - `gen_safe.sh` は既に `export CUDA_VISIBLE_DEVICES=0` を持つので重複させないこと。
  - **`SCRIPT_DIR` はそのまま残し、`cd` は追加しない**（概要参照）。
  - `FRAG_NAME` などの設定変数と python 実行部は変更しない。
- **Dependencies**: none

### Step 3: README の記述を戻す

- **Target file**: `README.md`, `README_ja.md`
- **Changes**:
  - 直前に追加した `$ conda activate fraggpt   # the .sh does not activate it` /
    `$ conda activate fraggpt   # .sh は activate しません` の行を削除する。
  - その下の `# Set FRAG_NAME/MODE at the top of the .sh.` /
    `# FRAG_NAME/MODE は .sh 上部で設定してください。` を、他手法と同じ
    `# The .sh activates fraggpt itself. Set FRAG_NAME/MODE at the top of the .sh.` /
    `# .sh が fraggpt を activate します。FRAG_NAME/MODE は .sh 上部で設定してください。`
    に戻す。
  - **存在しない `--no-augment` の記述を復活させないこと**（同日に削除済み）。
- **Dependencies**: after Step 1

### Step 4: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 8本すべてに shebang / `conda activate` / `CUDA_VISIBLE_DEVICES` が揃うこと
    （`cd` は `run_*.sh` 4本 + `gen_fraggpt.sh` のみ）
  - 各 `.sh` が activate する環境が実在すること（`conda env list` と照合）
  - 全8本の `bash -n`
  - **リポジトリルート以外から実行してもパスが解決すること。**
    学習・生成を起動せずに確認すること（`gen_*.sh` は `--help` を付けた形で試す、
    `run_*.sh` は python 実行行を読み替えて確認するなど）
  - `grep -rn "env_promptsmiles\|env_fraggpt"` が0件のままであること
- **Dependencies**: after Step 3
