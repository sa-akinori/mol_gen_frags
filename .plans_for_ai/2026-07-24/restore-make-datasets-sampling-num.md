# Plan: make_datasets.py の sampling_num 対応（{N}times_sampling 階層）

- **Date**: 2026-07-24
- **Status**: pending-approval

## Overview

`rffmg_frags.py` の復元で出力先が `data/rffmg/{frag}/{N}times_sampling/full_dataset.csv` になったが、
`src/make_datasets.py` は旧レイアウト（`data/rffmg/{frag}/...`、times_sampling 階層なし）のまま。
このままでは `full_dataset.csv` を読めず、各評価用データセットも旧パスに出力してしまう。
実データ・下流（`run_rffmg.sh` は `data/rffmg/${FRAG}/${N}times_sampling/normal` を参照）と揃える必要がある。

### 確認済みの前提

- `data/rffmg/{frag}/{N}times_sampling/` 配下に `full_dataset.csv`, `normal/`, `unique_frags.csv`, `frag_order/`, `frag_num/`, `dup_frags/`, `attach_point_num/` が存在（実データで確認）。
- `data/safe/{frag}/` は **times_sampling 階層を持たない**（`data/safe/{frag}/normal`）。→ **safe 系パスは変更しない**。
- 引数名・デフォルトは `rffmg_frags.py` と統一: `--sampling_num`（int, default=5）。

## Plan

### Step 1: `--sampling_num` 引数を追加

- **Target file**: `src/make_datasets.py`
- **Changes**:
  - `argparse` に `parser.add_argument('--sampling_num', type=int, default=5, help='number of sampling trials per fragment set (data/rffmg/<frag>/<N>times_sampling)')` を追加（`--frag_method` の直後）。
- **Dependencies**: none

### Step 2: rffmg 出力ベースディレクトリ変数を導入

- **Target file**: `src/make_datasets.py`
- **Changes**:
  - Setting 節（`frag_method = args.frag_method` の直後）に
    `rffmg_dir = f'{fd}/rffmg/{frag_method}/{args.sampling_num}times_sampling'` を追加。
  - 変数導入により以降のパス重複を排除（スタイル規約「重複処理の排除」に沿う）。
- **Dependencies**: after Step 1

### Step 3: rffmg 系パスを `rffmg_dir` ベースへ置換

- **Target file**: `src/make_datasets.py`
- **Changes**: 以下の `f'{fd}/rffmg/{frag_method}/...'` を `f'{rffmg_dir}/...'` に置換（safe 系は対象外）。
  - L57: `full_dataset.csv` 読み込み
  - L76–78: `normal/` の train/val/test source/target
  - L86–88: `normal/debug/` の debug 出力
  - L122: `full_dataset.csv` 再読み込み
  - L127–128: `unique_frags.csv`（`os.makedirs(f'{fd}/rffmg/{frag_method}')` → `os.makedirs(rffmg_dir)`）
  - L131–132: `normal/test.source|target` 読み込み
  - L142–146: `frag_order/`
  - L150: `unique_frags.csv` 読み込み
  - L154: `normal/train.source` 読み込み
  - L174–176: `frag_num/`
  - L180, L185: `unique_frags.csv`, `normal/train.source` 読み込み
  - L235–238: `dup_frags/`
  - L242, L245: `unique_frags.csv`, `normal/train.source` 読み込み
  - L276–278: `attach_point_num/`
- **Dependencies**: after Step 2

### Step 4: 動作確認（read-only）

- **Target file**: なし（確認のみ）
- **Changes**:
  - `python -m py_compile src/make_datasets.py` で構文確認。
  - `grep` で `{fd}/rffmg/{frag_method}` の残存が無い（safe 側のみ `{fd}/safe/{frag_method}` が残る）ことを確認。
  - `rffmg_dir` が `--sampling_num` 未指定→`5times_sampling`、`=10`→`10times_sampling` になることを静的に確認。
  - 本体（分子処理）は実行しない（既存データ保護）。
- **Dependencies**: after Step 3

## Notes / スコープ外・別件

- **safe 系パス**（L59, L105–106, L118–119）は変更しない（data/safe に times_sampling 階層が無いため）。
- **別件（今回のスコープ外・要判断）**: `src/gen_mols/gen_rffmg.py` L40 `f'{BASEPATH}/data/rffmg/{frag_method}/{additional_path}'` も times_sampling 未対応。実データはこの下にあるため本来は同様の修正が必要。今回の指示（make_datasets.py）には含めないが、対応要否をユーザーに確認する。
- worktree はユーザー方針によりメインで直接作業。
