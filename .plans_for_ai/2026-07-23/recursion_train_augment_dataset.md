# Plan: 成功生成データによる訓練データ拡張（recursion_train, 既存関数再利用・.sh実行）

- **Date**: 2026-07-23
- **Status**: approved

## 承認時の決定事項

- `out_name` の既定は `augmented`。
- `.sh` の既定 `FRAG_METHOD` は `brics`。冒頭変数のコメントで `rc_cms` / `brics` を切替可能にする。
- `load_file`/`save_file` は `make_datasets` から import して再利用する。

## Overview

frag_num / dup_frags / attach_point_num の3シナリオ（`make_datasets.py` が生成する、
訓練データに存在しないフラグメントセットのロバスト性評価用テストセット）で、
t5chem trained モデルが「成功」した生成物を抽出し、現在の訓練データ（`normal/train.*`）に
追加した拡張データセットを新規ディレクトリに構築する。学習の実行は含めず、データ準備のみ。

コードは新規フォルダ `src/recursion_train/` に配置。**新規関数は極力作らず既存関数を再利用**し、
抽出・構築ロジックは `make_datasets.py` と同様に `__main__` 配下へインライン記述する。
実行は既存同様 `.sh` ラッパで行い、設定は `.sh` 内の変数として持つ（`gen_rffmg.sh` と同スタイル）。

### 確定した仕様（ユーザー回答）

1. **成功の定義**: `curated_data.tsv` の `valid_smis_on_frags`
   （RDKitで有効 かつ 入力フラグメントを全て含む分子）。
2. **採用件数**: 1フラグメントセットあたり **ランダムに n_select 件**（既定5、未満なら全件）。
   再現性のため乱数シードを固定。
3. **保存先**: `normal` を上書きせず新規ディレクトリに保存。
4. **スコープ**: データ準備のみ。生成元は t5chem trained の結果。
5. **実行方法**: `.sh` ラッパ（他ファイル同様）。config は不要。
6. **実装方針**: 既存関数を再利用し、新規関数は作らない。

### 再利用する既存関数・資産（新規に作らない）

| 用途 | 再利用するもの | 定義元 |
|------|----------------|--------|
| 改行区切りファイルの読込 | `load_file(file_name)` | `src/make_datasets.py` |
| 改行区切りファイルの書込 | `save_file(target, save_file)` | `src/make_datasets.py` |
| パスのベース | `BASEPATH` | `src/func/utility.py` |
| CSV/TSV 読込・DataFrame・CSV書込 | `pandas` | 既存依存 |
| リスト文字列のパース | `ast.literal_eval` | 標準 |
| 乱数抽出 | `random.Random` | 標準 |

- `load_file`/`save_file` は `make_datasets.py` にのみ存在するため、そこから import して再利用する
  （`make_datasets` の処理は全て `if __name__=='__main__'` 配下なので import による副作用はない）。

### 設計上の判断（レビュー対象）

- **ランダム抽出**: 各行の `valid_smis_on_frags` を `ast.literal_eval` でリスト化し、
  `rng.sample(smis, min(n_select, len(smis)))` で無作為抽出。`predictions.csv` は不使用。
  全行で単一 `random.Random(seed)` を順に消費して決定的にする。
- **重複除去**: 追加ペア間の完全重複（source+target一致）を除去し、`normal/train` に既存の
  ペアは追加しない。除去件数はログに残す。
- **val/test**: `normal` の val/test を `load_file`→`save_file` でそのままコピー（拡張は train のみ）。
- **新規依存なし**: 標準 + pandas + 既存 `func`/`make_datasets` のみ。

## Plan

### Step 1: 抽出・構築スクリプトの作成（インライン実装・新規関数なし）

- **Target file**: `src/recursion_train/build_augmented_dataset.py`（新規）
- **Changes**:
  - モジュール docstring を付す。import順は stdlib→third-party→local。
  - サブフォルダから import するため先頭で
    `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`（= `src/`）。
  - import: 標準 `argparse, os, ast, random`, `pathlib.Path`；third-party `pandas as pd`；
    local `from func.utility import BASEPATH`, `from make_datasets import load_file, save_file`。
  - **新規の名前付き関数は定義しない**（`make_datasets.py` と同じくロジックは `__main__` にインライン）。
- **Dependencies**: none

### Step 2: `__main__` — argparse と抽出・構築処理（インライン）

- **Target file**: `src/recursion_train/build_augmented_dataset.py`
- **Changes**:
  - argparse 引数（既存スクリプトの命名に合わせる）:
    - `--frag_method`（required, choices=['rc_cms','brics']）
    - `--model_name`（default 't5chem'）, `--model_ver`（default 'trained'）
    - `--scenarios`（nargs='+', default=['frag_num','dup_frags','attach_point_num']）
    - `--n_select`（type=int, default=5）, `--out_name`（default 'augmented'）, `--seed`（type=int, default=0）
  - パス: `results_dir = f'{BASEPATH}/results/{model_name}/{model_ver}/rffmg/{frag_method}/beam/{scenario}'`、
    `data_dir = f'{BASEPATH}/data/rffmg/{frag_method}'`。
  - `rng = random.Random(seed)`。
  - シナリオごとに `pd.read_csv(f'{results_dir}/curated_data.tsv', sep='\t', index_col=0)` を読み、
    行を走査（`nvalid_onfrags == 0` はスキップ）。`ast.literal_eval(row['valid_smis_on_frags'])` を
    `rng.sample(..., min(n_select, len))` で抽出し、`(source=row['fragment'], target=smi, scenario)` を蓄積。
  - 追加ペアを `pd.DataFrame`（カラム `source, target, scenario`）化し、`drop_duplicates(['source','target'])`。
  - `load_file(f'{data_dir}/normal/train.source')` と `.target` から既存 (source,target) 集合を作り、
    既存ペアを除外（`~` フィルタ）。
  - 拡張 train:
    - `save_file('\n'.join(train_source_lines + added['source'].tolist()) + '\n', f'{data_dir}/{out_name}/train.source')`
    - target 側も同様に `save_file(...)`。
  - `normal/{val,test}.{source,target}` を `save_file('\n'.join(load_file(src)) + '\n', dst)` でコピー。
  - `added.to_csv(f'{data_dir}/{out_name}/added_pairs.csv')`。
  - `save_file(...)` で `augmentation_log.txt` にパラメータ（frag_method, model, n_select, seed, scenarios）と
    メトリクス（シナリオ別採用件数・重複除去件数・normal件数・最終train件数）を記録。
  - `os.makedirs(f'{data_dir}/{out_name}', exist_ok=True)` を書込前に実行（既存スクリプト準拠）。
  - map/filter 的処理は lambda を用いる（コードスタイル規約）。
- **Dependencies**: after Step 1

### Step 3: 実行用シェルスクリプト（.sh ラッパ）

- **Target file**: `src/recursion_train/build_augmented_dataset.sh`（新規）
- **Changes**: `src/gen_mols/gen_rffmg.sh` と同形式。冒頭に設定変数
  （`FRAG_METHOD`, `MODEL_NAME`, `MODEL_VER`, `SCENARIOS`, `N_SELECT`, `OUT_NAME`, `SEED`）を置き、
  `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` を解決して
  `python ${SCRIPT_DIR}/build_augmented_dataset.py --frag_method ${FRAG_METHOD} ...` を実行。
  `FRAG_METHOD` は既定 `brics`、コメントで `# "brics" or "rc_cms"` と明記しユーザーが切替可能にする
  （`gen_rffmg.sh` と同じ体裁）。
- **Dependencies**: after Step 2

## 実行後の確認（実装後の軽い検証）

- `<out_name>/train.source` と `train.target` の行数一致を確認。
- `added_pairs.csv` 件数が「シナリオ別採用件数合計 − 重複除去」と整合するか確認。
- 追加 target を数件 RDKit で有効性チェック。

## 確認したい点（承認時に指定があれば反映）

- `out_name` の既定 `augmented` で良いか（例: `recursion` など希望あれば変更）。
- `.sh` の既定 `FRAG_METHOD` は `rc_cms` / `brics` どちらにするか。
- `make_datasets` からの import で良いか（`load_file`/`save_file` はそこにのみ定義。共通 util へ切り出す案もあるが、その場合 `src/` の既存ファイル変更が必要）。
