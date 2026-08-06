# Plan: gen_safe_denovo.py を generation_safe_func.py として本番経路に組み込む

- **Date**: 2026-08-01
- **Status**: pending-approval

## Overview

`src/gen_safe_denovo.py` の内容を `src/func/generation_safe_func.py` として新規作成し、
`src/gen_mols/gen_safe.py` → `run_and_record_time` → subprocess の本番経路に接続する。
旧実装は既にユーザーによって `generation_safe_func_old.py` にリネーム済み。

### ユーザー確定事項

| 論点 | 決定 |
|---|---|
| 引数方式 | `--frag_method` / `--model_ver` を受け取り、`BASEPATH` からパスを組み立てる |
| シャーディング | **不要**。`gen_mols/gen_safe.py` からも `--machine_id` / `--total_machines` を削除 |
| 生成時間の記録 | `gen_safe.py` 側の `run_and_record_time` のみ（func 側には何も追加しない） |
| `src/gen_safe_denovo.py` | **削除する** |

### 現状の食い違い（解消対象）

| 項目 | 現状 |
|---|---|
| 引数 | `gen_safe.py` は `--model_path` / `--dataset_dir` / `--output_dir` / `--machine_id` / `--total_machines` を渡すが、`gen_safe_denovo.py` は受け取らない |
| `--batch_size` | `gen_safe.py` が渡さない。既定8では OOM（実測40行が失敗）。2 で OOM ゼロ |
| 行数制限 | `test_df.head(1000)`（L104）がハードコードされている。本番では全行必要 |
| `--n_generate` | 引数は定義されているが未使用（デッドコード） |
| 出力名 | スクリプトは `predictions.csv` を書くが、`run_and_record_time` は `predictions_{machine_id}.csv` を数える設定 |
| 引数名 | `gen_safe.py` は `--model_ver`、`gen_safe_denovo.py` は `--model_version` |

### 既知の設計上の割り切り

`--frag_method` / `--model_ver` 方式を採るため、**パスの組み立てロジックが2箇所に重複**します。

- `generation_safe_func.py`: `model_path` / `dataset_dir` / `output_dir` を組み立てて実行に使う
- `gen_mols/gen_safe.py`: `output_dir` を組み立てて `run_and_record_time` に渡す（時間記録の保存先とし
  て必要）

`--model_path` 等を明示的に渡す方式ならこの重複は避けられますが、ユーザー判断により本方式を採ります。
両者の規則が食い違うと時間記録が別ディレクトリに出るため、**同じ規則であることを保つ必要があります**。

## Plan

### Step 1: `src/func/generation_safe_func.py` を新規作成する

- **Target file**: `src/func/generation_safe_func.py`（新規）
- **Changes**: 現行 `src/gen_safe_denovo.py` の内容をベースに、以下を変更して作成する。

  1. **モジュール docstring を追加**する（`generation_rffmg_func.py:1-11` と同じ粒度）。
     - `SAFEDesign` の高レベル生成APIを使わず `model.generate()` を直接呼ぶこと
     - プロンプトは `pass_fragments` を `SAFEConverter(slicer=None)` でエンコードしたもの
     - 出力は `predictions.csv`（`target` / `prompt_safe` / `fragment` / `prediction_1..N`）
  2. **引数名を `--model_version` → `--model_ver`** に変更する（`gen_safe.py` / `gen_fraggpt.py` と統一）。
  3. **`--batch_size` の既定値を 2 にする**（実測: 8 は OOM、2 は OOM ゼロ）。
  4. **`test_df.head(1000)`（L104）を削除**し、`--n_generate` を実装する。

     ```python
     test_df = datasets.load_from_disk(dataset_dir)["test"].to_pandas()
     if args.n_generate is not None:
         test_df = test_df.head(args.n_generate)
     ```

     既定 `None` で全行。小規模確認用に残す。
  5. **`--random_seed` を削除**する。`do_sample=False` のビームサーチは決定的で、現在 `set_seed` も
     呼んでいないため完全に効かない引数になっている。provenance は `gen_safe.py` の `params` に残す。
  6. 関数（`decode_safe_smiles` / `encode_prefixes`）と生成ロジックは**現状のまま移植**する。
     - `encoder.encoder(f, canonical=True, randomize=False, constraints=None, allow_empty=True)`
     - `num_beams` / `n_samples` / `max_length` / `early_stopping=True`
     - バッチ単位の `try/except` と `error_logs.csv`
  7. 出力は `predictions.csv` / `error_logs.csv`（シャード接尾辞なし）。
- **Dependencies**: none

### Step 2: `src/gen_mols/gen_safe.py` を新しい引数に合わせる

- **Target file**: `src/gen_mols/gen_safe.py`
- **Changes**:
  1. `--machine_id` / `--total_machines` の `add_argument` を**削除**する。
  2. `--batch_size`（既定 2）を**追加**する（`gen_fraggpt.py:37` と同じ様式）。
  3. `cmd` を新しい引数に合わせる。

     ```python
     cmd = [
         "python", f"{BASEPATH}/src/func/generation_safe_func.py",
         "--frag_method", frag_method,
         "--model_ver", model_ver,
         "--n_samples", str(args.n_samples),
         "--max_length", str(args.max_length),
         "--num_beams", str(args.num_beams),
         "--batch_size", str(args.batch_size),
     ]
     ```

     `--model_path` / `--dataset_dir` / `--output_dir` / `--random_seed` /
     `--machine_id` / `--total_machines` は渡さない。
  4. `run_and_record_time` の呼び出しを変更する。
     - `record_name="generation_time.json"`（既定値なので引数ごと削除可）
     - `predictions_pattern="predictions.csv"`
     - `params` から `machine_id` / `total_machines` を削除し、`batch_size` を追加する
  5. `dataset_dir` はもう使わないので削除する。`model_path` は `params` の provenance に使うため残す。
     `output_dir` は `run_and_record_time` に渡すため残す。
- **Dependencies**: after Step 1

### Step 3: `src/gen_mols/gen_safe.sh` に `--batch_size` を追加する

- **Target file**: `src/gen_mols/gen_safe.sh`
- **Changes**: 実行行に `--batch_size 2` を追加する。`--random_seed 42` は `gen_safe.py` が
  `params` の provenance に使うので残す。

  ```bash
  python ${SCRIPT_DIR}/gen_safe.py --frag_method ${FRAG_NAME} --model_ver ${MODEL_VER} \
      --n_samples 50 --num_beams 50 --batch_size 2 --random_seed 42
  ```
- **Dependencies**: after Step 2

### Step 4: `src/gen_safe_denovo.py` を削除する

- **Target file**: `src/gen_safe_denovo.py`（削除）
- **Changes**: `generation_safe_func.py` に一本化されるため削除する。git 履歴には残る。
- **Dependencies**: after Step 1

### Step 5: 検証

- **Target file**: なし（検証のみ）
- **Changes**:
  - `python -m py_compile src/func/generation_safe_func.py src/gen_mols/gen_safe.py`
  - `gen_safe.py` が渡す引数と `generation_safe_func.py` の `add_argument` が
    過不足なく一致することを確認する
  - `--n_generate 20` 程度で実際に1回実行し、`predictions.csv` /
    `error_logs.csv` / `generation_time.json` の3つが期待どおり出ることを確認する
    （`--n_generate` は `gen_safe.py` 経由では渡せないため、func を直接叩いて確認する）
- **Dependencies**: after Step 3, Step 4

## 確認したい点

- **`--random_seed` の削除**: `do_sample=False` のビームサーチでは完全に無効な引数のため、
  func からは削除する案にしている。`gen_safe.py` 側は `params` の provenance 用に残す。
  ただし他ベースライン（rffmg / fraggpt）は実際に `set_seed` を呼んでいるため、
  「インターフェースを揃える」観点で残す選択もありうる。
- **`--batch_size` の既定値 2**: RTX 4090（24GB）での実測に基づく。他マシンでは調整が必要。
- **`--n_generate` を `gen_safe.py` にも通すか**: 現案では func のみが持つ。
  `gen_safe.py` からも小規模実行したい場合は追加する。

## Out of scope

- `src/func/generation_safe_func_old.py`（参照用に残す。削除しない）
- 他のベースラインスクリプト、評価パイプライン
- コミット（指示があるまで行わない）
