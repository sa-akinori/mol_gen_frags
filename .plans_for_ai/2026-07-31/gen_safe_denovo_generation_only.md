# Plan: gen_safe_denovo.py を生成専用にする（評価・比較の削除）

- **Date**: 2026-07-31
- **Status**: pending-approval

## Overview

`src/gen_safe_denovo.py` は現在、direct / library の 2 経路で de novo 生成を行い、
Validity を計算して比較サマリ（`comparison.csv`）を出力する「比較スクリプト」になっている。
これを **生成のみを行うスクリプト** に変更する。

削除するのは「評価（Validity 計算）」と「2 経路を回して比較する」部分。
生成経路そのものは 2 つとも残し、引数 `method` で選択する形にする
（比較はしないが、どちらの経路でも生成できる状態は維持する）。

### 事前調査

| 確認項目 | 結果 |
|---|---|
| `run_denovo_comparison` / `compute_validity` の外部呼び出し | なし（同ファイル内のみ） |
| `comparison.csv` / `generated.csv` を読む後続コード | なし（`src/evaluation.py` が読むのは `predictions.csv`） |
| `.sh` からの呼び出し | なし |
| 既存の出力ディレクトリ | `results/safe/gpt/pretrained/` は未作成（実行実績なし） |

### 残すもの・消すもの

- **残す**: `_canonicalize_smiles` / `_decode_safe_smiles`
  （Validity の指標計算ではなく、SMILES バリデーション + canonical 化。
  CLAUDE.md の「SMILES のバリデーションは省略しない」に該当するため生成側の責務として残す）
- **残す**: `generate_denovo_direct` / `generate_denovo_library`（生成本体、変更なし）
- **消す**: `compute_validity`（評価指標）
- **消す**: 2 経路を for で回す比較ループ、`comparison.csv` の出力、サマリの print

## Plan

### Step 1: `compute_validity` を削除する

- **Target file**: `src/gen_safe_denovo.py`（L98-113）
- **Changes**: 関数 `compute_validity` を丸ごと削除する。他から参照されていないことは確認済み。
- **Dependencies**: none

### Step 2: 生成経路のディスパッチ表を追加する

- **Target file**: `src/gen_safe_denovo.py`（`generate_denovo_library` の直後）
- **Changes**: モジュールレベルに以下を追加する。

  ```python
  GENERATORS = {"direct": generate_denovo_direct, "library": generate_denovo_library}
  ```

  比較ループ内のローカル辞書 `methods` を廃し、経路の選択肢を 1 箇所で定義する。
- **Dependencies**: none

### Step 3: `run_denovo_comparison` を `run_denovo_generation` に置き換える

- **Target file**: `src/gen_safe_denovo.py`（L116-166）
- **Changes**: 比較用関数を、単一経路で生成して保存するだけの関数に置き換える。

  ```python
  def run_denovo_generation(
      model_path: str | Path,
      output_dir: str | Path,
      method: str = "direct",
      n_samples: int = 1000,
      max_length: int = 200,
      random_seed: int = 42,
  ) -> pd.DataFrame:
      """指定した経路で de novo 生成し、結果を CSV に保存する。

      Args:
          model_path: 学習済み SAFE-GPT モデルのパス。
          output_dir: 生成結果の保存先ディレクトリ（配下に <method>/generated.csv を作る）。
          method: 生成経路。'direct'（model.generate を直接呼ぶ）または
              'library'（SAFEDesign.de_novo_generation を使う）。
          n_samples: 生成本数。
          max_length: 生成配列の最大長。
          random_seed: 乱数シード（生成前に設定）。

      Returns:
          pd.DataFrame: 生成結果。カラムは ['raw_index', 'smiles']
              （'smiles' は canonical SMILES。妥当でない分子は欠損値）。
      """
  ```

  処理の中身:
  1. `method` が `GENERATORS` にない場合は `ValueError` を送出する。
  2. モデル / トークナイザのロード、`device` 決定、`SAFEDesign` 構築は現行のまま流用する。
  3. `set_seed(random_seed)` の後、選択した 1 経路のみ実行する。
  4. `Path(output_dir) / method` を作成し `generated.csv` を保存する（現行と同じ配置。
     経路を変えて実行しても上書きされない）。
  5. `print` は保存先の通知のみとし、Validity の集計・表示は行わない。
  6. 戻り値は生成結果の DataFrame（比較サマリではない）。
- **Dependencies**: after Step 1, Step 2

### Step 4: `__main__` を単一経路の実行に合わせる

- **Target file**: `src/gen_safe_denovo.py`（L169-179）
- **Changes**: `run_denovo_comparison(...)` の呼び出しを `run_denovo_generation(...)` に変更し、
  `method` をファイル冒頭の他スクリプト（旧 `gen_safe.py` 等）と同じくローカル変数として明示する。

  ```python
  if __name__ == '__main__':
      fd = Path(__file__).resolve().parent.parent
      method = 'direct'
      model_path = fd / 'models' / 'safe' / 'gpt' / 'pretrained'
      output_dir = fd / 'results' / 'safe' / 'gpt' / 'pretrained' / 'denovo'
      run_denovo_generation(
          model_path=model_path,
          output_dir=output_dir,
          method=method,
          n_samples=1000,
          max_length=200,
          random_seed=42,
      )
  ```
- **Dependencies**: after Step 3

### Step 5: 構文チェック

- **Target file**: なし（検証のみ）
- **Changes**: `python -m py_compile src/gen_safe_denovo.py` が通ることを確認する。
  実際の生成実行（GPU・モデル要）は行わない。
- **Dependencies**: after Step 4

## 確認したい点

- **生成経路の扱い**: 上記は「direct / library の両方を残し、`method` 引数で選択」する案です。
  もし「一方の経路だけ残して他方も削除する」意図であれば、どちらを残すかご指定ください。
- **CLI 化**: `src/gen_mols/` 配下のスクリプトは `argparse` 化されていますが、本ファイルは
  現行どおりハードコードのままとしています（今回の指示の範囲外のため）。CLI 化も必要なら追加します。

## Out of scope（今回は触らない）

- `generate_denovo_direct` / `generate_denovo_library` の生成ロジック本体
- `src/evaluation.py` など他ファイル
- コミット（指示があるまで行わない）
