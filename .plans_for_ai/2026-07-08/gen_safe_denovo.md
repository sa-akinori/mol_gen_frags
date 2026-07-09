# Plan: pretrained SAFE-GPT による de novo 生成（直接 vs de_novo_generation 比較）

- **Date**: 2026-07-08
- **Status**: pending-approval

## Overview

pretrained SAFE-GPT（ローカル保存: `models/safe_gpt/pretrained/`）を用いて de novo 分子生成を行う。
2 通りの生成経路を比較する:

1. **direct**: `model.generate(inputs=None, do_sample=True, ...)` を直接呼び、`safe.decode(fix=True, ...)` でデコードする。
2. **library**: `SAFEDesign.de_novo_generation(how="random", ...)` を用いる。

公平な比較のため、両者は **同一の `SAFEDesign` インスタンス**を共有し、同じ model / tokenizer / `generation_config` /
サンプリング設定（`do_sample=True`, 同一 `max_length`, 同一 `n_samples`, 同一 seed）を使う。差分は生成コード経路のみ。

指標は **Validity（妥当率）のみ**。de novo なので参照データセットは不要（Novelty/Uniqueness は算出しない）。
また de novo は slice に依存しないため、モデル・出力パスに slice_name は含めない。

### de_novo_generation の工夫（確認結果）

`SAFEDesign.de_novo_generation` (`safe/sample.py:726`) は薄いラッパーで、本質は `_generate`/`_decode_safe`:
- `how="random"` → `do_sample=True`（de novo は多様性のため多項サンプリング）
- プレフィックス無し = `inputs=None` で BOS から生成
- デコード時 `fix=True`（壊れた SAFE 文字列を修復）+ `remove_dummies=True` + `canonical=True`
- `generation_config` に bos/eos/pad を補完、`token_type_ids` 除去

direct 方式でもこの工夫（①do_sample=True ②inputs=None ③fix=True デコード）を再現する。

## Plan

### Step 1: 新規ファイル `src/gen_safe_denovo.py` の骨格とデコードヘルパー

- **Target file**: `src/gen_safe_denovo.py`（新規作成）
- **Changes**:
  - import（標準 → サードパーティ → ローカル `func.utility.set_seed`）。`os.environ['CUDA_VISIBLE_DEVICES']='0'`。
  - `_decode_safe_smiles(seq: str) -> str | None`:
    `safe.decode(seq, as_mol=False, fix=True, remove_added_hs=True, canonical=True, ignore_errors=True, remove_dummies=True)`
    後に `Chem.MolFromSmiles` の None チェックを行い、canonical SMILES か `None` を返す。
- **Dependencies**: none

### Step 2: 2 つの生成関数

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:
  - `generate_denovo_direct(designer: SAFEDesign, n_samples: int, max_length: int) -> list[str | None]`:
    `designer.tokenizer.get_pretrained()` で HF トークナイザを取得し、
    `designer.model.generate(inputs=None, generation_config=designer.generation_config, do_sample=True,
    num_return_sequences=n_samples, max_length=max_length, early_stopping=True)` を実行。
    `batch_decode(skip_special_tokens=True)` → `_decode_safe_smiles` で SMILES 化したリストを返す。
  - `generate_denovo_library(designer: SAFEDesign, n_samples: int, max_length: int) -> list[str | None]`:
    `designer.de_novo_generation(n_samples_per_trial=n_samples, sanitize=False, how="random",
    max_length=max_length)` を呼ぶ。返り値（既に SMILES、invalid は None）を canonical 化して揃える。
- **Dependencies**: after Step 1

### Step 3: 指標計算（Validity のみ）

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:
  - `compute_validity(smiles_list: list[str | None]) -> dict[str, float | int]`:
    - `n_valid` = None でない要素数
    - `validity` = n_valid / n_total（n_total が 0 なら 0.0）
    - 戻り値に `n_total, n_valid, validity` を含む。docstring に戻り値 dict のキーを明記。
- **Dependencies**: after Step 1

### Step 4: 比較ドライバ

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:
  - `run_denovo_comparison(model_path, output_dir, n_samples=1000, max_length=200,
    random_seed=42) -> pd.DataFrame`:
    - `SAFEDesign(model=..., tokenizer=...)` を 1 つ生成（model/tokenizer は from_pretrained、device 移動、eval）。
    - 各手法ごとに `set_seed(random_seed)`（再現性）→ 生成 → `compute_validity`。
    - 各手法の生成 SMILES を `{output_dir}/{method}/generated.csv`（列: `raw_index, smiles`）に保存。
    - 比較サマリを `{output_dir}/comparison.csv`（列: `method, n_total, n_valid, validity`）に保存し、
      print で表示。DataFrame を返す。
    - `os.makedirs(..., exist_ok=True)`。
- **Dependencies**: after Steps 2, 3

### Step 5: `__main__` ブロック

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:
  - `gen_safe.py` に倣い `fd = os.path.dirname(os.path.dirname(__file__))`。
  - `model_path=f'{fd}/models/safe_gpt/pretrained'`（ローカル保存の datamol-io/safe-gpt）、
    `output_dir=f'{fd}/results/safe_gpt/pretrained/safe/denovo/'`（slice 非依存）。
  - `run_denovo_comparison(model_path=..., output_dir=..., n_samples=1000, max_length=200, random_seed=42)` を呼ぶ。
- **Dependencies**: after Step 4

## Notes / 確認事項

- 実行環境は conda `safe`（`safe-mol`, `torch`, `rdkit`）。src 内は既存同様 `from func.utility import set_seed`
  なので `src/` をカレントにして実行する想定（既存 `gen_safe.py` と同じ）。
- 既定 `n_samples=1000` は de novo の 1 バッチ生成本数。
- 指標は Validity のみ（参照データセット不要、slice 非依存）。
