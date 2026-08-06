# Plan: gen_safe_denovo.py のプロンプトを pass_safe から encoder(pass_fragments) に変更

- **Date**: 2026-08-01
- **Status**: pending-approval (rev.2 — 不要な Step を削除)

## Overview

`src/gen_safe_denovo.py` のプロンプトを、test split の `pass_safe` 列から
**`pass_fragments` を `safe.SAFEConverter` でエンコードしたもの**に変更する。

### 変更理由

`pass_safe` は元分子の断片対応関係を保持しており、タスクの答えを部分的に与えてしまっている。

`convert2safe`（`src/func/fragment_for_safe.py:54-65`）は同位体ラベル付きダミー原子 `[N*]` を
拾って番号を振るため、**同じ `[1*]` を持つ2つの断片には同じ閉環数字が入る**。つまり
「元の分子でどの結合手とどの結合手が繋がっていたか」が埋め込まれている。

一方 `pass_fragments` は `convert_dummy_atoms_rdkit`（`src/gen_frags/safe_frags.py:18-33`）で
同位体を除去済みの匿名 `*` なので、その情報を持たない。

```
pass_fragments: *C1Sc2ccccc2C(=O)C1=C.*c1ccccc1        ← * は2つ、どちらも匿名

pass_safe     : C13Sc2ccccc2C(=O)C1=C.c13ccccc1   未閉鎖[]     ← 両方「3」= 繋がる相手が確定
encoder出力   : C14Sc2ccccc2C(=O)C1=C.c13ccccc1   未閉鎖[3,4]  ← どちらも開放端
```

### 他ベースラインとの公平性

| 手法 | 入力 | 対応関係の情報 | 参照 |
|---|---|---|---|
| FragGPT | `pass_fragments` | なし | `generation_fraggpt_func.py:97` |
| PromptSMILES | `pass_fragments` | なし | `gen_promptsmiles.py:122` |
| RFFMG | `test.source`（同位体除去形） | なし | `make_datasets.py:87-90` |
| **SAFE（現状）** | **`pass_safe`** | **あり** | — |

タスクが「指定した断片から分子を生成する」である以上、断片の接続先はモデルが決めるべきで、
`pass_fragments` を入力とするのが正しい。

### 実測（brics / finetuning、40行 × 8本 = 320系列）

| プロンプト | 全閉鎖 | 単一分子 | validity |
|---|---|---|---|
| `pass_safe`（現状） | 32.8% | 32.8% | 95.6% |
| **`encoder(pass_fragments)`** | **16.2%** | **15.0%** | **93.4%** |

16.2% は性能低下ではなく、公平な条件での実力値。

### エンコード方法の確定事項（実測済み）

- `SAFEConverter` は既定で `slicer="brics"`（`converter.py:61`）。公開APIは全て
  `do_not_fragment_further=True` を既定とし、`sf.utils.attr_as(encoder, "slicer", None)` で
  一時的に無効化している（`sample.py:911`）。
- **`SAFEConverter(slicer=None)` で直接構築した場合と `attr_as` 版は 100/100 で一致**（実測）。
- `randomize=False` とする。`_completion` は `randomize=True`（試行ごとの拡張）だが、
  本スクリプトは1回の決定的な生成なので不要。再現性のためにも固定する。
- **encoder 失敗は 3,000 行中 0 件**（実測）。専用の例外処理は設けず、既存のバッチ単位
  `try/except` に任せる。

### 検討して見送ったもの

- **RDKit ログの抑止**: encoder の stderr は 200行あたり17行（`unclosed ring` 8件）で、
  82,441行でも約7,000行。実行時間から見て誤差。`RDLogger.DisableLog('rdApp.*')` は
  全ログを止めるため、本当に見たい警告まで消える。**入れない**。
- **ループ前の一括エンコードと失敗行の分離処理**: encoder 失敗が実測0件のため、
  専用処理を書く価値がない。**入れない**。

## Plan

対象ファイルは **`src/gen_safe_denovo.py` のみ**。他ファイルは変更しない。

### Step 1: エンコーダをモジュールレベル定数として追加する

- **Target file**: `src/gen_safe_denovo.py`（import 直後）
- **Changes**: 以下の1行を追加する。

  ```python
  ENCODER = safe.SAFEConverter(slicer=None)
  ```

  `slicer=None` は `SAFEDesign` の `do_not_fragment_further=True` に相当する。
  ラッパー関数は作らず、呼び出し側でそのまま使う。
- **Dependencies**: none

### Step 2: バッチループのプロンプトを差し替える

- **Target file**: `src/gen_safe_denovo.py`（バッチループ内）
- **Changes**: `prefixes = batch_df['pass_safe'].tolist()` を次に置き換える。

  ```python
  prefixes = [ENCODER.encoder(f, canonical=False, randomize=False,
                              constraints=None, allow_empty=True)
              for f in batch_df['pass_fragments']]
  ```

  既存の `try:` の**中**に置く（encoder が例外を投げたらバッチ単位の except が拾う）。
- **Dependencies**: after Step 1

### Step 3: 出力カラムを実際のプロンプトに差し替える

- **Target file**: `src/gen_safe_denovo.py`（バッチループと `base_df` の組み立て）
- **Changes**:
  - `prefixes` は生成結果と同様にループ外のリストへ蓄積する（`generated_safe` と同じ扱い）。
  - `base_df` の `pass_safe` を、蓄積したプロンプト列 `prompt_safe` に差し替える。
  - 変更後の出力カラム: `['target', 'prompt_safe', 'fragment']` + `safe_1..N` + `prediction_1..N`
  - `error_logs` の `pass_safe` 列は `pass_fragments` に変える（失敗したのはエンコード前の入力のため）。
- **Dependencies**: after Step 2

### Step 4: 構文チェック

- **Target file**: なし（検証のみ）
- **Changes**:
  - `python -m py_compile src/gen_safe_denovo.py`
  - `grep -n "pass_safe" src/gen_safe_denovo.py` が空になることを確認する。
- **Dependencies**: after Step 3

## 確認したい点

- **`prompt_safe` 列を出力に残すか**。デバッグ用途がなければ削って
  `['target', 'fragment']` + `safe_1..N` + `prediction_1..N` にもできる。

## Out of scope

- `src/func/generation_safe_func.py`、他のベースラインスクリプト
- 制約付きデコード（`DisjunctiveConstraint`）— 手法の性能をそのまま測る方針のため行わない
- コミット（指示があるまで行わない）
