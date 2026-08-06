# Plan: PromptSMILES の学習データ重複排除を smiles 基準に修正

- **Date**: 2026-08-02
- **Status**: pending-approval

## Overview

`src/make_datasets.py` の PromptSMILES 出力が SAFE の `full_safe` 基準の重複排除を流用しているため、
**rc_cms で同一分子が 3.51 回重複**して学習コーパスに入る。

### 原則（ユーザー判断）

学習データの重複排除は「**その手法が実際に生成する表現**」の列で行う。

| 手法 | 生成するもの | dedup 対象 | 現状 |
|---|---|---|---|
| SAFE | SAFE 文字列 | `full_safe` | 正しい |
| FragGPT | FU-SMILES | `full_fragments` | 正しい |
| **PromptSMILES** | **プレーンな SMILES** | **`smiles`** | **誤り（`full_safe` 由来）** |

SAFE と FragGPT は SMILES を生成しないので、同じ表現に潰れる分子（二重結合の切断で E/Z が消えた
立体異性体など）はモデルにとって区別不能であり、表現で dedup するのが正しい。
一方 PromptSMILES はプレーンな SMILES を生成するため、`smiles` で dedup しなければならない。

### 実測（`make_datasets.py` と同一手順を断面で再現）

| | train 行数 | ユニーク分子数 | 1分子あたり |
|---|---|---|---|
| brics | 16,279 | 16,279 | 1.00（問題なし） |
| **rc_cms** | **41,764** | **11,913** | **3.51（重複）** |

brics は BRICS 分解が決定的で 1分子1パターンのため露見しない。
rc_cms は `RandomFragmentize` が分子ごとに複数パターンを作るため重複する。

### なぜ問題か

`.plans_for_ai/2026-07-28/add_promptsmiles_baseline.md` で明示的に決めた前提が崩れる。

> **Augmentation: 行わない（ユーザー判断）**。**1分子 = 1系列**とし、学習データはカノニカルSMILESのみ。
> 理由: データ量が他手法に対して極端に多くなり、公平な比較にならないため。

rc_cms では実質 3.51 倍の水増しとなり、同一分子を繰り返し学習することになる。

## Plan

### Step 1: PromptSMILES の train / validation を smiles 基準の dedup に変更

- **Target file**: `src/make_datasets.py`
- **Changes**: PromptSMILES ブロック（`promptsmiles_train` / `promptsmiles_valid`）の
  `Dataset.from_pandas(...)` に `drop_duplicates('smiles').reset_index(drop=True)` を追加する。

  ```python
  promptsmiles_train = Dataset.from_pandas(safe_tr.loc[:, ['smiles', 'pass_fragments']].drop_duplicates('smiles').reset_index(drop=True))
  promptsmiles_valid = Dataset.from_pandas(safe_val.loc[:, ['smiles', 'pass_fragments']].drop_duplicates('smiles').reset_index(drop=True))
  ```

  - **`promptsmiles_test` は変更しない。** 生成の入力は `pass_fragments` であり、
    RFFMG・SAFE・FragGPT と行を揃える必要があるため（1分子あたり最大5行のまま）。
  - なぜ `full_safe` ではなく `smiles` で dedup するのかを**1行コメントで残す**
    （PromptSMILES はプレーンな SMILES を生成するため、という非自明な理由）。
  - SAFE と FragGPT の出力ブロックは**一切変更しない**。
- **Dependencies**: none

### Step 2: `train_promptsmiles.py` の docstring を実態に合わせる

- **Target file**: `src/train_model/train_promptsmiles.py`
- **Changes**: `read_corpus()` の docstring にある以下の記述が brics での計測値であり、
  rc_cms には当てはまらないため修正する。

  > both splits hold one molecule per row (measured: train 1,714,298 rows / 1,714,298 molecules,
  > validation 45,203 rows / 45,203 molecules), so no deduplication is needed.

  - 「重複排除は不要」ではなく「`make_datasets.py` が `smiles` で重複排除済みなので 1分子1行」
    という記述に改める。
  - 計測値が brics のものであることを明示するか、frag_method 非依存の書き方にする。
  - **学習ロジックは一切変更しない。**
- **Dependencies**: after Step 1

### Step 3: `next_plan.md` に立体比較の課題を追記

- **Target file**: `next_plan.md`
- **Changes**: 「保留中の項目」に以下を追加する。
  - **評価の立体比較が表現ベース手法に不利に働く**（未対応・今後実装予定）
    - `evaluation_func.py` の `Smi2CanSmi` は `Chem.MolToSmiles(mol)`（`isomericSmiles=True` が既定）
      を使うため、立体を含めて学習分子・正解分子と比較している
    - SAFE と FragGPT は二重結合を切った箇所の E/Z を復元できない
      （FragGPT の組立の実測: 立体無視 100% / 立体込み 86.1%）
    - 実測（brics）: 学習分子の 30.7%、test 分子の 36.4% が立体情報を持つ
    - 影響: 生成分子の立体が落ちると学習分子と一致しないため **novelty が過大**、
      正解分子とも一致しないため **top-k accuracy が過小**に出る。
      SMILES を直接生成する RFFMG は影響を受けないため、手法間比較にバイアスが入る
    - ベースライン比較表を作る前に対応する
  - PromptSMILES の dedup 修正（本計画）を実施済みとして記録する。
- **Dependencies**: after Step 2

### Step 4: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**: GPU 不要の範囲で確認する。
  - `make_datasets.py` の構文チェック
  - 同一手順を断面で再現し、promptsmiles train が **brics・rc_cms とも 1分子1行**になること
  - **SAFE と FragGPT の出力が変わっていないこと**（行数・列構成）
  - `promptsmiles_test` の行数が変わっていないこと（RFFMG・SAFE・FragGPT と揃うこと）
- **Dependencies**: after Step 3

## 影響範囲

- **brics**: 変化なし（既に 1分子1行）
- **rc_cms**: PromptSMILES の train が 41,764行 → 11,913行 相当に減る（断面での実測値）
- SAFE・FragGPT・RFFMG の出力は変わらない
- `data/promptsmiles/` はまだ生成されていないため、既存データの作り直しは発生しない

## スコープ外

- SAFE・FragGPT の dedup 対象（現行が正しい）
- 立体比較の問題そのものの修正（`next_plan.md` への記録のみ。実装は今後）
- `evaluation.py` 等の `sampling_num` 階層の欠落（既存の保留項目）
