# Plan: FragGPT の組み立て失敗を握り潰さないようにする

- **Date**: 2026-08-06
- **Status**: approved

## Overview

`assemble_fragments_with_reason` と `label_attachment_points` には「失敗が失敗として集計されない」
箇所が3つある。いずれも生成結果の集計を実態より良く見せてしまう。

### 1. 結合次数の食い違いが断片の並び順で結果を変える（実測）

`assemble_fragments_with_reason` の77行目

```python
bond_type = next((bt for bt in bond_types if bt != Chem.BondType.SINGLE), Chem.BondType.SINGLE)
```

`next()` はタプルの先に来たほうを採るため、同じ矛盾入力でも断片の書かれた順で結果が変わる。

| 入力 | 採用 | 結果 |
|---|---|---|
| `[1*]=C(C)C.[1*]#CC` | DOUBLE | `CC=C(C)C` → **ok**（三重結合が二重結合に化ける） |
| `[1*]#CC.[1*]=C(C)C` | TRIPLE | → **sanitize_failure** |
| `[1*]=C(C)C.[1*]C1CCCCC1` | DOUBLE | `CC(C)=C1CCCCC1` → **ok**（単結合側を無視） |

切断された結合の両端は本来同じ次数なので、食い違い自体がモデル出力の矛盾を示す。

### 2. パースできない断片が黙って捨てられる（実測）

49行目の walrus フィルタにより、壊れた断片は捨てられ残りだけで組み上がる。

```
assemble_fragments_with_reason("[1*]CC.[1*]O.C1CC")  ->  ('CCO', 'ok')
```

`parse_failure` になるのは全断片が壊れたときだけで、docstring の
「RDKit could not parse one of the fragments」と食い違う。

### 3. `label_attachment_points` の `Raises: ValueError` が嘘（実測）

同じ書き方で断片が黙って消える。モジュール内に `raise ValueError` は1つも無い。

```
label_attachment_points(["*c1ccc(*)cc1", "C1CC", "*O"], rng)  ->  2断片（1つ消失、例外なし）
```

### 生成側の死んだコード

テスト分割の `pass_fragments` は全行パース可能であることを実測で確認した。

| frag | 行数 | 断片数 | パース失敗 | 空断片 |
|---|---:|---:|---:|---:|
| brics | 82,441 | 300,336 | 0 | 0 |
| rc_cms | 90,974 | 240,808 | 0 | 0 |

`label_attachment_points` が断片を落とした行も 20,000行×2手法で 0 件。
よって `generation_fraggpt_func.py` のプロンプト失敗まわりは到達不能であり、
`try/except` を足すのではなく削除する。Step 4 で `ValueError` を送出するようにすれば、
万一データが壊れていた場合はその場で落ちる（黙って断片を捨てるより正しい）。

### 空断片の扱い（Step 3 の注意点）

プロンプトは必ず `.` で終わる（`generation_fraggpt_func.py:90` の `'.'.join(fragments) + '.'`）。
そのため `batch_prompt + completion` は、completion が空・`.` で終わる・`..` を含む場合に
空断片を生む。現状は `Chem.MolFromSmiles('')` が None ではなく原子0個の Mol を返すため無害。

**Step 3 で空断片を除外せずに厳格化すると、空 completion が `parse_failure` に分類される。**
現在それは `unmatched_dummy`（相手の断片が来なかった）と報告されており、そちらが正しい。
よって空文字のフィルタは必ず残す。

### スコープ外

- `predictions[start + position]` の修正（**ユーザーが修正済み**）
- 評価の立体比較（別タスク）

## Plan

### Step 2: 結合次数の食い違いを失敗として扱う

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**:
  - `assemble_fragments_with_reason` の75-77行目（`bond_types` の組み立てと `next()` による選択）を
    次の形に差し替える。
    ```python
    if first_bond.GetBondType() != second_bond.GetBondType():
        return None, "bond_order_mismatch"
    bond_type = first_bond.GetBondType()
    ```
  - 78行目の `rwmol.AddBond(first_neighbor, second_neighbor, bond_type)` は変更しない。
  - docstring の `Returns` の失敗理由一覧に `bond_order_mismatch`（切断された結合の両端で
    結合次数が一致しない）を追加する。既存の理由の説明文と同じ体裁に揃えること。
  - docstring 冒頭の「of the two dummy bonds the one that is not single wins」という説明は
    挙動と合わなくなるので、「両端の結合次数は一致していなければならない」旨に書き換える。
  - `format_failure_summary`（`generation_fraggpt_func.py`）は理由文字列を汎用に列挙しているため
    **変更不要**。
- **Dependencies**: none

### Step 3: パースできない断片を失敗として扱う

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**:
  - `assemble_fragments_with_reason` の49行目
    ```python
    mols = [mol for fragment in fragments.split('.') if (mol := Chem.MolFromSmiles(fragment)) is not None]
    ```
    を、**空文字の断片を除外したうえで**、残りが1つでもパースできなければ `parse_failure` を
    返す形に変える。空文字のフィルタ（`if fragment`）は必ず残すこと
    （Overview「空断片の扱い」参照）。
  - 既存の `try/except ValueError`（50-51行目）と `if not mols`（52-53行目）の扱いは、
    新しい実装で `parse_failure` が返る経路が二重にならないよう整理してよい。
    ただし**断片が1つも無い入力は従来どおり `parse_failure`** を返すこと。
  - 46-47行目のコメント（断片を1つずつパースする理由）は残す。
- **Dependencies**: none

### Step 4: `label_attachment_points` を docstring どおり `ValueError` にする

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**:
  - 116行目の walrus フィルタをやめ、`Chem.MolFromSmiles` が None を返した断片があれば
    `ValueError` を送出する。例外メッセージにはパースできなかった断片の SMILES を含める。
  - docstring の `Raises: ValueError` はそのまま（これで記述と挙動が一致する）。
  - 117-122行目（ダミー原子の収集・ラベル付け・`MolToSmiles`）は変更しない。
- **Dependencies**: none

### Step 5: 生成側の到達不能なコードを削除する

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - `main()` から次を削除する。
    - `prompt_failures: Counter[str] = Counter()`（81行目）
    - `prompt_fragments: list[list[str]] = []`（82行目）と
      `prompt_fragments.append(fragments)`（91行目）
    - `prompts` の型注釈 `list[str | None]` を `list[str]` にする（83行目）
    - 93行目の `sum(prompt is not None for prompt in prompts)` を `len(prompts)` にする
  - `format_failure_summary` から次を削除する。
    - 引数 `prompt_failures: Counter[str]`（16行目）と docstring の該当行（24行目）
    - `f"rows without a prompt: {sum(prompt_failures.values())}"`（35行目）
    - `lines += [f"  prompt failure ({reason}): {count}" ...]`（39行目）
  - 150行目の呼び出しから `prompt_failures` 引数を外す。
  - `from collections import Counter` は `assembly_reasons` で使い続けるので**残す**。
  - 生成ループ（113-138行目）は一切変更しない（`predictions[start + position]` は修正済み）。
- **Dependencies**: after Step 4

### Step 6: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 全変更ファイルの構文チェックと import 確認
  - `bond_order_mismatch` が断片の並び順に依存しないこと
    （`[1*]=C(C)C.[1*]#CC` と `[1*]#CC.[1*]=C(C)C` の両方が同じ理由を返す）
  - 空 completion（`"[1*]CC." + ""`）が `parse_failure` ではなく
    `unmatched_dummy` のままであること。末尾 `.` / `..` を含む入力も従来どおり組み上がること
  - パース不能な断片を含む入力が `parse_failure` を返すこと
  - `label_attachment_points` がパース不能な断片で `ValueError` を送出すること
  - **実データ1万件で、変更前後の組み立て結果が一致すること**（正常系に影響が無いこと）
  - テスト全行（brics 82,441 / rc_cms 90,974）で `label_attachment_points` が
    例外を出さないこと
- **Dependencies**: after Step 5
