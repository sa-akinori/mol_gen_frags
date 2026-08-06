# Plan: save_file / load_file を func/utility.py に集約する

- **Date**: 2026-08-04
- **Status**: pending-approval

## Overview

同一の `save_file` が **3箇所に重複定義**されており、対になる `load_file` は
`make_datasets.py` にしか無いため、`build_augmented_dataset.py` が `sys.path` を書き換えて
実行スクリプトから import するという不自然な構造になっている。
両方を `src/func/utility.py` に移して1箇所にする。

### 重複の実測（AST で本体を比較）

3つとも**本体は1文字も違わない**。差は引数名だけ。

```python
with open(<path>, 'w', newline='\n', encoding='utf-8') as f:
    f.write(target)
```

| 定義 | 引数 | 型ヒント / docstring |
|---|---|---|
| `src/func/generation_fraggpt_func.py:15` | `(target, save_path)` | あり |
| `src/gen_mols/gen_promptsmiles.py:99` | `(target, save_path)` | あり（上と完全一致） |
| `src/make_datasets.py:15` | `(target, save_file)` | なし。**引数名が関数名を隠している** |

`load_file` は `src/make_datasets.py:22` のみ（型ヒントは引数のみ、戻り値なし）。

### 現在の呼び出し構造

```
make_datasets.py            save_file / load_file を定義して自分で使う
gen_promptsmiles.py         save_file を自前で定義
generation_fraggpt_func.py  save_file を自前で定義
build_augmented_dataset.py  sys.path.insert 後に from make_datasets import load_file, save_file
                            ← 実行スクリプトからの import。utility.py に移せば不要になる
```

### 移設先の確認（実測）

- `src/func/utility.py` に `save_file` / `load_file` と**同名の定義は無い**（衝突しない）
- `utility.py` に `__all__` は無いため、`from func.utility import *` で自動的に公開される
- `import *` しているのは `evaluation.py` / `analyze_predictions.py` / `gen_frags/rffmg_frags.py` の3ファイル。
  いずれも `save_file` / `load_file` を自前定義していないため、名前が隠される心配は無い

## Plan

### Step 1: `func/utility.py` に2関数を追加する

- **Target file**: `src/func/utility.py`
- **Changes**: 以下2つを追加する。既存の `pickle_save` / `pickle_load` の近くに置き、
  周囲のコードスタイルに合わせること。
  - `save_file(target: str, save_path: str) -> None`
    本体は既存3実装と同一（`newline="\n"`, `encoding="utf-8"`）。
    引数名は型ヒント付きの版（`save_path`）を採用する。
    `make_datasets.py` 版の引数名 `save_file` は関数名を隠すため採用しない。
  - `load_file(file_name: str) -> list[str]`
    `make_datasets.py:22` の実装（各行を `rstrip()` してリストで返す）をそのまま移す。
    戻り値の型ヒントを補う。
  - 両方に Google style docstring を付ける。
  - **`newline="\n"` を明示している点は非自明な制約**（環境に依らず LF で書き出すため）なので、
    docstring で触れること。
- **Dependencies**: none

### Step 2: `make_datasets.py` の定義を削除して import に置き換える

- **Target file**: `src/make_datasets.py`
- **Changes**:
  - 15-28行付近の `save_file` / `load_file` の定義を削除する。
  - `from func.utility import ...` に `save_file`, `load_file` を追加する
    （既存の import 文の形に合わせること）。
  - 呼び出し箇所（90, 91, 99, 100, 187, 188行 と 173, 174行）は**引数の渡し方が
    位置引数なので変更不要**。コメントアウト済みの呼び出しも触らない。
- **Dependencies**: after Step 1

### Step 3: `gen_promptsmiles.py` の定義を削除して import に置き換える

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **Changes**: 99行付近の `save_file` 定義を削除し、既存の `from func.utility import ...` に
  `save_file` を追加する。呼び出し（484, 485行）は変更不要。
- **Dependencies**: after Step 1

### Step 4: `generation_fraggpt_func.py` の定義を削除して import に置き換える

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**: 15行付近の `save_file` 定義を削除し、既存の
  `from func.utility import BASEPATH, LogFile` に `save_file` を追加する。
  呼び出し（175, 176行）は変更不要。
- **Dependencies**: after Step 1

### Step 5: `build_augmented_dataset.py` の import 元を変更する

- **Target file**: `src/recursion_train/build_augmented_dataset.py`
- **Changes**:
  - `from make_datasets import load_file, save_file`（24行）を削除し、
    `from func.utility import BASEPATH`（23行）に `load_file`, `save_file` を加える。
  - **`sys.path.insert(0, ...)`（22行）が他の用途に使われていないか確認すること。**
    `make_datasets` の import のためだけなら、`import sys` と `from pathlib import Path` も含めて
    削除できる。他で使っていれば残す。
  - 呼び出し箇所（79, 80, 90, 91, 96, 119行）は変更不要。
- **Dependencies**: after Step 1

### Step 6: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 全変更ファイルの構文チェックと import 確認
  - `save_file` / `load_file` の定義が `utility.py` の1箇所だけになっていることを grep で確認
  - `save_file` → `load_file` の往復で内容が保たれること（LF 改行、末尾行の扱い）
  - `from func.utility import *` を使う3ファイル（`evaluation.py` / `analyze_predictions.py` /
    `gen_frags/rffmg_frags.py`）が import できること
  - `build_augmented_dataset.py` が import できること
- **Dependencies**: after Step 5

## スコープ外

- `MakeFolder` / `MakeFolders` / `MakeFolderWithCurrentFuncName` の削除
  （いずれも `utility.py` の外から呼ばれていない未使用コードだが、今回は触らない）
- `utility.py` の他の関数の整理
- `from func.utility import *` を明示 import に改める作業

## 注意

`utility.py` は RFFMG・SAFE を含む全手法から読まれる共有モジュールである。
2関数の**追加**のみで既存の名前は一切変更しないため、他手法への影響は無い想定だが、
Step 6 で 3つの `import *` 経路が通ることを必ず確認する。
