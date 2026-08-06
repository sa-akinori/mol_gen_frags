# Plan: FragGPT を HF DatasetDict 形式に追随させる

- **Date**: 2026-08-02
- **Status**: pending-approval

## Overview

`src/make_datasets.py` の未コミット変更により、PromptSMILES と FragGPT のデータ保存形式が
プレーンテキスト（`.smi` / `.target`）から **HF `DatasetDict`** に変更された。
PromptSMILES 側（`train_promptsmiles.py` / `gen_promptsmiles.py`）は追随済みだが、
**FragGPT 側と共有の `loadTrainSmiles` が追随しておらず、現状ではパイプラインが通らない。**

### 現在の保存形式（`make_datasets.py:138-162`、`if args.sampling_num == 5:` 内）

| データセット | パス | 列 |
|---|---|---|
| promptsmiles | `data/promptsmiles/{frag}/normal` | train/validation/test すべて `smiles`, `pass_fragments` |
| fraggpt | `data/fraggpt/{frag}/normal` | train/validation: `smiles`, `full_fragments`<br>test: `smiles`, `pass_fragments` |

### 現在壊れている箇所（実行すれば例外になる）

| 箇所 | 現在の実装 | 起きること |
|---|---|---|
| `train_fraggpt.py:201-202` | `read_lines(data_dir / "train.smi")` | `FileNotFoundError`。`.smi` は生成されない |
| `evaluation.py:64` | `tr_file_name = .../normal/train.target` | `FileNotFoundError`。`train.target` は生成されない |
| `evaluation_func.py:32-33` | `arc_name=='t5chem'` → `pd.read_csv(file_name)` | **PromptSMILES でも** `IsADirectoryError`。`evaluation.py:57` がディレクトリを渡すため |
| `generation_fraggpt_func.py:88` | `data/safe/{frag}/normal` の test split | 動作はするが、fraggpt データセットに同じ列の test split があるのに外部を参照している |

### 方針（ユーザー決定）

- **FragGPT を HF 形式に合わせる**（`make_datasets.py` は現状維持）
- **PromptSMILES の `loadTrainSmiles` 問題も同時に直す**（共通の原因のため）

### 設計判断: `loadTrainSmiles` の分岐をパス種別で行う

現在は `arc_name` で分岐しているが、`arc_name` は「predictions.csv の形式」を表す軸であり、
「学習分子をどこから読むか」とは独立している。実際 PromptSMILES と FragGPT は
`arc_name='t5chem'`（T5Chem 形式の predictions.csv）でありながら、学習分子は HF データセットにある。

そこで **`os.path.isdir()` による分岐**に変更する。現行の4手法すべてが正しく振り分けられる。

| 手法 | `tr_file_name` | 種別 | 読み方 |
|---|---|---|---|
| safe_gpt | `data/safe/{frag}/normal` | ディレクトリ | `load_from_disk(...)['train']['smiles']`（現行と同一） |
| promptsmiles | `data/promptsmiles/{frag}/normal` | ディレクトリ | 同上（**新規に通るようになる**） |
| fraggpt | `data/fraggpt/{frag}/normal` | ディレクトリ | 同上（**新規に通るようになる**） |
| t5chem / gpt | `data/rffmg/{frag}/normal/train.target` | ファイル | `pd.read_csv(...)`（現行と同一） |

safe_gpt / t5chem の挙動は変わらないため、SAFE・RFFMG の評価経路には影響しない。

## Plan

### Step 1: `loadTrainSmiles` をパス種別で分岐させる

- **Target file**: `src/func/evaluation_func.py`
- **Changes**:
  - `loadTrainSmiles(arc_name, file_name)` の分岐を `arc_name` から
    **`os.path.isdir(file_name)`** に変更する。
    - ディレクトリ → `datasets.load_from_disk(file_name)['train']['smiles']`
    - ファイル → `pd.read_csv(file_name, header=None).squeeze().tolist()`
  - 引数 `arc_name` は他の呼び出し規約（`loadGenSmiles` と対で使われる）との一貫性のため
    **シグネチャは変更しない**。使わなくなる場合はその旨を docstring に明記する。
  - docstring を Google style で追加し、どの手法がどちらの経路を通るかを表で示す。
  - 既存の `safe_gpt` / `t5chem` の挙動が変わらないことをコメントで明示する。
- **Dependencies**: none

### Step 2: `evaluation.py` の fraggpt 分岐をデータセットディレクトリに変更

- **Target file**: `src/evaluation.py`
- **Changes**:
  - fraggpt 分岐の `tr_file_name` を
    `f'{BASEPATH}/data/fraggpt/{frag_method}/normal/train.target'` から
    `f'{BASEPATH}/data/fraggpt/{frag_method}/normal'` に変更する（promptsmiles 分岐と同形）。
  - `testInputfile` は変更しない（`generation_fraggpt_func.py` がデータセットディレクトリ内に
    `test.source` を書き出す。promptsmiles と同じ作法）。
  - コメントを実態に合わせて更新する。
- **Dependencies**: after Step 1

### Step 3: `train_fraggpt.py` を `load_from_disk` に移行

- **Target file**: `src/train_model/train_fraggpt.py`
- **Changes**:
  - `train_promptsmiles.py` の `read_corpus()` に倣い、
    `read_corpus(frag_method: str) -> tuple[list[str], list[str]]` を追加する。
    `datasets.load_from_disk(f"{BASEPATH}/data/fraggpt/{frag_method}/normal")` を読み、
    `dataset["train"]["full_fragments"], dataset["validation"]["full_fragments"]` を返す。
  - `read_lines()` が未使用になるため削除する。`pathlib.Path` の import も未使用になれば削除する。
  - `import datasets` を追加する（import 順は 標準 → サードパーティ → ローカル を維持）。
  - docstring を実態に合わせる（`.smi` ではなく HF データセットの `full_fragments` 列を読む旨、
    および `make_datasets.py` が `full_fragments` で重複排除済みである旨）。
  - **学習ロジック（無条件LM・拡張・ハイパラ）は一切変更しない。**
- **Dependencies**: after Step 1

### Step 4: `generation_fraggpt_func.py` の test split の読み元を変更

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - `read_test_split()` の読み元を
    `data/safe/{frag_method}/normal` から **`data/fraggpt/{frag_method}/normal`** に変更する
    （`gen_promptsmiles.py:125` と同形）。列は `smiles` / `pass_fragments` で変更なし。
  - モジュール docstring と `read_test_split` の docstring を更新し、
    fraggpt データセットの test split が `make_datasets.py` で `safe_te` から作られており、
    SAFE・PromptSMILES・RFFMG と同一の行であることを明記する。
  - **`make_datasets.py` を再実行すると `save_to_disk` がディレクトリを作り直すため、
    生成済みの `test.source` / `test.target` が消える**点をコメントで注意喚起する
    （実行順序は データ生成 → 学習 → 生成 → 評価）。
  - **生成ロジック（付番・beam search・組立・行アラインメント）は一切変更しない。**
- **Dependencies**: after Step 1

### Step 5: `next_plan.md` の更新

- **Target file**: `next_plan.md`
- **Changes**: 2026-07-30 時点の記述が旧テキスト形式を前提にしているため、現状に合わせる。
  - 検証表の `data/promptsmiles/{frag}/normal/train.smi` / `data/fraggpt/{frag}/normal/train.target`
    の行を、HF データセットの split 行数に置き換える
    （promptsmiles train/validation/test、fraggpt train/validation/test）。
  - 「`data/fraggpt/{frag}/normal/test.*` は意図的に生成されない」という記述を削除する
    （現在は fraggpt データセット自体に test split が含まれる）。
  - 保存形式が HF `DatasetDict` に統一されたことを追記する。
- **Dependencies**: after Step 4

### Step 6: 小規模検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**: サンドボックスで以下を確認する。
  - 全変更ファイルの構文チェック
  - 小さい HF DatasetDict を作り、`train_fraggpt.py` が読めること（数ステップ学習）
  - `generation_fraggpt_func.py` が fraggpt データセットの test split から生成できること
  - `loadTrainSmiles` が 4手法すべてのパス形（ディレクトリ / ファイル）で正しく動くこと
  - `evaluation.py --model_name fraggpt` が最後まで通ること
  - **SAFE・RFFMG の評価経路が壊れていないこと**（`loadTrainSmiles` は共有関数のため）
- **Dependencies**: after Step 5

## スコープ外（今回やらないこと）

- `make_datasets.py` の変更（現状の HF 形式を正とする）
- `evaluation.py` 等の `sampling_num` 階層の欠落（`next_plan.md` に記載の保留項目）
- `data/dummy/` を参照する残置コード（`figure.py` / `check_reproducibility.py`）
- FragGPT の仕様変更（付番方式・デコード方式・学習条件はすべて確定済みのまま）

## 注意

`data/fraggpt/` と `data/promptsmiles/` は**まだ生成されていない**。
本計画はコードを実データの形式に合わせる作業であり、データ生成はユーザーが実施する。
