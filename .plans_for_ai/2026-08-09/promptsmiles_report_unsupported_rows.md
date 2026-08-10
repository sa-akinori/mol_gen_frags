# Plan: PromptSMILES が対応できない行を代替生成せず、非対応として記録する

- **Date**: 2026-08-09
- **Status**: pending-approval (rev.2 — 副表を廃止する方針に改訂)

## Overview

同日の `promptsmiles_fair_comparison_subset` で `fragment` 列に元の断片集合を書くようにしたが、
**生成側が依然として対応不可の行を無理やり生成している**ため、主表の数値が他手法と同じ意味を
持たない。本計画でそこを直す。

### 問題1: 対応できない課題をすり替えて生成している

`select_prompt_fragments` は、PromptSMILES が表現できない断片集合（attachment point を2個以上
持つ断片を含む集合）に対し、**最大断片1個だけの scaffold decoration** という代替課題を与えて
生成させている。該当は test 全行の **90.5%（brics）/ 84.3%（rc_cms）**。

この代替は PromptSMILES の仕様ではなく我々の設計判断であり、主表に載る数値は
「手法が課題を解けたか」と「我々の代替規則がどれだけ当たったか」の混合物になっている。
選択規則をサイズ最大から結合点最多に変えるだけで条件に渡る原子量が 48.5% → 36.8%（brics）と
変わることからも、この数値が比較の基準になっていないことが分かる。

実測した代替の内訳（フォールバック行のうち prompt された断片の attachment point 数）:

| | 1個 | 2個 | 3個以上 |
|---|---|---|---|
| brics (74,631行) | 33,052 (44.3%) | 32,088 (43.0%) | 9,491 (12.7%) |
| rc_cms (76,690行) | 19,797 (25.8%) | 35,803 (46.7%) | 21,090 (27.5%) |

attachment point が1個の断片を scaffold として渡した場合、`ScaffoldDecorator` は置換基を
1本伸ばして終わる。scaffold decoration ではなく「種を1個与えた de novo 生成」であり、
brics ではフォールバック行の44%がこれに該当する。

### 問題2: 失敗行を predictions.csv から落としている

他の2手法は失敗行を落とさない。

| 手法 | 失敗時の扱い | 行数 |
|---|---|---|
| FragGPT | `predictions` を `INVALID_SMILES` で初期化し成功分だけ上書き（`generation_fraggpt_func.py:109`） | test と常に同じ |
| SAFE | デコード失敗候補のみ `INVALID_SMILES`（`generation_safe_func.py:161`） | test と常に同じ |
| **PromptSMILES** | **`continue` で行ごと破棄（300, 305, 341行）** | **test より減る** |

PromptSMILES だけが行を落とすため、主表の平均が「生成できた行だけの平均」になり、
他手法と分母が揃わない。

### 方針

**PromptSMILES が表現できない集合は生成しない。** その行は他手法の失敗行と同じく
`INVALID_SMILES` で埋めて `predictions.csv` に残す。

これで `stats.csv` の時点で4手法が「同一の test 行」「同一の断片集合」「同一の失敗の数え方」で
採点されるため、**主表だけで厳密な比較が成立する**。我々が発明した代替規則が数値に混入せず、
「サイズ最大か結合点最多か」という決められない選択そのものが消える。

### 副表を廃止する

`analyze_predictions.py` の `spec_cond_frags` / `spec_stats.csv` / `no_spec_stats.csv` は
初回コミット `d139920`（2025-12-16）からある分析で、scaffold decoration と linker design しか
できない生成器を想定したもの。SAFE は `pass_safe` に断片集合全体を入れて条件付けする形になり、
この制約が当てはまらなくなったため**当初の動機は消えている**。

上記の方針により主表だけで厳密比較が取れるので、この分析は不要と判断し削除する。
`results/` に `spec_stats.csv` は1つも存在せず、実際に回されたこともない。

### 生成に回されなかった行の記録

削除するのは部分集合の**統計**であって、記録は残す。非生成行は次の3箇所に残る。

| 粒度 | 場所 | 内容 |
|---|---|---|
| 行単位 | `predictions.csv` の `sampler` 列 | `unsupported` / `invalid_target` / `generation_error` |
| 行単位 | `predictions.csv` の `fragment` 列 | 生成に回されなかった断片集合そのもの |
| 集計 | `generation_params.txt` | 理由別の件数と test 全行に対する割合 |

### 副次的な効果

生成対象が brics 82,441行 → 7,810行、rc_cms 90,974行 → 14,284行 に減るため、
**生成時間が約1/10になる**。

## スコープ外

- promptsmiles ライブラリ本体の変更。
- 他手法（RFFMG / SAFE / FragGPT）の生成・評価コード。
- `evaluation_func.py:443` の `validfragratio_exH` が非 exH 版を呼んでいる件（別件）。
- 図の作成、生成の実行。

## Plan

### Step 1: 対応できない行を生成せず、非対応として記録する

- **Target file**: `src/func/generation_promptsmiles_func.py`
- **Changes**:
  - `is_promptsmiles_expressible(fragment_set: str) -> bool` をこのモジュールに定義する。
    条件は **「断片が1個」または「全断片の attachment point がちょうど1個」**。
    - 断片1個 → `ScaffoldDecorator`（attachment point は何個でもよい）
    - 全断片が1点 → `FragmentLinker`（3断片以上でも `scan=True` が自動で有効になる）
    実装は文字列操作のみ（`split('.')` と `count('*')`）。Google style docstring に、
    この条件が PromptSMILES の表現可能範囲と一致する理由を1〜2行で書く。
    **それ以上は書かない。**
  - `select_prompt_fragments` を書き換える。
    - 冒頭で `is_promptsmiles_expressible(fragment_set)` が False なら **`None` を返す**。
      これで戻り値型 `tuple[str, list[str]] | None` が実態と一致する（現在は None を返す経路が
      無く型注釈が嘘になっている）。
    - True の場合のみ、断片1個なら `("scaffold", [その断片])`、
      それ以外は `("linking", 全断片をサイズ降順)` を返す。
    - **`max(fragments, key=GetNHA)` による最大断片の選択を削除する。** これが代替課題を
      作っていた本体。`GetNHA` / `func.fragmentation` の import が他で使われていなければ外す。
    - docstring を新しい契約に書き直す。「表現できない集合は None を返し、呼び出し側が
      非対応行として記録する」ことを明記する。
  - `main()` の生成ループを FragGPT と同じ形にする（`generation_fraggpt_func.py:109` が手本）。
    - `predictions` を **全 test 行ぶん `[[INVALID_SMILES] * n_samples]` で初期化**する。
    - `sampler_names` も全行ぶん用意し、既定値を非対応を表す値にする。
    - `fragment` 列は全 test 行の元の断片集合、`target` 列は全 test 行の SMILES。
      `prompt_fragments` は生成した行のみ埋め、それ以外は空文字。
    - 生成できた行だけ上書きする。**`continue` による行の破棄を全廃する**（現 300, 305, 341行）。
  - `sampler` 列の値を5種類にする。既存の `scaffold` / `linking` に加えて:
    - `unsupported` — 断片集合が PromptSMILES で表現できない
    - `invalid_target` — 目的分子の SMILES が RDKit で読めない
    - `generation_error` — promptsmiles が例外を投げた
    後ろ2つは実データで0件だが、行を落とさないために残す。
  - **生成に回されなかった行を記録すること。** `skip_counts` は破棄数ではなく
    「`INVALID_SMILES` で埋めた行数」を数えるものになるので、変数名と `format_run_summary` の
    docstring・出力文言をその意味に直す。`format_run_summary` は理由別の件数に加えて
    **test 全行に対する割合**も出すこと（どれだけが生成に回らなかったかが一目で分かるように）。
    `generation_error` の例外メッセージは従来どおり `log_lines` に残す。
  - `MIN_LINK_FRAGMENTS` の定数とコメントは `is_promptsmiles_expressible` の条件と重複するので、
    どちらか一方に集約する。`FragmentLinker` の `IndexError` を避ける根拠として残す必要があるかを
    判断し、不要なら削除する。
  - 行数の `assert` を、全リストが `len(test_smiles)` と一致することを確かめる形に変える。
- **Dependencies**: none

### Step 2: 用途の消えた部分集合分析を削除する

- **Target file**: `src/analyze_predictions.py`
- **Changes**:
  - `spec_cond_frags` を削除する。
  - `summarize_generation_stats` を削除する（この関数はこのブロックからしか呼ばれていない）。
  - `spec_stats.csv` / `no_spec_stats.csv` を書いている `if 1:` ブロック全体を削除する。
  - 削除により未使用になる import があれば外す。**他のブロック（`if 0:`）が使っているものは
    残すこと。** 特に `np` は他ブロックでも使われているので消さないこと。
  - 他の `if 0:` ブロックには手を触れない。
- **Dependencies**: none

### Step 3: README を主表だけの説明に直す

- **Target file**: `README.md`, `README_ja.md`
- **Changes**:
  - 同日の計画で追記した「主表 / 副表」の段落を、次の内容に**差し替える**。
    書く内容は2点だけ。**それ以上は書かない。**
    1. `stats.csv` は全 test 行を対象とし、4手法とも同じ断片集合を要求される。
       PromptSMILES が表現できない断片集合の行は生成されず `INVALID_SMILES` として
       0点で集計されるので、**カバレッジを含んだ数値**になる。これは FragGPT の組み立て失敗・
       SAFE のデコード失敗と同じ扱いである。
    2. どの行が生成に回らなかったかは `predictions.csv` の `sampler` 列で分かる
       （`scaffold` / `linking` / `unsupported` / `invalid_target` / `generation_error`）。
       件数の集計は `generation_params.txt` にある。
  - PromptSMILES の生成コマンドの説明に、**表現できない断片集合では生成を行わない**旨を1行足す。
  - `spec_stats.csv` に言及している箇所が他に無いことを確認して、あれば消す。
- **Dependencies**: after Step 1, Step 2

## 検証（実装後にメインエージェントが実施）

- `python -m py_compile` が対象ファイルすべてで通ること。
- `is_promptsmiles_expressible` を test 全行に適用し、True の行数が
  brics **7,810 / 82,441** / rc_cms **14,284 / 90,974** になること。
- `analyze_predictions.py` から `spec_cond_frags` / `summarize_generation_stats` /
  `spec_stats` の参照が0件になっていること。残った `if 0:` ブロックが壊れていないこと
  （import 済みの名前がすべて解決すること）。
- 一時的な `--additional_path` で PromptSMILES を少数行だけ生成し、次を確認する。
  - `predictions.csv` の行数が **入力した test 行数と完全に一致**すること（1行も落ちない）。
  - `sampler` 列に `scaffold` / `linking` / `unsupported` が現れること。
  - `unsupported` の行は `prediction_*` がすべて `INVALID_SMILES`、`prompt_fragments` が空、
    `fragment` に元の断片集合が残っていること。
  - `scaffold` の行は `fragment` に `.` を含まない（断片1個）こと。
    **代替生成が消えたことをこれで確認する。**
  - `linking` の行は `prompt_fragments` が `fragment` と同じ断片集合であること。
  - `generation_params.txt` に理由別の件数と割合が出ていること。
- 確認後、一時ディレクトリを削除すること。
