# Plan: PromptSMILES を含む4手法の比較を成立させる（案C）

- **Date**: 2026-08-09
- **Status**: pending-approval

## Overview

PromptSMILES は、attachment point を2個以上持つ断片を含む集合を条件にできない。
`ScaffoldDecorator` は「何を置くか」を指定できず、`FragmentLinker` は断片を `(frag)` の
枝として文字列挿入するため結合を1本しか作れない（`samplers.py:558-560` の assert は
この構造から来る制約）。`scaffold=[...]` にリストを渡しても、`samplers.py:333-334` で
1サンプルにつき1つが乱択されるだけで、複数断片の同時条件付けにはならない。

その結果、test 全行の **90.5%（brics）/ 84.3%（rc_cms）** で、PromptSMILES は
断片集合の一部（最大断片1個）しか条件に取れない。他の3手法は集合すべてを受け取るため、
このままでは同じ課題を解いていない。

対応は案C（AとBの併記）を採る。

- **A（主表）**: 4手法を同じ課題で採点する。`fragment` 列に元の断片集合すべてを書き、
  PromptSMILES が条件を落とすぶん低いスコアになるのを結果として出す。
- **B（副表）**: PromptSMILES が断片集合を100%条件にできる行だけで4手法を比べる。
  この部分集合では全手法が同一の条件を受け取るので、厳密に等価な比較になる。

Bの仕組みは既存の `analyze_predictions.py` の `spec_cond_frags` / `spec_stats.csv` /
`no_spec_stats.csv` がそのまま使える。新規には作らない。

## 前提の確認（調査済み）

- `results/` に PromptSMILES の `predictions.csv` は**まだ無い**。Step 1 による再生成コストはゼロ。
- 4手法の `fragment` 列は同一形式（`*C(=O)C=*.*N1CCN(*)CC1...`）。RFFMG は `test.source`
  由来、他3手法は生成側が書く。よってBの判定は `fragment` 列だけで手法ごとに独立に計算でき、
  手法間の行 join は不要。
- `spec_cond_frags` の現行条件は「断片1個」または「2断片かつ各1点」。PromptSMILES の実力は
  「断片1個」または「**全断片**が各1点」。差は brics 528行 (0.64%) / rc_cms 457行 (0.50%)。

| | 断片1個 | 2断片・各1点 | 3断片以上・各1点 | 表現不可 | 現行 spec | 修正後 spec |
|---|---|---|---|---|---|---|
| brics (82,441) | 2,690 | 4,592 | 528 | 74,631 | 7,282 (8.83%) | 7,810 (9.47%) |
| rc_cms (90,974) | 9,473 | 4,354 | 457 | 76,690 | 13,827 (15.20%) | 14,284 (15.70%) |

## スコープ外

- promptsmiles ライブラリ本体の変更（`FragmentLinker` のバグ含む）。上流のまま比較する方針は
  `.plans_for_ai/2026-08-09/patch_promptsmiles_fragmentlinker.md` で確定済み。
- `evaluation_func.py:443` の `validfragratio_exH` が非 exH 版を呼んでいる件（別件）。
- 生成の実行、図の作成。

## Plan

### Step 1: `fragment` 列に元の断片集合すべてを書く

- **Target file**: `src/func/generation_promptsmiles_func.py`
- **Changes**:
  - `main()` のループで集める値を2系統に分ける。
    - `fragment` 列 = **元の断片集合** `fragment_set`（プロンプトに使えなかった断片も含む）
    - `prompt_fragments` 列 = **実際にプロンプトした断片** `".".join(fragments)`
    （SAFE が `fragment` の隣に `prompt_safe` を持つのと同じ体裁）
  - 変数 `prompted_fragments` は意味が変わるのでリネームし、`prompt_fragments` を新設する。
  - `predictions_df` の列順は `fragment`, `target`, `sampler`, `prompt_fragments`,
    `prediction_1..N` とする。`evaluation.py:80` は `['fragment', 'target'] + pred_cols` を
    明示選択するので、列が増えても評価は壊れない。
  - 行数一致を確かめる `assert`（現 360 行）に `prompt_fragments` を加える。
  - 現 359 行のコメント「The evaluation reads the prompt from the fragment column」は
    事実でなくなるので、**評価は要求された断片集合を読み、実際のプロンプトは
    `prompt_fragments` に残る**という趣旨に直す。
  - `select_prompt_fragments` の docstring のうち、返り値が評価に直結する旨に触れている箇所を
    実態に合わせる。**それ以外の説明は増やさない。**
  - デッドコードを削除する: `parse_fragments`（32-42行）は docstring だけで本体が空で、
    暗黙に `None` を返す。どこからも呼ばれておらず、ロジックは `select_prompt_fragments:68` に
    インライン展開済み。
- **Dependencies**: none

### Step 2: `spec_cond_frags` を PromptSMILES の表現可能条件に一致させる

- **Target file**: `src/analyze_predictions.py`
- **Changes**:
  - 「3断片以上を一律 False にする」分岐を外し、条件を
    **「断片が1個」または「全断片の attachment point がちょうど1個」** にする。
    `FragmentLinker` は3断片以上でも `scan=True` を自動で有効にして扱えるため、
    現行条件は PromptSMILES の実力より狭い。
  - 型ヒント（`frags: str -> bool`）と Google style docstring を付ける。
    docstring には**この条件が PromptSMILES の表現可能範囲と一致すること**を1〜2行で書く。
  - 分岐を素直に1つの述語として書き直す（`all(...)` を使い、`True if x else False` の
    冗長な書き方をやめる）。
  - 43行目付近の `if 1:` ブロックのコメント「Condition: Input is one fragment with multiple
    attachment points or two fragments with one attachment point each」を新条件に合わせて直す。
- **Dependencies**: none

### Step 3: spec / no_spec の統計計算を1つの関数にまとめる

- **Target file**: `src/analyze_predictions.py`
- **Changes**:
  - `if 1:` ブロック内で `spec_stats` と `no_spec_stats` を組み立てている2つの dict は、
    参照する DataFrame が違うだけで**中身が完全に同一の22行**。
    `summarize_generation_stats(df: pd.DataFrame) -> pd.Series` を新設して両方から呼ぶ。
  - 関数は `avg/std` の `validity`, `validity_onfrags`, `uniqueness`, `novelty`,
    `tanimoto_sim`, `rediscovery` を返す。集計対象の絞り込み（`nnovel != 0`）も含め、
    現行の計算内容を**一切変えない**。
  - Google style docstring に返り値のインデックス名を明記する。
  - 呼び出し側は `summarize_generation_stats(spec_cond_df).to_csv(...)` の2行になる。
- **Dependencies**: after Step 2

### Step 4: 2つの表の位置づけを README に書く

- **Target file**: `README.md`, `README_ja.md`
- **Changes**:
  - 「Evaluation of Generated Molecules」節の末尾に短い項を足す。
  - 書く内容は3点だけ。**それ以上は書かない。**
    1. `evaluation.py` が出す `stats.csv` が主表（全 test 行、4手法とも同じ断片集合を要求）。
    2. `analyze_predictions.py` が出す `spec_stats.csv` が副表で、PromptSMILES が断片集合を
       100%条件にできる行だけを集めたもの。`no_spec_stats.csv` はその補集合。
    3. PromptSMILES は attachment point を2個以上持つ断片を条件にできないため、主表では
       断片集合の一部しか使えない行が brics 90.5% / rc_cms 84.3% を占める。
       `predictions.csv` の `prompt_fragments` 列に実際に使われた断片が残る。
  - PromptSMILES の生成コマンド（`bash src/gen_mols/gen_promptsmiles.sh`）の説明に、
    `sampler` 列が `scaffold` / `linking` のどちらを通ったかを示す旨を1行足す。
- **Dependencies**: after Step 1, Step 2

## 検証（実装後にメインエージェントが実施）

- `python -m py_compile` が4ファイルとも通ること。
- `spec_cond_frags` を修正後の実装で test 全行に適用し、True の行数が
  brics 7,810 / rc_cms 14,284 になること（上表と一致すること）。
- `summarize_generation_stats` の出力が、既存の `spec_stats.csv` /
  `no_spec_stats.csv`（rffmg・safe の既存結果）と**数値・インデックスとも完全一致**すること。
  リファクタで値が変わっていないことをこれで担保する。
- PromptSMILES を `--additional_path` を分けた一時ディレクトリで少数行だけ生成し、
  `predictions.csv` の列が `fragment`, `target`, `sampler`, `prompt_fragments`,
  `prediction_*` の順で並び、`fragment` にドット区切りの元の集合、`prompt_fragments` に
  実際のプロンプトが入っていること。確認後、一時ディレクトリは削除する。
