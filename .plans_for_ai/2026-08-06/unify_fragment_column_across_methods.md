# Plan: unify_fragment_column_across_methods

- **Date**: 2026-08-06
- **Status**: completed (2026-08-06 実装・検証済み。判断 A=フィルタ削除 / B=残骸削除 ともユーザー承認のうえ実施)

## Overview

`evaluation.py` が `safe` と `promptsmiles` / `fraggpt` を別分岐にしている理由は設計上の必然ではなく、
T5Chem 用リーダ (`test.source` を行番号で横結合する) を使い回した歴史的経緯でしかない。
その結果、生成スクリプトが `data/` 配下に `test.source` / `test.target` を書き出すという歪んだ構造になっている。

### 調査で確定した事実

**1. `.source` / `.target` を本当に持つのは rffmg だけ**

| ディレクトリ | `.source` の有無 | 実態 |
|---|---|---|
| `data/rffmg/{frag}/{N}times_sampling/{add}/` | あり | `make_datasets.py` が作る正規のデータセット |
| `data/safe/{frag}/normal/` | **なし** | HF DatasetDict のみ |
| `data/promptsmiles/{frag}/normal/` | **なし** | HF DatasetDict のみ |
| `data/fraggpt/{frag}/normal/test.source` | あり (241 byte / **4 行**) | 2026-08-04 のスモークランで `generation_fraggpt_func.py` が書いた副産物。本来の test split は 82,441 行 |

生成スクリプトが `data/` に書き出しているため、`make_datasets.py` を回すと消える
(コード内のコメント自身が「rerunning the data generation deletes them」と認めている)。

**2. fraggpt の `test.source` は HF データセットと完全に重複**

`generation_fraggpt_func.py:157` が書いているのは `test_dataset["pass_fragments"]` そのままであり、
情報量はゼロ。行のスキップもないので、HF データセットを読めば同じものが得られる。

**3. promptsmiles だけは「実際にプロンプトしたフラグメント」の記録が必要**

`select_prompt_fragments` は `scaffold` ルートでは `pass_fragments` のうち**最大 1 フラグメントしか**
プロンプトしない (`linking` ルートは全フラグメントだが順序を大きい順に並べ替える)。
さらに生成例外時は行ごとスキップする。よって「条件付けに実際に使ったフラグメント」は行ごとに違う。
ただしこれは **safe と同じく predictions.csv の `fragment` 列**に記録すべきものであり、
行番号 join する別ファイルにする理由にはならない。むしろ行スキップがあるぶん join は危険。

**4. 失敗行の扱いが safe だけ非対称 (評価の公平性に関わる)**

| repr_name | 生成失敗時のセル | `loadGenSmiles` の扱い |
|---|---|---|
| safe | `"error"` (`generation_safe_func.py:158-159`) / `time_out` | 全予測が該当する行を**除外** |
| rffmg / promptsmiles / fraggpt | `""` | 行を**残す** (invalid としてカウント) |

safe だけ失敗行が統計から消えるため、validity / uniqueness / novelty が他手法より有利に出る。
現存の `results/safe/gpt/finetuning/brics/beam/predictions.csv` (1000 行の部分実行) では該当 0 行だが、
コード上の非対称は残っている。

### あるべき姿

**プロンプトしたフラグメントは、生成した本人が predictions.csv の `fragment` 列に書く。**
これは safe が既に行っている方式であり、行番号 join が不要で最も堅牢。
rffmg だけは本当にテキストデータセット (`.source` / `.target`) なので join を残す。

結果として `evaluation.py` の分岐は **rffmg / それ以外** の 2 つになり、
生成スクリプトが `data/` へ書き込むことも無くなる。

## Plan

### Step 1: FragGPT の predictions.csv に `fragment` 列を追加

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - `predictions_df` に `fragment` 列 (= `test_fragment_sets`, すなわち HF データセットの `pass_fragments`)
    を挿入する。列順は `fragment` / `target` / `prediction_1..N`。
  - `test.source` / `test.target` の `save_file` 呼び出し (現 157-158 行) を削除する。
    それに伴い `dataset_dir` の `os.makedirs` と `func.utility.save_file` の import が不要なら整理する。
  - 行番号 join を前提にしたコメント (現 152-156 行) を、`fragment` 列を自分で書く旨の説明に置き換える。
  - 行数の整合を確認する `assert` は残す (predictions と test_smiles の対応は依然として必要)。
- **Dependencies**: none

### Step 2: PromptSMILES の predictions.csv に `fragment` 列を追加

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **Changes**:
  - `predictions_df` に `fragment` 列 (= 既存の `sources`, すなわち実際にプロンプトしたフラグメント) を
    挿入する。列順は `fragment` / `target` / `sampler` / `prediction_1..N`
    (`sampler` は評価では捨てられるが、どちらのサンプラに振られたかの記録として残す)。
  - `test.source` / `test.target` の `save_file` 呼び出し (現 473-474 行) を削除する。
  - モジュール docstring の「出力」記述 (現 28 行付近の `data/promptsmiles/.../test.source` の説明) と
    行番号 join を前提にしたコメント (現 469-471 行) を更新する。
- **Dependencies**: none

### Step 3: SAFE の predictions.csv を index なしに統一

- **Target file**: `src/func/generation_safe_func.py`
- **Changes**: `predictions_df.to_csv(predictions_path)` を `to_csv(predictions_path, index=False)` にする。
  現在 safe だけ index 付きで書いており、`loadGenSmiles` が `index_col=0` で読み分けている。
  index は test split の 0..N-1 と一致しており (全行を保持するため) 情報を失わない。
  docstring の列の説明も更新する。
- **Dependencies**: none

### Step 4: `evaluation.py` の分岐を 2 つに

- **Target file**: `src/evaluation.py`
- **Changes**:
  - データパスの分岐を `rffmg` / それ以外の 2 つにする。
    - `rffmg`: `tr_file_name` は `.../train.target`、`testInputfile` は `.../{additional_path}/test.source`
    - それ以外: `tr_file_name` は `data/{repr_name}/{frag_method}/normal` (HF データセット)、
      `testInputfile` は `None`
  - `safe` のときの `additional_path = 'normal'` 強制を削除する。
    promptsmiles / fraggpt について「固定しない」と決めた方針に揃える
    (存在しない条件を指定した場合はパスが見つからず落ちる、という同じ挙動になる)。
- **Dependencies**: after Step 1, 2

### Step 5: `loadGenSmiles` の分岐を反転して 2 つに

- **Target file**: `src/func/evaluation_func.py`
- **Changes**:
  - 分岐を `if repr_name == 'rffmg':` (predictions.csv + `test.source` を行番号で横結合) と
    `else:` (predictions.csv がすでに `fragment` 列を持つのでそのまま読む) にする。
  - `index_col=0` を廃止する (Step 3 で全手法が index なしになるため)。
  - **time_out / error 行のフィルタを削除する** (下記「判断が必要な点 A」参照)。
  - docstring を新しい役割分担に合わせて書き直す。
- **Dependencies**: after Step 3

### Step 6: 検証

- **Target file**: なし (検証のみ)
- **Changes**:
  - `python -m py_compile` で変更ファイルの構文を確認する。
  - 既存の rffmg predictions.csv で `loadGenSmiles('rffmg', ...)` が従来どおり
    `fragment` / `target` / `prediction_1..50` を返すことを確認する。
  - safe の既存 predictions.csv (index 付きの旧形式) を index なしに読み替えた場合の挙動を確認する。
    旧形式のファイルは Step 3 の変更前に生成されたものなので、SAFE の再生成が必要になる点を報告する。
  - `data/` 配下に生成スクリプトが書き込む箇所が残っていないことを grep で確認する。

## 判断が必要な点

### A. time_out / error 行のフィルタをどうするか (Step 5)

- **推奨: 削除する。** 生成に失敗したこと自体が手法の性能であり、safe だけ失敗行を統計から除くと
  4 手法の比較が公平でなくなる。他 3 手法は失敗行を invalid としてカウントしている。
- 現存の safe 結果では該当 0 行なので、現時点の数値は変わらない。
- 残す場合は、逆に他 3 手法にも「全予測が空の行を除外する」処理を入れて揃える必要がある。

### B. `data/fraggpt/{frag}/normal/test.{source,target}` を削除するか

Step 1 でこれらを書き出さなくなるため、2026-08-04 のスモークラン残骸 (4 行) が孤立する。
削除してよいか確認したい。`data/` 配下のファイル削除なので勝手には行わない。

## Notes (実装対象外・報告のみ)

- Step 3 により SAFE の predictions.csv の形式が変わるため、**SAFE の再生成が必要**。
  現存の `results/safe/gpt/finetuning/brics/beam/predictions.csv` は 1000 行しかなく
  (test split は 82,441 行)、いずれにせよ部分実行の結果である。
- 本計画では `evaluation.py` の `--additional_path` の choices はそのままにする
  (手法ごとに有効な値が違うが、存在しなければパス解決で落ちる)。
