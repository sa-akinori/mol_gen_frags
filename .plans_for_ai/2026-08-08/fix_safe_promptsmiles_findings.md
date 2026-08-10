# Plan: SAFE / PromptSMILES のレビュー指摘を修正する

- **Date**: 2026-08-08
- **Status**: approved

## Overview

SAFE と PromptSMILES を検証エージェントに確認させ、22件の指摘を得た。ユーザーと1件ずつ
突き合わせた結果、**5件を修正**、残りは対応不要と決定した。

うち1件（Step 1）は、同日の `address_review_findings.md` の Step 18 で私が入れた**回帰**である。

### 対応不要と決めたもの（判断の理由つき）

| 指摘 | 対応しない理由 |
|---|---|
| SAFE の学習到達 epoch が 9.3〜23.2 でばらつく | **EarlyStopping(patience=15) の正常動作**。データ量も brics 1.7M / rc_cms 6.6M 行と違うので step/epoch も違う |
| SAFE の既存結果が test の 1.2%（1,000/82,441行） | `--n_generate` の既定は `None`＝全件（`generation_safe_func.py:101`）。既存ファイルは 8/01 の手動部分実行で、`gen_safe.sh` を回せば全件出る。バグではない |
| SAFE の出力先が `beam/normal/` に移動し既存ファイルを評価できない | 未コミットの変更。再生成すれば新パスに出るので自然に解消する |
| batch_size で SAFE の結果が変わる | **乱数ではなく左パディングの浮動小数点誤差**でビームのタイブレークが変わる。SAFE に sampling は無いので FragGPT の対処（sampling で batch_size=1）は適用できない。本番設定（num_beams=50）では 60行中1行の raw SAFE が変わるだけでデコード後は 0/60、実害なし |
| PromptSMILES の `--n_samples` 依存 | 1行ずつ回すと **25.1倍**（16時間 → 400時間）。割に合わない |
| 旧 checkpoint が `save_total_limit` を占有 | transformers の仕様。再学習前に `output_dir` を空にする運用の話 |
| `--resume_from_checkpoint auto` が旧 checkpoint を掴む | `auto` の仕様どおり。help に明記済み |
| SAFE の `--random_seed` が未使用 | beam は乱数を使わないので出力に影響しない |
| SAFE の候補失敗理由が未記録（実測で72.2%が非連結分子） | あると有用だが不具合ではない |
| `safe-train` が site-packages の手パッチに依存 | `README.md:77-112` に記載済み |
| PromptSMILES の beam で uniqueness 1/50（brics の42.6%） | beam の性質。sampling を使うかは研究上の判断 |
| PromptSMILES の prompt 断片が平均1.07個 | 手法の性質。比較の解釈の問題 |
| `joblib` 欠落で生成が起動しない | **ユーザーが別途対応する** |
| `promptsmiles` ライブラリの `FragmentLinker` バグ疑い | 外部・未検証 |

### 決着した事実: datasets のキャッシュはコード変更を検知しない

2つのエージェントで結論が食い違ったため、実ファイルを編集して別プロセスで再実行する形で検証した。

```
版1（s.upper()）    : fingerprint dd7b0ded3eb30c0a  先頭 "ROW00"
関数を s.lower() に書き換えて別プロセスで実行
版2（s.lower()）    : fingerprint dd7b0ded3eb30c0a  先頭 "ROW00"   ← 本来 "row00"
```

`.map()` の fingerprint は「元データセットの fingerprint」「関数の識別子」「`fn_kwargs` の値」から
作られるが、**関数は dill が「モジュール名 + 関数名」だけをピクルする**（実測60バイト）ため
本体の変更が反映されない。`address_review_findings.md` の Step 5 で入れた `CODE_VERSION`
（関数ソースの sha1 を `fn_kwargs` に渡す）で塞がっている。**この件の追加対応は不要。**

## Plan

### Step 1: 可視化に渡す前に NaN を空文字にする（回帰修正）

- **Target file**: `src/func/evaluation_func.py`
- **Changes**:
  - `sc3_check_genmol_results` の `genmols = genmols.where(genmols.notnull(), '')` を、
    **可視化ブロック（`if not skipCreateExcel:`）より前**に移動する。
  - 現在は可視化が先に走るため、NaN セルがそのまま `WriteDataFrameSmilesToXls` に渡り、
    `visualization.py:158-162` が `isinstance(item, str)` でない値を Mol とみなして落ちる。
    ```
    AttributeError: 'float' object has no attribute 'NeedsUpdatePropertyCache'
    ```
  - **RFFMG だけがこれを踏む。** RFFMG の `predictions.csv` は外部コマンド（`t5chem predict` 等）が
    書くため埋め文字を指定できず、失敗が空セル＝NaN になる（実測: 先頭2万行で 5,665セル / 1,409行）。
    他3手法は `INVALID_SMILES` を書くので NaN にならない。
  - **指標の値は変わらない。** `where` は元々 `all_smiles` を作る前に適用されており、
    移動しても計算順序は変わらない。可視化 Excel の該当セルが空の画像になるだけ。
  - `dropna(axis=0)` を戻してはいけない（同計画 Step 18 で削除した理由が有効なため）。
- **Dependencies**: none

### Step 2: SAFE の生成ループから try/except を外す

- **Target file**: `src/func/generation_safe_func.py`
- **Changes**:
  - 生成ループ（140-167行目付近）の `try:` / `except Exception as error:` を外し、
    本体をそのままループ直下に置く。例外はそのまま送出させる。
  - **理由**: 現在はあらゆる例外を飲んでそのバッチを `'error'` で埋めるため、
    `num_return_sequences > num_beams` のような設定ミスや CUDA OOM でも
    **全行 `error` の predictions.csv を書いて exit 0 で終了**する。実測で再現済み。
    ユーザーの方針は「テストデータが欠けるより、やり直しになる方がよい」。
  - 併せて以下を削除する。
    - `error_logs = []` の初期化（136行目付近）
    - `except` 節の `prompts.extend(["error"] * ...)` / `generated_safe.extend(...)` /
      `error_logs.extend(...)`
    - `error_logs.csv` の書き出し（182-184行目付近）
  - **`error_logs.csv` は既存ファイルがヘッダ1行のみ**＝これまで一度もバッチ失敗が
    起きていない。使われていない救済措置のために設定ミスを隠している状態だった。
  - モジュール冒頭の docstring（17行目・19行目付近）の `error_logs.csv` と
    「バッチ単位で例外を捕捉し、失敗した行も `error` を埋めて表に残す」という記述を、
    実際の挙動に合わせて書き換える。
  - **この変更で SAFE の埋め文字 `'error'` が消える**ため、埋め文字の統一漏れも同時に解消する。
- **Dependencies**: none

### Step 3: PromptSMILES の generation_params.txt を predictions.csv と連続保存する

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **Changes**:
  - `LogFile` は `open(fname, "w")` なので、`main()` の冒頭（292行目付近）で開いた時点で
    `generation_params.txt` が切り詰められる。途中でクラッシュすると **params だけ新しく、
    `predictions.csv` は前回の古いものが残る**（エージェントが実際に再現済み）。
  - **`src/func/generation_fraggpt_func.py` と同じ形に揃える。** ログ行を実行中はリストに
    溜めて標準出力にだけ出し、`predictions_df.to_csv(...)` の**直後**に `LogFile` を開いて
    まとめて書き出す。FragGPT 側の `log_line()` の実装をそのまま参考にすること。
  - 生成ループ内の `logfp.write(...)`（生成失敗の記録、340行目付近）も同じリストに積むこと。
  - **起動時に古い `predictions.csv` を削除してはいけない**（誤起動で前回の結果が消えるため）。
    両方が同じタイミングで更新されることだけを保証する。
- **Dependencies**: none

### Step 4: PromptSMILES の生成を行独立にする

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **Changes**:
  - 生成ループ内、`prompter.sample()` を呼ぶ直前（336-340行目付近）に
    `torch.manual_seed(args.random_seed + idx)` を追加する。
  - 現在 `prompter_seed` は `promptsmiles` 側の `random.seed()` にしか渡らず、
    **torch の RNG は行をまたいで持ち越される**。そのため前の分子の生成結果が後の行に影響し、
    行を1つ落として再実行すると以降が全部変わる（エージェントが実測）。
    1行入れるだけで、中断再開・部分再生成・行のスキップに耐えるようになる。コストはゼロ。
  - **`--n_samples` を変えると結果が変わる点は解消しない**（`promptsmiles` 側が
    `n_samples` をバッチ幅として使うため。1行ずつ回すと25.1倍かかるので対応しない）。
    この制約をコメントに残すこと。
  - `import torch` が無ければ追加する（サードパーティの import 順に従う）。
- **Dependencies**: none

### Step 5: 存在しない conda 環境名を直す

- **Target file**: `src/train_model/run_promptsmiles.sh`, `README.md`, `README_ja.md`
- **Changes**:
  - `run_promptsmiles.sh:9` の `conda activate env_promptsmiles` を
    `conda activate promptsmiles` にする。**実在する環境名は `promptsmiles`**（`conda env list` で確認済み）。
    現在は activate が失敗し、`set -e` も `|| exit` も無いため直前の env の python で学習が走る。
  - `README.md` の 195, 231 行目の `env_promptsmiles` を `promptsmiles` にする。
  - `README.md` の 203, 238 行目の `env_fraggpt` を `fraggpt` にする。
  - `README_ja.md` の 33, 34, 41, 42, 53, 54, 193, 200, 227, 234 行目の
    `env_promptsmiles` / `env_fraggpt` を `promptsmiles` / `fraggpt` にする。
  - `src/gen_mols/gen_fraggpt.sh:11` の `conda activate env_fraggpt` も
    `conda activate fraggpt` にする（同じ誤り）。
  - **`run_fraggpt.sh` に `conda activate` を追加してはいけない**（ユーザーが手動で
    activate する方針。同日の別計画で決定済み）。
- **Dependencies**: none

### Step 6: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 全変更ファイルの構文チェックと import 確認
  - **Step 1**: NaN セルを含む DataFrame で `sc3_check_genmol_results` が
    `skipCreateExcel=False` でも落ちないこと。RFFMG の実 `predictions.csv`（NaN を含む）で確認。
    **指標（`nvalid` / `nunique` / `nnovel` / `rank` / `validratio`）が変更前と一致すること**
  - **Step 2**: `num_return_sequences > num_beams` の設定ミスが、
    全行 `error` の CSV を書かずに**例外で落ちる**こと。`error_logs.csv` が作られないこと
  - **Step 3**: 途中で例外を起こしたとき `generation_params.txt` が更新されないこと
  - **Step 4**: 行を1つ落として再実行しても残りの行の結果が変わらないこと（実モデル・少数行）
  - **Step 5**: `conda activate promptsmiles` / `fraggpt` が実際に成功すること。
    `grep -rn "env_promptsmiles\|env_fraggpt"` が0件になること
- **Dependencies**: after Step 5
