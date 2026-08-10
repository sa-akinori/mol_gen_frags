# Plan: レビューで見つかった20件を修正する

- **Date**: 2026-08-08
- **Status**: approved

## Overview

3つの検証エージェントが FragGPT の学習・生成・評価パイプラインを実行して確認し、34件の指摘を出した。
ユーザーと1件ずつ突き合わせた結果、**19件を修正**、15件は対応不要と決定した
（Step 6 は一度採用したのち取り下げた。理由は Step 6 の項に記載）。

対応不要と決めたもの: 失敗理由の優先順位がラベル名で変わる件、`MOLZIP_PARAMS` の関数ローカル化、
train/validation の seed 空間共有、GPU beam の CUDA エラー（`gen_fraggpt.py:11` の
`CUDA_LAUNCH_BLOCKING=1` で回避済み）、`report_to=["wandb"]`（実際に使用中）、
uniqueness/novelty の分母（分子生成の慣例どおり）、`rows=0` での `range()` エラー、
`evaluation.py` が `xlsxwriter` を要求する件、`sanitize_failure` の削除（RDKit の想定外例外の
受け皿として残す）、生成ループのバッチ単位 try/except（**全テスト行で生成が完走している必要が
あるため、落ちたら落ちたと分かる方がよい**という判断）。

## 前提となる実測（すべてこのセッションで確認済み）

### datasets のキャッシュは関数を名前でしかハッシュしない

`encode_fusmiles` の pickle ペイロードは60バイトで、中身はモジュール名と関数名だけ。

```
11: SHORT_BINUNICODE 'train_model.train_fraggpt'
39: SHORT_BINUNICODE 'encode_fusmiles'
本体を差し替えた後の hash: 6a6e5cdefe0d2dfa（元と同一）
```

`--seed` や `--max_length` の変更は正しくキャッシュを無効化するので気づきにくい。

### `seed + idx` はシード間で1行ずれの同一系列になる

```
現行 seed+idx     : seed42/43 の1行ずれ一致 1999/1999
f'{seed}-{idx}'   : 0/1999
sha1(f'{seed}-{idx}') % (2**31-1) : 0/1999
```

実データ（brics 20,000行）で、修正すると **92.9%の行で拡張結果が変わる**。
変わるのは付番と並び順だけで、分子・断片の切り方・断片集合は同一。分布も変わらない。

学習時に行ごとの乱数を使うのは **FragGPT と PromptSMILES の2手法だけ**
（RFFMG は拡張をデータ生成時に済ませてテキストに固定、SAFE は外部 CLI `safe-train` に委譲）。

### `random.Random` はタプルを受け取らない

Python 3.12 で `TypeError: The only supported seed types are: None, int, float, str, bytes, and bytearray`。
文字列シードは `PYTHONHASHSEED` を変えても同一結果を返すことを実測済み（プロセス間で安定）。

### RDKit の `randomSeed` は 2**32-1 まで、2**63-1 で OverflowError

PromptSMILES は RDKit に整数を渡すため、文字列シードは使えない。
`int(sha1(f"{seed}-{idx}").hexdigest()[:8], 16) % (2**31 - 1)` で31bitに収める。

### 失敗時の埋め文字が手法ごとにバラバラ

| 手法 | 埋め文字 | 実データ |
|---|---|---|
| SAFE | `'safe_invalid'` | 7,632個 |
| FragGPT | `''` | — |
| PromptSMILES | `''` | — |
| RFFMG | 埋めない（外部スクリプトが書く） | 欠損0 |

### `all_smiles` の作り方が候補の位置を失わせている

`evaluation_func.py:494` の `(' '.join(x)).strip().split()` は空文字を消すため、
`prediction_k` の k と位置が対応しなくなる。rank を「失敗を詰めない位置」で取るには、
ここを `list(x)` に変える必要がある。

### `Chem.MolFromSmiles('')` は None ではなく原子0個の Mol を返す

```
MolFromSmiles('') is None?  -> False
Smi2CanSmi('')              -> ''
```

`evaluation_func.py:416` の `if Smi2Mol(s) is not None` は空文字を通す。
現在実害が無いのは上記の join/split が先に消しているからで、意図した防御ではない。

### `novel_smi` は chunk TSV 経由で文字列化する

`evaluation_func.py:513` で TSV に書き `:518` で読み直すため、集合が `"{'CCC', 'CCO'}"` という
文字列になる。`evaluation.py:84` はそれを `for` で回すので1文字ずつ取り出す。

```
gensmiles = [' ', "'", ',', 'C', 'O', '{', '}']
```

## スコープ外

- 生成ループのバッチ単位 try/except（ユーザー判断で不採用）
- `sanitize_failure` の削除（受け皿として残す）
- `--eval_strategy`/`--save_strategy` の食い違いがトークナイズ後に落ちる件
  （`choices` によるタイプミス防止のみ行う。食い違い自体はユーザー判断で許容）
- 既存 `predictions.csv` の再生成

## Plan

### Step 1: ダミーのみの断片を失敗として扱う

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**:
  - `assemble_fragments_with_reason` に、**重原子を1つも持たない断片**（`[1*][2*]` など）が
    あれば `dummy_only_fragment` を返す検査を追加する。断片を個別にパースした直後、
    `CombineMols` の前に置くこと。
  - molzip はこうした断片を橋渡しとして繋いでしまう
    （`[1*][2*].[1*]CC.[2*]OC` → `CCOC`）。生成としては失敗なので弾く。
  - docstring の `Returns` の失敗理由一覧に `dummy_only_fragment`（重原子を持たない断片が
    含まれる）を追加する。既存の理由の説明文と同じ体裁に揃えること。
  - `format_failure_summary` は理由文字列を汎用に列挙するため変更不要。
- **Dependencies**: none

### Step 2: 存在しない `ASSEMBLY_OK` への参照を消す

`ASSEMBLY_OK` 定数は存在しない（grep で0件）。2箇所で扱いを変える。

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - 24行目の `assembly_reasons` の説明から ``(:data:`ASSEMBLY_OK` for the assembled ones)`` の
    **一文を丸ごと削除**する。「理由ごとの件数」という説明で足りており、括弧の中は何も足していない。
- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**:
  - 38行目の ``reason`` is :data:`ASSEMBLY_OK` を ``reason`` is ``"ok"`` に置き換える。
    **こちらは残す**。失敗理由は6種類すべて列挙されているのに成功時の値だけ欠けることになり、
    `"ok"` は呼び出し側が実際に使う文字列（`generation_fraggpt_func.py:34` の
    `assembly_reasons['ok']`）で API の一部だから。
  - 同時に冗長な言い回しを詰める。
    ```
    Pair ``(smiles, reason)``. On success ``smiles`` is the canonical SMILES and
    ``reason`` is ``"ok"``. On failure ``smiles`` is None and ``reason`` is one of:
    ```
- **Dependencies**: none

### Step 3: `--eval_strategy` / `--save_strategy` に `choices` を付ける

- **Target file**: `src/train_model/train_fraggpt.py`, `src/train_model/train_promptsmiles.py`
- **Changes**:
  - 両引数に `choices=["steps", "epoch"]` を追加する。他の引数と同じ体裁に揃えること。
  - **`TrainingArguments` の位置は変更しない**（食い違いがトークナイズ後に落ちる点は
    ユーザー判断で許容）。
- **Dependencies**: none

### Step 4: `--resume_from_checkpoint` を追加する

- **Target file**: `src/train_model/train_fraggpt.py`, `src/train_model/train_promptsmiles.py`
- **Changes**:
  - `--resume_from_checkpoint` を追加する（`type=str, default=None`）。
    help には「チェックポイントのディレクトリ、または `auto` で `output_dir` 内の最新」と書く。
  - `trainer.train()` に渡す。`auto` のときは `True` を渡す（transformers が最新を探す）。
    未指定（None）のときは現在と同じ挙動になること。
  - **`save_total_limit` の削除順序についての注意をコメントで残すこと**:
    transformers はステップ番号の昇順に並べて先頭から削除するため、既存の高番号
    チェックポイントが残ったディレクトリで新規学習すると新しい低番号側が先に消える。
- **Dependencies**: none

### Step 5: キャッシュがコード変更で無効化されるようにする

- **Target file**: `src/train_model/train_fraggpt.py`, `src/train_model/train_promptsmiles.py`
- **Changes**:
  - `encode_fusmiles` / `encode_smiles` に引数 `code_version: str` を追加する。
    **本体では使わない**。docstring に「datasets は関数を名前だけでハッシュするため、
    拡張のコードを変えてもキャッシュが無効化されない。この引数を `fn_kwargs` に入れることで
    コード変更が fingerprint に反映される」旨を明記すること（非自明な制約なのでコメントに値する）。
  - モジュールレベルで、拡張に関わる関数のソースから版を作る。
    ```python
    CODE_VERSION = hashlib.sha1(
        (inspect.getsource(encode_fusmiles) + inspect.getsource(augment_fusmiles)).encode()
    ).hexdigest()[:12]
    ```
    `train_promptsmiles.py` では `encode_smiles` のソースを使う（`RandomizeSMILES` は同モジュール内の
    lambda なので、その定義行を含むよう `inspect.getsource(encode_smiles)` に加えて
    モジュール内の該当行も含めるか、少なくとも `encode_smiles` を対象にすること）。
  - `build_lm_dataset` の `.map(fn_kwargs=...)` に `"code_version": CODE_VERSION` を追加する。
  - `import hashlib`, `import inspect` を標準ライブラリの import 順に従って追加する。
- **Dependencies**: none

### Step 6: 【取り下げ】行ごとの乱数をシード間で独立にする

**実装しない。** `random.Random(seed + idx)` / `RandomizeSMILES(smiles, seed + idx + 1)` は
現状のまま残す。

取り下げの理由:

- brics の学習は完了済み、rc_cms も別マシンで進行中で、いま変えると再開時に
  拡張が約93%の行で変わる
- 影響が出るのは「複数シードで学習して分散を測る」用途だけで、その予定がない
- 変わるのは付番と並び順だけで、分布は歪んでいない（拡張として不正ではない）

将来この問題に対処するときの記録として、実測した修正案だけ残す。

```python
# FragGPT: random.Random(f"{seed}-{idx}")           -> シード間の1行ずれ一致が 1999/1999 -> 0/1999
# PromptSMILES: int(sha1(f"{seed}-{idx}").hexdigest()[:8], 16) % (2**31 - 1)
#   （RDKit の randomSeed は整数のみ。2**63-1 で OverflowError になるため31bitに収める）
```

### Step 7: 効かない `PYTHONHASHSEED` の設定を消す

- **Target file**: `src/func/utility.py`
- **Changes**:
  - `set_seed` 内の `os.environ['PYTHONHASHSEED'] = str(seed)`（29行目）を削除する。
    Python は起動時にしかこの変数を読まないため実行中の設定は無効で、
    「設定したつもり」になるだけ。**削除しても過去の結果は変わらない**（元々効いていない）。
  - `os.environ['CUBLAS_WORKSPACE_CONFIG']` の行は**残す**（こちらは有効）。
- **Dependencies**: none

### Step 8: `run_fraggpt.sh` にリポジトリルートへの `cd` を戻す

- **Target file**: `src/train_model/run_fraggpt.sh`
- **Changes**:
  - `cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1` を、`FRAG_NAME` の定義より前に追加する。
    他3本の `run_*.sh` と同じ位置・同じコメントに揃えること。
  - これがないと、リポジトリルート以外から実行したとき
    `python: can't open file 'src/train_model/train_fraggpt.py'` で落ち、
    さらに `mkdir -p "${WANDB_DIR}"` が実行時のカレントディレクトリに `wandb/` を作る。
  - **`conda activate` と shebang は追加しない**（ユーザーが手動で activate する方針）。
- **Dependencies**: none

### Step 9: 生成の乱数を batch_size 非依存にする

**当初「バッチ先頭で seed を張り直せば batch_size 非依存になる」と記載したが、これは誤りだった。**
`torch.multinomial` はバッチ全体に対して1回呼ばれるため、バッチに何行入っているかで乱数の消費が
変わる。バッチ生成を続ける限り行独立にはできない（実測: バッチ先頭で seed を張っても
bs=4 と bs=8 で全行不一致）。

実測した解決策と代償:

| 方式 | 48行 / n_samples=50 | 行独立 |
|---|---|---|
| バッチ生成 (bs=8) | 11.76s | × |
| **1行ずつ生成 + 行番号で seed** | **15.54s (1.32x)** | **○** |

`n_samples=50` で既に実効バッチが50あるため行方向のバッチ化がほとんど効いておらず、
1行ずつでも 1.32 倍に収まる。「全48行の後半24行」と「後半24行だけを生成した結果」が
一致することを実測済み。

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - 生成ループの先頭で `torch.manual_seed(args.random_seed + start)` を呼ぶ（実装済み）。
  - **`gen_method == "sampling"` のときバッチ幅を 1 にする。** ループの刻みと
    `prompts[start:start + ...]` の両方に同じ値を使うこと。バッチ幅が 1 なら `start` が
    行番号そのものになるので、上記の seed の張り方がそのまま行番号 seed として機能する。
    ```python
    # sampling は torch.multinomial がバッチ全体に対して1回呼ばれるため、同じ行でもバッチに
    # 何行入っているかで結果が変わる。1行ずつ生成して行番号で seed を張り、--batch_size にも
    # 生成順序にも依存しないようにする（実測でバッチ生成の 1.32 倍）。beam は乱数を使わないので
    # バッチのままにする。
    batch_size = 1 if args.gen_method == "sampling" else args.batch_size
    ```
  - `--batch_size` の help に **beam のみ有効**である旨を書き添える
    （`sampling` では1行ずつ生成するため無視される）。
  - beam のバッチ生成の挙動は変えないこと。GPU で `bs=24 / bs=8 / bs=1` がすべて一致し、
    別プロセス間でも一致することを実測済み。
- **Dependencies**: none

**PromptSMILES はスコープ外。** 同じ性質を持つが、バッチ処理は外部ライブラリ
`promptsmiles` の `ScaffoldDecorator` / `FragmentLinker` が握っており、構造が異なるため別途扱う。
なお既に生成済みの `predictions.csv` 27件はすべて beam で、sampling の結果は1つもない。

### Step 10: 生成側の `set_seed` を学習側と揃える

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - `from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed` から `set_seed` を外し、
    `from func.utility import BASEPATH, LogFile, set_seed` に含める。
  - 学習は `func.utility.set_seed`（`torch.use_deterministic_algorithms(True)` と
    `CUBLAS_WORKSPACE_CONFIG` を設定）を使っているのに、生成は transformers 版で
    決定性設定が入っていなかった。
  - **メインエージェントが GPU で検証する**。`torch.use_deterministic_algorithms(True)` が
    beam search で例外を起こす場合は報告すること。
- **Dependencies**: none

### Step 11: `generation_params.txt` と `predictions.csv` を連続して保存する

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - `LogFile` は `open(fname, "w")` なので、`main()` の冒頭で開いた時点で
    `generation_params.txt` が切り詰められる。途中でクラッシュすると
    **params だけ新しく、`predictions.csv` は前回の古いものが残る**。
  - ログ行を実行中はリストに溜めて標準出力にだけ出し、
    `predictions_df.to_csv(...)` の**直後**に `LogFile` を開いて書き出す形に変える。
  - **起動時に古い `predictions.csv` を削除してはいけない**（誤って起動したときに
    前回の結果が消えるため）。両方が同じタイミングで更新されることだけを保証する。
- **Dependencies**: none

### Step 12: 候補総数を検算できる形にする

- **Target file**: `src/func/generation_fraggpt_func.py`
- **Changes**:
  - `format_failure_summary(...)` の `n_candidates` に、
    現在の `sum(assembly_reasons.values())`（内訳の合計そのもの）ではなく
    `len(test_smiles) * args.n_samples` を渡す。
  - 現状は「候補総数」と「内訳の合計」が同じ値なので必ず一致し、数え漏れを検知できない。
- **Dependencies**: after Step 11

### Step 13: `gen_fraggpt.py` の出力先に `additional_path` を反映する

- **Target file**: `src/gen_mols/gen_fraggpt.py`
- **Changes**:
  - 38行目の `output_dir` の `normal` ハードコードを `{args.additional_path}` に変える。
    子プロセスには既に `--additional_path` を渡しているのに、時間記録の出力先だけ固定だった。
    `gen_rffmg.py:44` は正しく埋め込んでいる。
- **Dependencies**: none

### Step 14: 誤った help と不要なコメントを消す

- **Target file**: `src/gen_mols/gen_fraggpt.py`
- **Changes**:
  - 26行目 `--additional_path` の help から `(default: empty string)` を削除する
    （実際の default は `"normal"`）。
  - 30行目 `--batch_size` の help から `(default: 24)` を削除する（実際は `8`）。
  - 41-42行目のコメント（`predictions.csv holds the columns ...`）を削除する。
    実際には先頭に `fragment` 列があり記述と食い違っている。
- **Dependencies**: none

### Step 15: 失敗時の埋め文字を手法間で統一する

- **Target file**: `src/func/utility.py`, `src/func/generation_safe_func.py`,
  `src/func/generation_fraggpt_func.py`, `src/gen_mols/gen_promptsmiles.py`
- **Changes**:
  - `func/utility.py` に定数 `INVALID_SMILES = "invalid"` を追加する。
    既存の定数と同じ体裁に揃えること。
  - 以下を `INVALID_SMILES` に置き換える。
    - `generation_safe_func.py:172` の `'safe_invalid'`
    - `generation_fraggpt_func.py` の `smiles or ""`（組立失敗時の埋め）
    - `gen_promptsmiles.py:87` の `to_prediction_row` の右詰め `[""] * n_samples`
  - **RFFMG は埋め文字を出さない**（外部スクリプトが `predictions.csv` を書き、実データでも欠損0）
    ので変更対象外。
  - `Chem.MolFromSmiles("invalid")` は `None` を返すので invalid として数えられ、
    CSV 読み戻し時に NaN にならないため Step 18 の `dropna` 問題も同時に解消する。
- **Dependencies**: none

### Step 16: `novel_smi` を集合として復元する

- **Target file**: `src/evaluation.py`
- **Changes**:
  - 84行目の `for smi in row['novel_smi']` が、TSV 往復で文字列化した集合を
    1文字ずつ回している。`ast.literal_eval` で復元してから回すようにする。
  - **空集合は `"set()"` と書かれ `ast.literal_eval` が `ValueError` を投げる**ので、
    その場合は空集合として扱うこと。この制約をコメントに残す。
  - `import ast` を標準ライブラリの import 順に従って追加する。
- **Dependencies**: none

### Step 17: `nmaxgen` を prediction 列数から決める

- **Target file**: `src/func/evaluation_func.py`
- **Changes**:
  - 475行目の `nmaxgen = 50` を削除し、491行目付近で
    `predcols` を列名から拾って `nmaxgen = len(predcols)` とする。
  - 482行目の可視化列の選択（`np.arange(1, nmaxgen+1)` から選ぶ）も、
    この `nmaxgen` を使うように順序を入れ替える。
  - `evaluation_func` に `nmaxgen` を渡す引数はそのまま残す
    （492行目で prediction 列を落とした後に使うため、その時点では数えられない）。
  - 現状は `--n_samples 10` で `KeyError`、`--n_samples 60` で51番目以降が黙って無視され、
    `validratio` の分母も50のままになる。
- **Dependencies**: none

### Step 18: 候補の位置を保ち、rank を失敗込みで計算する

- **Target file**: `src/func/evaluation_func.py`
- **Changes**:
  - 494行目の `genmols[catsmiCol] = genmols[predcols].apply(lambda x: (' '.join(x)).strip().split(), axis=1)`
    を `list(x)` を使う形に変える。**join/split は空文字を消すため候補の位置が失われる**。
    これにより `all_smiles` は常に `nmaxgen` 個の要素を持つ。
  - 416行目の `valid_smis` の生成に**空文字を除外するガードを追加**する。
    `Chem.MolFromSmiles('')` は `None` ではなく原子0個の Mol を返すため、
    現在の `if Smi2Mol(s) is not None` は空文字を通してしまう。
    これまで実害が無かったのは join/split が先に消していたからで、Step 18 でその防御が外れる。
  - 422-423行目の rank を、**位置を保ったリスト**に対して計算するよう変える。
    無効な候補は `None` などで位置を埋め、`calculateRank` が
    「`prediction_k` の k」を返すようにすること。
  - `unique_smis` / `novel_smi` など他の指標は**詰めたリスト（`valid_smis`）のまま**にし、
    値が変わらないようにすること。
  - **`evaluation_func` は全手法共通なので、SAFE / RFFMG / PromptSMILES の
    top-k accuracy も変わる**（下がる方向）。これはユーザー了承済み。
  - 481-483行目の `dropna(axis=0)` を削除する。FragGPT の空文字が NaN になり
    行ごと落ちるため、可視化サンプルが全成功行に偏っていた（実測 p=0.5 で 0/1000 行）。
    Step 15 で埋め文字が入るので NaN 自体が発生しなくなるが、
    保険として `dropna` は外しておくこと。
- **Dependencies**: after Step 15, Step 17

### Step 19: chunk を完走時にだけ差し替える

- **Target file**: `src/func/evaluation_func.py`
- **Changes**:
  - 502-518行目の chunk 処理を、`{outfd}/chunks.tmp/` に書き出し、
    **全 chunk を書き終えてから** 既存の `{outfd}/chunks/` を削除して `chunks.tmp` を
    リネームする形に変える。
  - 現状は `os.makedirs(exist_ok=True)` のみで古い `chunk_*.tsv` を消さず、
    `glob` で全部拾って concat するため、行数の違う run を続けて回すと別 run の行が混ざる。
  - **起動直後に古い chunks を削除してはいけない**（誤って起動して途中で落ちたときに
    前回の結果が消えるため）。完走したときだけ差し替える。
  - 読み込みは `chunks.tmp` 内のファイルから行い、リネーム後のパスと混同しないこと。
- **Dependencies**: none

### Step 20: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 全変更ファイルの構文チェックと import 確認
  - **Step 1**: `[1*][2*].[1*]CC.[2*]OC` が `dummy_only_fragment` を返すこと。
    正常な FU-SMILES の組み立て結果が実データ1万件×2手法で**変わらない**こと
  - **Step 3**: `--eval_strategy foo` が argparse で弾かれること
  - **Step 4**: `--resume_from_checkpoint auto` で最新チェックポイントから再開すること、
    未指定なら step 0 から始まること
  - **Step 5**: `augment_fusmiles` のソースを変えると `.map` のキャッシュ名が変わること
  - **Step 6**: 取り下げのため検証なし。ただし `random.Random(seed + idx)` が
    **変更されていない**ことを確認すること
  - **Step 7**: `set_seed` が例外なく動き、`CUBLAS_WORKSPACE_CONFIG` が設定されること
  - **Step 8**: リポジトリルート以外から `run_fraggpt.sh` を実行してもパスが解決すること
    （`--help` までで確認し、学習は起動しないこと）
  - **Step 9**: sampling が `--batch_size` を変えても同じ出力になること（実モデル・少数行）
  - **Step 10**: **GPU で beam / sampling が例外なく動くこと**（最重要）
  - **Step 11**: 途中で例外を起こしたとき `generation_params.txt` が更新されないこと
  - **Step 15**: 4手法の `predictions.csv` で埋め文字が統一され、
    `Chem.MolFromSmiles` が `None` を返すこと
  - **Step 16**: 空集合・非空集合の両方で `novel_smi` が正しく復元されること
  - **Step 17**: `--n_samples` が 10 / 50 / 60 のいずれでも `KeyError` にならず、
    `validratio` の分母が正しいこと
  - **Step 18**: 失敗を含む行で rank が `prediction_k` の k を返すこと。
    `nvalid` / `nunique` / `nnovel` が変更前と一致すること（rank 以外は不変）
  - **Step 19**: 途中で落ちたとき古い `chunks/` が残ること。完走したら差し替わること
- **Dependencies**: after Step 19
