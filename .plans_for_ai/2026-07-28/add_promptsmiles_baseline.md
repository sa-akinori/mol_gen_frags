# Plan: PromptSMILES をベースラインモデルとして追加

- **Date**: 2026-07-28
- **Status**: pending-approval

## Overview

PromptSMILES (Thomas et al., *J. Cheminform.* 2024) をベースラインモデルとして追加する。

### 論文調査の結論（設計の根拠）

- 論文は**事前学習済みの化学言語モデル(prior)を流用**している。しかも汎用priorではなく、
  **比較相手の手法の著者が公開したprior**をそのまま使う（SAMOA比較→ChEMBL、LibINVENT/LinkINVENT比較→各著者のChEMBLデータ、GuacaMol→GuacaMol train）。
  つまり論文の思想は「**比較相手と同じデータで学習したpriorを使って公平性を担保する**」こと。
- priorは**プレーンSMILES**で学習された decoder-only モデル（SAFE/SELFIESではない）。
- 同一priorを de novo / scaffold decoration / fragment linking で**使い回す**のが論文の売り。
- 論文のprior学習ハイパラ: batch 128 / LR 1e-3 / epoch 5〜10 / **10-fold restricted randomization augmentation**。

### 本プロジェクトでの設計方針

上記より、**本プロジェクトの既存2手法と同一条件でpriorを用意することが論文に忠実**である。
既存の RFFMG-GPT が既に `entropy/gpt2_zinc_87m`（ZINCのSMILESで事前学習されたGPT2 decoder）を使っているため、
これを PromptSMILES の prior にも流用し、ChEMBL31 で finetuning する。

| | 事前学習重み | 学習データ | 表現・手法 |
|---|---|---|---|
| RFFMG-GPT | `entropy/gpt2_zinc_87m` | ChEMBL31 | フラグメント文 `source>>target` |
| SAFE-GPT | SAFE pretrained | ChEMBL31 | SAFE表現 |
| **PromptSMILES（新規）** | **`entropy/gpt2_zinc_87m`** | **ChEMBL31（同一分割）** | **プレーンSMILES + 推論時プロンプト** |

→ **事前学習重み・分子集合・ハイパラを揃え、表現／手法だけを変える**という理想的な比較になる。

- **ハイパラ**: 既存2手法に合わせる（LR=1e-4 / 50 epoch / batch 32 / warmup 10000 / eval・save 5000 steps / EarlyStopping patience=15 / seed 42）。論文値（LR=1e-3, epoch 5〜10）とは異なるが比較の公平性を優先する。
- **Augmentation: 行わない（ユーザー判断）**。**1分子 = 1系列**とし、学習データはカノニカルSMILESのみ。
  論文は 10-fold restricted randomization を使っているが、本プロジェクトでは採用しない。
  理由: データ量が他手法に対して極端に多くなり、公平な比較にならないため。データ水増しのオプションは実装しない。

- **非カノニカルSMILESへの頑健性（既知のリスクと緩和策）**: PromptSMILES は結合点が末尾に来るよう
  SMILES を**再ルート化＝非カノニカル化**した文字列をプロンプトとして与える手法である。
  prior がカノニカルSMILESしか学習していない場合、非カノニカルなプロンプトへの尤度・生成品質が落ちる可能性がある
  （論文がランダム化を用いていたのはこのため）。
  緩和策として **`--randomize_smiles` フラグ（default: False = 無効）** のみ実装しておく。
  これは**データ件数を増やさず**、各エポックで1分子1系列のまま表記だけをランダム化するオプションであり、
  有効化してもデータ量は RFFMG / SAFE と同条件のまま保たれる。
  デフォルトは無効なので、既定の学習挙動はカノニカルSMILESのみ。

  参考（既存手法のデータ倍率。PromptSMILES はこれらと異なり1分子1系列とする）:

  | モデル | 1分子あたりの行数 | 備考 |
  |---|---|---|
  | SAFE | 実測 5.51（9,956,758行 / 1,808,325分子, brics） | フラグメント化パターン `nFragmentPatterns=5` に由来 |
  | RFFMG | 5 or 10 | `sampling_num`（`run_rffmg.sh` は現在 `SAMPLING_NUM=10`） |
  | **PromptSMILES（本計画）** | **1** | **augmentation なし** |
- **学習目的の違い**: PromptSMILES の prior は無条件言語モデルなので `<bos> SMILES <eos>` の全トークンに loss をかける（RFFMG のようなプロンプトマスクはしない）。
- **モード**: 既存に倣い `finetuning` / `from_scratch` の両方をサポート。

### データ分割の統合（重要）

`src/make_datasets.py` が既に **SAFE と RFFMG の共通分割点**になっている:

- `rffmg_frags['smiles'].unique()` → `random.seed(0)` でシャッフル → 0.95/0.025/0.025 で `tr_smiles`/`val_smiles`/`te_smiles`
- 同一リストを RFFMG と SAFE の両方に適用。test は `--test_mol_num 20000` でサブサンプルし両者共通。

したがって PromptSMILES 用の独立コーパススクリプトは**作らない**。
`make_datasets.py` 内の既存 `tr_smiles`/`val_smiles`/`te_smiles` をそのまま使って `.smi` を書き出す。

注意: `unique_smiles` は `data/rffmg/{frag_method}/{N}times_sampling/full_dataset.csv` 由来のため
**分割は frag_method ごとに異なる**。よって PromptSMILES の出力も SAFE と同様 `frag_method` 別ディレクトリに置く。

### 環境

プロジェクト規約に合わせ、`requirements/` 配下の requirements 方式で用意する（`safe_requirements.txt` / `t5chem_requirements.txt` に倣う）。

## Plan

### Step 1: requirements ファイルの追加

- **Target file**: `requirements/promptsmiles_requirements.txt`（新規）
- **Changes**: 既存の requirements に倣いバージョンを固定して記述。
  `promptsmiles`, `rdkit==2025.3.6`, `transformers==4.45.2` を記載（torch は既存同様 requirements 外で個別導入）。
  想定 conda 環境名は `env_promptsmiles`。
- **Dependencies**: none

### Step 2: トークナイザ適合性の事前確認（小規模・先行実施）

- **Target file**: `src/debug_promptsmiles_tokenizer.py`（新規・使い捨て確認用）
  **※ 2026-07-28 に役目を終えたため削除済み。** 本番パイプラインからは参照されていなかった。
  結果は下記「実施結果メモ」に記録してある（再確認が必要な場合は作り直すこと）。
- **Changes**: `entropy/gpt2_zinc_87m` の tokenizer が PromptSMILES のプロンプト
  （**環開き・分岐が未閉のまま途中で切れた部分SMILES**、および結合点 `*` を含む文字列）を
  破綻なくトークナイズ／デコードできるかを検証する。往復一致率を標準出力とログに記録。
  ここで不適合が判明した場合は Step 3 以降の設計（tokenizer 差し替え等）を見直すため、**先に実施する**。
- **Dependencies**: after Step 1

### Step 3: make_datasets.py へ PromptSMILES 用データ出力を統合

- **Target file**: `src/make_datasets.py`
- **Changes**: 既存の `tr_smiles` / `val_smiles` / `te_smiles`（SAFE・RFFMG と共通）をそのまま使い、
  1行1SMILES のプレーンテキストを `data/promptsmiles/{frag_method}/normal/{train,val,test}.smi` に出力する処理を追加。
  **SAFE と同様に `if args.sampling_num == SAFE_SAMPLING_NUM:` のガード内で出力する**
  （`sampling_num` により分子分割が変わるため。実測差異は「実施結果メモ」参照）。
  test は既存のサブサンプル後 `te_smiles` と完全一致させる。既存に倣い `debug/` 版（先頭10000件）も出力。
  新規の分割ロジックは追加しない（既存変数の再利用のみ）。件数を既存ログ機構に記録。
- **Dependencies**: none（Step 2 と並行可）

### Step 4: prior 学習スクリプト

- **Target file**: `src/train_model/train_promptsmiles_gpt.py`（新規）
- **Changes**: `train_gpt.py` の構成・コードスタイルを踏襲しつつ、**無条件言語モデル**として学習する。
  - Dataset: `<bos> SMILES <eos>`、labels は全トークン（`-100` マスクなし）。`max_length` 超過は例外送出（既存同様）。
  - Augmentation: **行わない**。1分子1系列（カノニカルSMILES）。データ水増しの引数は設けない。
    代わりに `--randomize_smiles`（default: `False`）のみ用意し、有効時は件数を変えずに表記のみランダム化する。`set_seed(42)` 配下で再現可能に。
  - `--mode {finetuning, from_scratch}`、`--pretrain entropy/gpt2_zinc_87m`。
  - ハイパラ default は既存2手法に一致（epoch 50 / LR 1e-4 / batch 32 / warmup 10000 / eval・save 5000 / save_total_limit 5 / EarlyStopping 15 / seed 42 / max_length 256）。
  - 出力: `models/promptsmiles/gpt/{mode}/{frag_method}/`、`best_model/` を保存。
  - 型ヒント・Google style docstring 必須。
- **Dependencies**: after Step 2, Step 3

### Step 5: 学習実行シェルスクリプト

- **Target file**: `src/train_model/run_promptsmiles.sh`（新規）
- **Changes**: `run_rffmg.sh` / `run_safe.sh` の作法を踏襲。リポジトリルートへ cd、
  `conda activate env_promptsmiles`、`CUDA_VISIBLE_DEVICES=0`、
  `WANDB_MODE=offline` と `WANDB_DIR="wandb/promptsmiles/gpt/${MODE}/${FRAG_NAME}"` を設定して
  `train_promptsmiles_gpt.py` を実行。`FRAG_NAME` / `MODE` を上部変数で切替。
- **Dependencies**: after Step 4

### Step 6: 生成スクリプト（scaffold decoration / fragment linking）

- **Target file**: `src/gen_mols/gen_promptsmiles.py` + `src/gen_mols/gen_promptsmiles.sh`（新規）
- **Changes**: finetuning 済み GPT2 をロードし、`promptsmiles` の `ScaffoldDecorator` / `FragmentLinker` に
  - `sample_fn(prompt, batch_size) -> list[str]`（HF `model.generate` によるサンプリング）
  - `evaluate_fn(smiles_list) -> negative log-likelihood`
  を渡して生成する。
  評価対象のスキャフォールド／フラグメントは、**共通テスト分子 `te_smiles` から既存の `src/func/fragmentation.py` を用いて生成**し、
  PromptSMILES 記法（結合点 `*`、フラグメント区切り `.`）へ変換する。
  引数: `--task {scaffold, linking}`, `--frag_method {brics, rc_cms}`, `--mode {finetuning, from_scratch}`,
  `--n_samples`, `--random_seed 42`。SMILES は `Chem.MolFromSmiles` の None チェックを必ず実施。
  出力は `src/evaluation.py` が読める形式・パスへ保存。
- **Dependencies**: after Step 5

### Step 7: 評価パイプラインへの統合

- **Target file**: `src/evaluation.py`（+ 必要に応じて `src/func/evaluation_func.py`）
- **Changes**: `--model_name` の choices に `promptsmiles` を追加し、`str_name` / `model_dir` / `arc_name` の分岐と
  生成結果の読み取りを追加。validity / uniqueness / novelty / 再構成率を RFFMG・SAFE と**同一関数**で算出できるよう配線する。
- **Dependencies**: after Step 6

### Step 8: 再現性・ログの担保（横断）

- **Target file**: Step 3〜7 の各ファイル
- **Changes**: `set_seed(42)` の使用、乱数シードの明示、ハイパラ・メトリクスのログ保存、
  wandb offline ディレクトリの分離、データパスの `Path` / `BASEPATH` 経由での取り扱いを徹底。
- **Dependencies**: Step 3〜7 に内包

## 実行順序と実行時間の見積もり

- Step 1→2 を先に済ませ、tokenizer の適合性を確認してから Step 4 以降の本実装に進む。
- **学習規模**: brics の場合 分子 1,808,325 件、train は 95% で約 1,717,900 分子。
  augmentation を行わないため **1エポックあたり約172万系列**、`batch_size=32` なので **1エポック ≈ 約53,700 steps**。
  `num_train_epochs=50` に到達する前に EarlyStopping（検証15回連続＝75,000 steps 改善なし）で停止する想定。
  augmentation を行う場合と比べて1エポックあたりのstep数は約1/10であり、他手法より学習データ量は少なくなる
  （＝1分子1系列という設計上の意図通り）。
- **学習の実行はユーザーが行う**。本計画のスコープは実装までとし、実装完了後に実行手順を提示する。
  実行は SAFE / RFFMG と同じ GPU（`CUDA_VISIBLE_DEVICES=0`）を使う想定。
- **実施モード**: `finetuning` と `from_scratch` の**両方を学習する**（既存2手法と同じ構成に揃えるため）。
  スクリプトは両モードを切り替えられる形で用意する。

## 用語の定義（本計画内）

- **step**: 1バッチ分の学習＝1回のパラメータ更新。`1エポックのstep数 = 学習データ件数 ÷ batch_size`。
- `warmup_steps=10000`: 最初の10,000更新で学習率を 0→1e-4 へ線形に上げる。
- `eval_steps=5000` / `save_steps=5000`: 5,000更新ごとに検証・チェックポイント保存。
- `EarlyStopping(patience=15)`: 検証15回連続（＝75,000 steps）で改善がなければ学習を打ち切る。

## 実施結果メモ（2026-07-28 追記）

### Step 2 の結果: tokenizer は適合（未確定事項は解消）

`entropy/gpt2_zinc_87m`（vocab 2,707、bos/eos/pad/unk 完備）に対する往復一致率:

| カテゴリ | 件数 | 往復一致率 | unk を含む件数 |
|---|---|---|---|
| molecule（カノニカルSMILES） | 300 | 1.0000 | 0 |
| attachment（`*` 付き） | 291 | 1.0000 | 0 |
| prompt（`*` を末尾に再ルート化 → `*` 削除） | 269 | 1.0000 | 0 |
| truncated（環開き・分岐未閉が残る切断） | 269 | 1.0000 | 0 |

→ 非カノニカルな部分SMILESを問題なく扱えるため、**tokenizer の差し替えは不要**。Step 4 以降は計画どおり進める。

### sampling_num による分割差異の実測と対応（ユーザー決定）

`unique_smiles` は `data/rffmg/{frag}/{N}times_sampling/full_dataset.csv` 由来のため、`sampling_num` により分子集合が変わることを実測で確認した。

| | 5times | 10times |
|---|---|---|
| ユニーク分子数（brics） | 1,808,325 | 1,810,605 |
| 集合の一致 | 不一致（順序も不一致） | |

**対応（ユーザー決定）**: SAFE と同じガードを適用し、**`sampling_num == SAFE_SAMPLING_NUM`（=5）のときだけ**
PromptSMILES の `.smi` を出力する。これにより SAFE および RFFMG-5times と完全に同一の分子分割となり、
prior は frag_method あたり1本で済む。

**既知の既存不整合（PromptSMILES とは独立）**: SAFE は5timesの分割に固定される一方、
`run_rffmg.sh` は現在 `SAMPLING_NUM=10` のため、SAFE と RFFMG-10times は既に別の分子分割になっている。

## 追加ステップ（2026-07-28 追記・ユーザー承認済み）

### 背景: 実測で判明した手法固有の制約

`promptsmiles` のサンプラーは `DeNovo` / `ScaffoldDecorator` / `FragmentLinker` の3つのみ。
RFFMG のフラグメント集合をそのまま入力する用途には**どちらも使えない**ことを実測で確認した。

| 入力の形 | ScaffoldDecorator | FragmentLinker |
|---|---|---|
| 1フラグメント・`*` 1個 | OK | OK |
| 1フラグメント・`*` 複数個 | OK（順に装飾） | **AssertionError**（`*` は1個のみ） |
| 複数フラグメント（各 `*` 1個） | 受理されるが**連結されない**（成分数が入力のまま） | OK（3個以上で `scan=True` 自動化、4個まで検証済み） |
| 複数フラグメント（`*` 複数個を含む） | **IndexError: list index out of range** | AssertionError |

実データ（brics, test 82,441集合）の実測値:

- 1集合あたりのフラグメント数: 3個 29.4% / 4個 24.6% / 2個 17.7%（最大11個）
- `*` が2個以上のフラグメント: 全300,336件中 **172,725件（57.5%）**
- **全フラグメントが単一結合点の集合: 7,169件（8.7%）** ← linking で情報を捨てずに扱える集合
- 単一結合点が1個以下（linking 不可）: 39,789件（48.3%）

### 方針（ユーザー決定: ①②併用）

- **scaffold decoration**: 全件で評価。入力は「`*` を1個以上持つ最大フラグメント**1つ**」（連結した単一フラグメントのみ渡すので上記 IndexError を回避できる）。
- **fragment linking**: **全フラグメントが単一結合点の集合のみ**に限定し、その集合の**全フラグメントを渡す**（`N_LINK_FRAGMENTS` の2個固定を撤去）。RFFMG も同じ部分集合で評価すれば同一母集団の直接比較になる。

### Step 9: 例外処理とスキップ件数のログ化

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **Changes**:
  - **現状 `prompter.sample()` に例外処理が無く、1分子の失敗で生成全体が異常終了する**。これを修正する。
  - prompter の構築と `sample()` を `try/except Exception` で囲み、失敗分子はスキップする。
  - **例外を黙って飲み込まないこと**: 例外の型・メッセージ・対象分子・フラグメント集合を `LogFile` に必ず記録する。
  - スキップ理由ごとの件数（例外発生 / フラグメント条件を満たさない / 断片化失敗）を集計し、
    実行終了時に標準出力と `LogFile` の両方へ出力する。
  - **行アラインメントの担保（重要）**: 現在 `test.source` / `test.target` は生成の *前* に書き出されているため、
    失敗分子をスキップすると `predictions.csv` と行数がずれる。`evaluation_func.py` は
    **行番号で結合する**ため評価が壊れる。
    → `test.source` / `test.target` は**生成に成功した分子のみ**を対象に、生成後に書き出すよう変更し、
    3ファイルの行数と順序が常に一致することを保証する。
- **Dependencies**: after Step 6

### Step 10: linking の入力条件変更

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **Changes**:
  - 定数 `N_LINK_FRAGMENTS`（=2 固定）を撤去する。
  - `select_link_fragments` を、**フラグメント集合の全フラグメントが `*` をちょうど1個持つ場合にのみ
    その全フラグメントを返し、それ以外は None を返す**仕様に変更する。
  - 3個以上では `scan=True` が自動で有効になり計算量が増える点をコメントに明記する。
  - スキップ件数は Step 9 のログ機構に載せる。
- **Dependencies**: after Step 9

### Step 11: テスト入力を既存データの読み込みに変更（整合性の担保）

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **背景（実測）**:
  - 現状はテスト分子を `Smi2Sentences` で**断片化し直している**ため、**19.1% が断片化に失敗して脱落**し、
    残りもフラグメント集合が RFFMG/SAFE と別物になりうる。train/val は一致しているが **test だけ不整合**。
  - **生成時の入力は RFFMG も SAFE も `pass_fragments`** であることを確認した:
    - RFFMG: `test.source` = `pass_fragments`（同位体ラベルを除去した形）
    - SAFE: `generation_safe_func.py` 86-88行で `test_dataset['pass_fragments']` を読み、
      `scaffold_decoration(scaffold=pass_frag)` / `scaffold_morphing(side_chains=...)` に渡す
    - `full_safe` は**学習テキスト専用**で生成には使わない。
  - よって本研究の目的（指定したフラグメントから生成する）に対応する入力は **`pass_fragments`** である。
    `full_fragments` を使うと PromptSMILES だけ完全な分解を与えられ比較が成立しない。
  - SAFE も `scaffold_decoration` を使っており、PromptSMILES の `ScaffoldDecorator` と同じ枠組みで直接対応する。
- **Changes**:
  - 再断片化（`Smi2Sentences` / `make_fragment_option` / `build_fragment_set`）を廃止する。
  - `datasets.load_from_disk(f'{BASEPATH}/data/safe/{frag_method}/normal')` の **test split** を読み、
    - 正解分子 = `smiles` 列
    - プロンプトの元 = **`pass_fragments` 列**
    を使う（82,441行 / 固有分子20,000件。RFFMG の `test.source` と行数・分子とも一致）。
  - `select_scaffold_fragments` / `select_link_fragments` は `pass_fragments` 文字列に対してそのまま適用する。
  - Step 9 で入れた行アラインメント保証とスキップ件数ログはそのまま維持する。
- **Dependencies**: after Step 10

### Step 12: バッチプロンプト対応（生成時間の担保）

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **背景（実測）**: 現状は `batch_prompts=False` のため `sample_fn` が**1サンプルずつ逐次**で呼ばれる
  （n_samples=50 なら1行あたり50回、CPU実測 3.7秒/行・結合点1個）。82,441行では**85時間以上**かかり非現実的。
  `batch_prompts=True` にすると呼び出しが 1/50 になり、GPUバッチが効く。
- **Changes**:
  - `GPT2PromptSampler.sample` の `prompt` を `str | list[str]` に対応させる。
    リストの場合は**左パディング**でバッチ生成し、`prompt[i]` に対応する補完を `smiles[i]` として返す。
  - `ScaffoldDecorator` / `FragmentLinker` に `batch_prompts=True` を渡す。
  - **返す SMILES が対応するプロンプトを接頭辞として含むこと**（`samplers.py` の
    `assert smiles.startswith(prompt)`）を、バッチ経路でも必ず満たすこと。
  - `evaluate` 側は変更不要。
- **Dependencies**: after Step 11

### Step 13: デコード方式（sampling / beam）の選択対応

- **Target file**: `src/gen_mols/gen_promptsmiles.py`, `src/gen_mols/gen_promptsmiles.sh`, `src/evaluation.py`
- **背景（実測）**: 現状のデコード方式は3手法で揃っていない。
  | 手法 | 方式 | パラメータ |
  |---|---|---|
  | RFFMG (T5Chem / GPT) | ビームサーチ | `do_sample=False`, `num_beams=50`, `num_return_sequences=50` |
  | SAFE | ビームサーチ | `do_sample=False`, `num_beams=50`, `how='beam'` |
  | PromptSMILES（現状） | 多項サンプリング | `do_sample=True` のみ |

  デコード方式が違うと validity / uniqueness / novelty が系統的にずれ、「手法の差」と「デコードの差」が混ざる。
- **Changes**:
  - `gen_promptsmiles.py` に `--gen_method {sampling, beam}`（default: `sampling`）と
    `--num_beams`（default: 50。SAFE / RFFMG と同値）を追加する。
    - `sampling`: 現状どおり `do_sample=True`
    - `beam`: `do_sample=False`, `num_beams=args.num_beams`
  - `batch_prompts=True` の経路では「プロンプト1本につき補完1本」を返す契約なので、
    beam では `num_return_sequences=1` とし、プロンプト数分の出力を返すこと。
  - **プロンプトを接頭辞として含む保証（`assert smiles.startswith(prompt)`）は beam 経路でも維持**すること
    （補完部分のみをデコードしてプロンプト文字列と連結する現在の方式をそのまま使う）。
  - 出力先を `results/promptsmiles/gpt/{model_ver}/{frag_method}/{gen_method}/{task}/` とし、
    `sampling` と `beam` の結果が同居できるようにする（既存の `gen_method` によるパス分離をそのまま利用）。
  - `gen_promptsmiles.sh` の冒頭変数に `GEN_METHOD` を追加する。
  - `evaluation.py`: 現在 `promptsmiles` は `gen_method='sampling'` 固定。`--gen_method {beam, sampling}`
    （default: `None`）を追加し、未指定ならモデルごとの既定（既存3モデルは `beam`、promptsmiles は `sampling`）に
    フォールバックする。**既存3モデルの挙動・パスは変更しないこと**。
- **既知の注意点（実装時にコメントで残す）**:
  - beam は同一プロンプトから常に同じ出力になるため、scaffold decoration では
    生成された `n_samples` 本が重複しやすい。uniqueness が下がる想定。
  - `num_beams=50` × バッチ内プロンプト数の分だけビームが同時展開されるためメモリを消費する。
    OOM する場合は `--n_samples` を下げて対応する旨をコメントに残す。
- **Dependencies**: after Step 12

### Step 14: タスクの自動振り分け（`--task` の廃止・全行を生成対象にする）

- **Target file**: `src/gen_mols/gen_promptsmiles.py`, `src/gen_mols/gen_promptsmiles.sh`, `src/evaluation.py`
- **背景**: 従来は `--task {scaffold, linking}` で別々に実行し、条件を満たさない行は**スキップ**していた。
  そのため linking は全体の 8.4% しか処理できず、RFFMG / SAFE（全82,441行）と母集団が揃わなかった。
  **スキップをやめ、行ごとに適したサンプラーへ振り分ける**ことで全行を生成対象にする。
- **振り分け規則（実測に基づく）**:

  | 条件 | 使用するサンプラー |
  |---|---|
  | 全フラグメントが `*` ちょうど1個 **かつ 2個以上** | `FragmentLinker`（全フラグメントを渡す） |
  | それ以外（多価フラグメントを含む / 1フラグメントのみ） | `ScaffoldDecorator`（`*` を持つ最大フラグメント1個） |

  `FragmentLinker` の制約（実測で確認済み）:
  - 各フラグメントは `*` ちょうど1個（`samplers.py` の `assert`）
  - **フラグメントが1個だけだと `IndexError: pop from empty list`**（614/692行の `fragments.pop()` が空リストに当たる）。
    `*` が1個でも単一フラグメントでは使えない。
- **Changes**:
  - `gen_promptsmiles.py`
    - **`--task` 引数を廃止**する。1回の実行で全行を処理する。
    - 上記規則で行ごとにサンプラーを振り分ける。**条件を満たさない行をスキップしない**
      （`fragment_condition_unmet` によるスキップは廃止）。
    - 例外は従来どおり `try/except` で捕捉し、対象分子・フラグメント集合・例外型とメッセージをログに記録して次の行へ進む。
      安全網として残すが、通常は発生しない想定。
    - **どちらのサンプラーで生成したかを行ごとに記録する**（`predictions.csv` に `sampler` 列を追加。
      値は `scaffold` / `linking`）。評価時に切り分けられるようにするため。
    - サンプラー別の処理件数を実行終了時のサマリに出す（例: `sampler linking: 6,900 / sampler scaffold: 75,541`）。
    - 出力先から `{task}` 階層を除き、`results/promptsmiles/gpt/{model_ver}/{frag_method}/{gen_method}/normal/` とする。
      プロンプトの書き出し先も `data/promptsmiles/{frag_method}/normal/test.{source,target}` とする。
  - `gen_promptsmiles.sh`: `TASK` 変数と `--task` の受け渡しを削除する。
  - `evaluation.py`: `--additional_path` の choices から `scaffold` / `linking` を削除し、
    promptsmiles も他モデルと同じ `normal` を使う。**既存3モデルの挙動は変更しないこと**。
- **注意点（実装時にコメントで残す）**:
  - `predictions.csv` の列が `target, sampler, prediction_1..N` になるため、
    `evaluation_func` 側が列位置に依存していないか確認すること。依存していれば `sampler` 列は別ファイルに出す。
- **Dependencies**: after Step 13

### Step 15: 学習・検証データも SAFE のデータセットから読む（単一データソース化）

- **Target file**: `src/train_model/train_promptsmiles.py`, `src/evaluation.py`
  （`src/make_datasets.py` の promptsmiles ブロック削除は**ユーザーが自分で行う**）
- **背景**: 生成（test）は既に SAFE のデータセットを読んでいるのに、学習・検証だけ
  `data/promptsmiles/{frag}/normal/{train,val}.smi` という別ファイルを読んでいた。
  データソースを1つに統一すれば取り違えが起きず、`data/promptsmiles/` 自体が不要になる。
- **実測（brics）**: SAFE のデータセットは train / validation とも **1分子1行**なので重複除去は不要。

  | split | 行数 | 固有分子 | 分割の分子数との差 |
  |---|---|---|---|
  | train | 1,714,298 | 1,714,298 | −3,610（0.21%） |
  | validation | 45,203 | 45,203 | −5（0.01%） |
  | test | 82,441 | 20,000 | 一致 |

  差は `make_datasets.py` の `drop_duplicates(subset='full_safe')` に由来する。
  結果として PromptSMILES の学習分子は **SAFE と厳密に一致**し、RFFMG より 0.21% 少なくなる。
  3手法すべてと完全一致させることは（SAFE 自体が RFFMG より少ないため）不可能で、この差は許容する。
- **使用する split と列**:

  | 用途 | split | 列 |
  |---|---|---|
  | 学習 | `train` | `smiles` |
  | 検証 | `validation` | `smiles` |
  | 生成 | `test` | `pass_fragments`（正解は `smiles`） |

- **Changes**:
  - `train_promptsmiles.py`: `.smi` の読み込みをやめ、
    `datasets.load_from_disk(f'{BASEPATH}/data/safe/{frag_method}/normal')` の
    `train` / `validation` split の `smiles` 列を使う。`read_lines` が不要になれば削除する。
  - `evaluation.py`: promptsmiles の `tr_file_name` を `f'{BASEPATH}/data/safe/{frag_method}/normal'` に変更する。
    - `loadTrainSmiles` には既に `safe_gpt` 経路（`load_from_disk` → `smiles` 列）があるため、
      **`src/func/evaluation_func.py` は変更しない**。
    - `arc_name` は `loadTrainSmiles` と `loadGenSmiles` の両方で使われており、promptsmiles は
      生成結果の読み取りには `t5chem` 経路が必要。よって**学習データ読み込み用の変数を分ける**
      （例: `train_arc_name`）。promptsmiles と safe_gpt は `'safe_gpt'`、それ以外は `'t5chem'`。
    - **既存モデル（t5chem / gpt / safe_gpt / fraggpt）の挙動・パスは一切変更しないこと。**
      `evaluation.py` は RFFMG でも実行されるため、影響が出ないことを最優先する。
- **Dependencies**: after Step 14

### Step 16: PromptSMILES 専用データセットへの切り替え（ユーザーが make_datasets.py に追加）

- **Target file**: `src/train_model/train_promptsmiles.py`, `src/gen_mols/gen_promptsmiles.py`, `src/evaluation.py`
- **背景**: Step 15 では SAFE のデータセット（`data/safe/{frag}/normal`）を直接読んでいたが、
  「異なる手法のデータを読み込むとコードとして分かりにくい」というユーザー判断により、
  **ユーザーが `make_datasets.py` に PromptSMILES 専用データセットの出力を追加した**（139-149行付近）。
  コード側をこの新しいデータソースに合わせる。
- **新しいデータセット**: `data/promptsmiles/{frag_method}/normal`（HF `DatasetDict`）
  - splits: `train` / `validation` / `test`
  - 列: **`smiles`** と **`pass_fragments`** の2列のみ
  - 由来: `safe_tr` / `safe_val` / `safe_te` から該当2列を抜き出したもの。よって分子・行数は SAFE と一致する
    （train 1,714,298行=1分子1行 / validation 45,203行=1分子1行 / test 82,441行=20,000分子）
- **Changes**:
  - `train_promptsmiles.py`: 読み込み先を `data/safe/{frag}/normal` から
    **`data/promptsmiles/{frag}/normal`** に変更。`train` / `validation` split の `smiles` 列を使う。
  - `gen_promptsmiles.py`: 読み込み先を `data/safe/{frag}/normal` から
    **`data/promptsmiles/{frag}/normal`** に変更。`test` split の `pass_fragments`（入力）と `smiles`（正解）を使う。
  - `evaluation.py`: promptsmiles の `tr_file_name` を **`data/promptsmiles/{frag}/normal`** に変更。
    `train_arc_name` の仕組み（`safe_gpt` 経路で `load_from_disk` → `smiles` 列）はそのまま使える。
    **既存4モデルの挙動は一切変更しないこと。**
  - 生成時に書き出す `test.source` / `test.target` は同ディレクトリのままでよい
    （`save_to_disk` が既存ファイルを削除しないことを実測で確認済み）。
- **Dependencies**: after Step 15

### Step 17: 事前学習モデルの定数化（`--pretrain` の廃止）

- **Target file**: `src/train_model/train_promptsmiles.py`, `src/train_model/train_gpt.py`
  （`train_fraggpt.py` は別セッションが並行編集中のため対象外）
- **背景（ユーザー判断）**: 比較対象のベースラインはいずれも **ZINC で事前学習された GPT2
  （`entropy/gpt2_zinc_87m`）のみ**を使う。実測で確認した事実:
  - `run_promptsmiles.sh` / `run_rffmg.sh` とも `--pretrain` を渡しておらず、常に既定値が使われる
  - RFFMG の T5Chem は `run_rffmg.sh` のシェル分岐で外部CLI（`t5chem train`）が呼ばれるため、
    `train_gpt.py` は GPT2 専用。`train_*.py` の `--pretrain` はどれも実質使われていない
  - `entropy/gpt2_zinc_87m` は `pad=<pad>(id=1)` / `bos=<s>` / `eos=</s>` をすべて持つ
- **Changes**:
  - `--pretrain` 引数を廃止し、モジュール定数 `PRETRAINED_MODEL = "entropy/gpt2_zinc_87m"` に置き換える。
    `AutoTokenizer.from_pretrained` / `GPT2LMHeadModel.from_pretrained` / `GPT2Config.from_pretrained`
    の参照を差し替える。
  - 定数化により `if tokenizer.pad_token_id is None:` と
    `if tokenizer.bos_token_id is None or ...: raise ValueError` は**到達不能なデッドコード**になるため削除する
    （`train_promptsmiles.py` では既にユーザーが削除済み。`train_gpt.py` を揃える）。
  - `run_rffmg.sh` / `run_promptsmiles.sh` は `--pretrain` を渡していないので**変更不要**。
- **Dependencies**: after Step 16

### Step 18: SMILES ランダム化の採否（**最終決定: 行う**）

- **経緯**: いったん「行わない」と決定したが（理由: RFFMG・SAFE がランダム化 augmentation を
  行っておらず、PromptSMILES だけ行うと3手法の条件が揃わないため）、
  下記の実測により **2026-08-03 に「行う」へ変更**した。実装は Step 19。
- **判断を変えた根拠（実測）**: カノニカルSMILESのみで学習すると、
  **学習データの開始原子が系統的に偏る**ことが分かった。
  データセットの SMILES は既にカノニカル形式で保存されているため、
  パース後に再カノニカル化すると **6分子すべてで開始原子 index が 0**（カノニカル順の先頭）になった。

  | 原子数 | カノニカル出力の開始原子 index |
  |---|---|
  | 42 / 33 / 14 / 33 / 34 / 33 | いずれも **0** |

  一方 PromptSMILES の推論プロンプトは、結合点が末尾に来るよう再ルート化されるため
  **任意の位置から始まる**（例: `n1c(N)c(cccc2)c2nc1`）。
  つまり学習と推論で表現が食い違う。**これは PromptSMILES 固有の問題**であり、
  RFFMG・SAFE は推論時も学習時と同じ表現形式を使うため生じない。
  よって「他手法と条件を揃える」ことを理由にランダム化を外す論拠は成り立たない。
- **論文の記述（Appendix A）もこれを支持する**:

  > "We endeavoured to follow model architectures and parameters to compared models in each
  > experiment where possible, **with the exception that we always trained the respective RNN
  > with SMILES augmentation by 10-fold restricted randomization**. This was done due to the
  > fact that **PromptSMILES rearranges SMILES strings forming non-canonical representations**."

  著者も「他の条件は比較対象に揃えたが、ランダム化だけは例外として必ず入れた」としている。
- **論文との差異（報告時に明記すべき点）**:
  PromptSMILES 論文 Appendix A は、この点を**意図的な例外**として次のように述べている。

  > "We endeavoured to follow model architectures and parameters to compared models in each
  > experiment where possible, **with the exception that we always trained the respective RNN
  > with SMILES augmentation by 10-fold restricted randomization**. This was done due to the
  > fact that **PromptSMILES rearranges SMILES strings forming non-canonical representations**."

  つまり著者は「他の条件は比較対象に揃えたが、ランダム化だけは例外として必ず入れた」としている。
  本プロジェクトはプロジェクト内の公平性を優先し、**この要素を意図的に採用しない**。
  - 倍率10の根拠は論文に示されておらず、2倍・5倍との比較実験（アブレーション）も**行われていない**。
    `restricted randomization` の手法自体は Arús-Pous et al. (2019) を引用。
  - 想定される影響: プロンプトは再ルート化された非カノニカル文字列
    （例: `n1c(N)c(cccc2)c2nc1`）なので、カノニカルのみで学習した prior は不利になりうる。
    結果が想定より低い場合、この点を要因候補として検討する。
- **毎エポックのランダム化は採用しない**: 本プロジェクトは `eval_loss` で EarlyStopping と
  best model 選択を行うため、val の表記が毎回変わると評価が不安定になる。
  なお論文の10倍augmentationも事前生成の固定データセットであり、毎エポック引き直しではない。
- **Dependencies**: after Step 17

### Step 19: SMILES ランダム化の実装（データ量は1倍のまま）

- **Target file**: `src/train_model/train_promptsmiles.py`, `src/train_model/run_promptsmiles.sh`
- **決定内容**:
  - **1分子1系列のまま、表記だけランダム化する**（データ量は増やさない = RFFMG・SAFE と同条件）。
    論文は10倍だが、倍率10の根拠は論文に示されておらず、2倍・5倍とのアブレーションも行われていない。
    根拠があるのは「ランダム化すること」であって「10倍であること」ではない。
  - **データセット構築時に1回だけ変換し、以降は固定**（毎エポックの引き直しはしない）。
    本プロジェクトは `eval_loss` で EarlyStopping と best model 選択を行うため、
    表記が毎回変わると評価が不安定になる。論文の10倍augmentationも事前生成の固定データセットであり、
    毎エポック引き直しではない。
  - train / validation の**両方**をランダム化する（同一分布にして `eval_loss` を意味のある指標に保つ）。
  - `random.Random(seed)` で再現性を担保する。
  - **オプションではなく常時実行**とする（使われない引数を残さない方針に合わせる）。
    アブレーションが必要になった時点でフラグ化すればよい。
- **Changes**: `validate_smiles` にランダム化を戻す
  （`rootedAtAtom=rng.randrange(mol.GetNumAtoms())`, `canonical=False`）。
  `import random` と `rng` を復活させ、`run_promptsmiles.sh` のコメントも実態に合わせる。
- **Dependencies**: after Step 18

## 未確定事項

なし（Step 2 の結果により解消済み）。
