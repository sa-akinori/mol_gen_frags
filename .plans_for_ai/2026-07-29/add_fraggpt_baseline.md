# Plan: FragGPT をベースラインモデルとして追加

- **Date**: 2026-07-29
- **Status**: pending-approval

## Overview

FragGPT (Yue et al., *Chem. Sci.* 2024, DOI: 10.1039/d4sc03744h) をベースラインとして追加し、
本研究の比較対象（RFFMG / SAFE / PromptSMILES）に並べる。

### 調査の結論（設計の根拠）

論文本文には推論時のプロンプト仕様が書かれていないため、本家実装
(`https://github.com/pengbingxin/FragGPT-Interface`) を取得して確定させた。

| 項目 | 本家実装の実際 |
|---|---|
| 表現 | BRICS 断片 + 対になった `[i*]` ラベル（FU-SMILES） |
| 区切り | `<sep>`（本家独自 tokenizer の1トークン） |
| 学習 | FU-SMILES 上の**無条件言語モデル**。条件付けなし |
| 拡張 | ①番号 1..n のランダム置換 ②断片順のシャッフル |
| 推論プロンプト | `sep.join(与断片) + sep`（末尾の区切りが「次の断片を書け」の合図） |
| 番号の付与 | 未対の `*` に 1..k を新規付与、既存の対ラベルは +k シフト |
| 組立 | `combine_all_fragmens`: `[k*]` 同士を照合して結合、未消化なら `fail` |
| サンプリング | top-p 0.96 / temperature 1.0 |

### 本プロジェクトでの設計方針（ユーザー決定事項）

| 論点 | 決定 | 理由 |
|---|---|---|
| ベースモデル | `entropy/gpt2_zinc_87m`（RFFMG-GPT と同一） | ユーザー指示 |
| 区切り文字 | **`.`** | vocab 2707 を変更せず RFFMG-GPT と完全に同一のモデルを保てる。`<sep>` は `resize_token_embeddings` が必要 |
| 推論入力 | **SAFE の test split の `pass_fragments`** | SAFE・PromptSMILES と同一列。brics/rc_cms とも結合点が保たれている |
| 結合点ラベル | **open のみ**（与断片の `*` に別々の新規番号） | 本家の手順をそのまま適用した結果と一致。paired 版は実装しない |
| デコード | **beam search**（num_beams=50 / n_samples=50） | RFFMG・SAFE と同条件 |
| frag_method | **brics・rc_cms 両方** | 既存手法と同じマトリクス |
| mode | **finetuning・from_scratch 両方** | 既存手法と同じマトリクス |
| ハイパラ | LR 1e-4 / 50ep / batch 32 / warmup 10000 / eval・save 5000 / patience 15 / seed 42 / max_length 256 | 既存3手法と統一 |
| conda 環境 | **`env_fraggpt`（新規）** | 手法ごとに requirements + 専用 env という既存の慣習に合わせる。また学習は `t5chem`（`datasets` なし）・生成は `safe`（`datasets` あり）と環境が分裂するのを避け、学習から生成まで1環境で完結させる |

### 実測にもとづく事実（実装の前提）

1. **tokenizer**: `entropy/gpt2_zinc_87m` は `[1*]`〜`[18*]` を含む FU-SMILES を
   往復一致率 100% / unk 0 で扱える。トークン長は最大 236 < `max_length=256`。
2. **区切り文字 `.` の安全性**: brics 断片 1,239,697 個を検査し、内部に `.` を持つ断片は 0 件。
3. **学習コーパスは既存**: `data/rffmg/{frag}/5times_sampling/full_dataset.csv` の
   **`full_fragments` 列がそのまま FU-SMILES**（`BRICSFragmentize` が `dummyLabels=[(i+1,i+1)]` を付与）。
   フラグメント生成スクリプトの新規作成は不要。
4. **組立の検証**: `[i*]` 照合による組立を実装して検証した。
   **ダミー原子が持っていた結合次数を再現する必要がある**（単結合固定だと二重結合の切断箇所で壊れる）。

   | | 立体無視 | 立体込み |
   |---|---|---|
   | brics | 100.0% | 86.1% |
   | rc_cms | 100.0% | 88.9% |

   立体が落ちるのは二重結合を切った箇所の E/Z が復元できないため。SAFE と同種の制約
   （`src/gen_frags/safe_frags.py` にも同じ断り書きがある）。
5. **行対応**: SAFE の test split は RFFMG 5times の test と行数・target とも完全一致。

   | | train | val | test | 固有分子 |
   |---|---|---|---|---|
   | brics | 1,714,298 | 45,203 | 82,441 | 20,000 |
   | rc_cms | 6,648,549 | 175,488 | 90,974 | 20,000 |

   → FragGPT・SAFE・PromptSMILES・RFFMG(5times) を**同一行で直接比較**できる。
   `sampling_num` の軸は不要なので、データ配置は SAFE・PromptSMILES に揃えて
   `data/fraggpt/{frag}/normal/` とする。

### 学習データ量の比較（1分子あたりの系列数）

| モデル | 系列/分子 | train 系列数(brics) |
|---|---|---|
| RFFMG | 5 or 10 | — |
| SAFE | 5.51 | 1,714,298 |
| **FragGPT（本計画）** | **brics 1.0 / rc_cms 3.75** | **1,714,298**（SAFE と同数） |
| PromptSMILES | 1 | 1,717,900 |

`full_fragments` を `drop_duplicates` した件数は SAFE の `full_safe` の件数と一致するため、
FragGPT と SAFE は**同一のデータ量**で学習される。

### top-k accuracy にのみ効く FragGPT の性質

**生成そのものは正常に行われる。** 断片を与えれば分子が生成され、生成分子は与断片をすべて含む。
validity / uniqueness / novelty / フラグメント包含率にはいずれも影響しない。
影響を受けるのは**正解分子との完全一致（top-k accuracy）だけ**である。

理由は次のとおり。FU-SMILES は「同じ番号が2回現れたら結合」という書き方なので、
与断片同士を直接結合させるには `[1*][2*]`（ダミー2原子のみ）という断片が必要になるが、
これは BRICS 分解では決して生じないため学習分布に存在しない。
つまり FragGPT は与断片の間に必ず生成断片を1つ以上挟む。

brics の test 行のうち **82.5% は正解分子の中で与断片同士が直結している**（断片2個以上の行に限れば 84.7%）。
それらの行では、生成された50個の分子のいずれも正解分子と一致しない。
したがって **top-k accuracy の上限が 17.5% になる**。

これは実装の都合ではなく手法自体の性質なので、そのまま評価結果として報告する（ユーザー決定）。
必要なら直結を含まない部分集合での再集計は後から追加できる。

比較のための実測（既存の SAFE 生成結果 `results/safe_gpt/trained/safe/brics/beam/normal`）:

| | 完全一致率 |
|---|---|
| 与断片同士が直結している行 | 0.16%（29 / 18,101） |
| 直結していない行 | 1.08%（10 / 927） |

SAFE も同じ状況で約7倍のペナルティを受けるが、厳密にゼロではない。
一方 FragGPT は該当行で厳密に 0 になる。RFFMG は分子を直接 SMILES として生成するため
この制約を受けない。

### 命名規則（監査のうえ確定）

既存ファイルを監査し、以下の規則に揃える。

| 階層 | 規則 | 既存の例 | FragGPT |
|---|---|---|---|
| `src/train_model/train_*.py` | **手法名のみ** | `train_promptsmiles.py`（Step 0 で改名） | `train_fraggpt.py` |
| `src/train_model/run_*.sh` | 手法名のみ | `run_rffmg.sh` / `run_safe.sh` / `run_promptsmiles.sh` | `run_fraggpt.sh` |
| `src/gen_mols/gen_*.py` / `.sh` | 手法名のみ | `gen_rffmg.py` / `gen_safe.py` / `gen_promptsmiles.py` | `gen_fraggpt.py` |
| `src/func/generation_*_func.py` | 手法名 + `_func` | `generation_rffmg_func.py` / `generation_safe_func.py` | `generation_fraggpt_func.py` |
| `src/func/fragment_for_*.py` | 手法固有の表現変換 | `fragment_for_safe.py` | `fragment_for_fraggpt.py` |
| `data/{手法}/{frag}/normal/` | 手法名ディレクトリ | `data/safe/...` / `data/promptsmiles/...` | `data/fraggpt/...` |
| 学習テキスト | `.smi` | `data/promptsmiles/.../train.smi` | `train.smi` / `val.smi` |
| 分子 | `.target` | `data/rffmg/.../train.target` / `test.target` | `train.target` / `test.target` |
| 与断片 | `.source` | `data/rffmg/.../test.source` | `test.source` |

`src/train_model/train_gpt.py`（RFFMG-GPT）は改名しない。
`run_rffmg.sh` が `t5chem` と `gpt` の2バックエンドを切り替える構造で、
このファイルは「RFFMG の GPT バックエンド」を指すため、アーキテクチャ名で正しい。

## Plan

### Step 0: 既存 PromptSMILES 学習スクリプトの改名（命名規則の是正）

- **Target file**: `src/train_model/train_promptsmiles_gpt.py` → `src/train_model/train_promptsmiles.py`（改名）
- **Changes**: `gen_promptsmiles.py` / `run_promptsmiles.sh` が手法名のみなのに学習スクリプトだけ
  `_gpt` が付いており不整合なので是正する。
  - ファイルを `git mv` で改名する（履歴を保つ）
  - 参照元 `src/train_model/run_promptsmiles.sh:26` の
    `python src/train_model/train_promptsmiles_gpt.py \` を新しい名前に更新する
  - ファイルの中身（コード）は一切変更しない
  - 参照は上記1箇所のみであることを確認済み（`__pycache__` を除く）
  - 過去の計画ファイル `.plans_for_ai/2026-07-28/add_promptsmiles_baseline.{md,html}` は
    当時の記録なので**変更しない**
- **Dependencies**: なし

### Step 1: FU-SMILES ユーティリティの新規作成

- **Target file**: `src/func/fragment_for_fraggpt.py`（新規）
  （命名は `fragment_for_safe.py`（SAFE 表現への変換）に対応させる）
- **Changes**: FU-SMILES の分解・番号操作・組立をまとめたモジュールを作る。
  - `split_fragments(fusmiles: str) -> list[str]`: `.` 区切りで断片に分解
  - `assemble_fragments(fragments: list[str]) -> str | None`:
    `[i*]` が2箇所に現れる番号を照合して結合し、単一成分・ダミーなしの SMILES を返す。
    失敗（ダミー残存 / 複数成分 / sanitize 失敗 / パース失敗）は `None`。
    **ダミー原子が持っていた結合次数を再現すること**（単結合固定にしない）。
  - `renumber_open_attachments(fragments: list[str], rng: random.Random) -> list[str]`:
    素の `*` に 1..k の新規番号を付与する（推論プロンプト用）。本家の手順に対応。
  - `augment_fusmiles(fragments: list[str], rng: random.Random) -> list[str]`:
    番号 1..n のランダム置換 + 断片順のシャッフル。**系列数は増やさない**。
  - すべて型ヒント + Google style docstring。RDKit の `MolFromSmiles` の None チェックを省略しない。
- **Dependencies**: none

### Step 2: make_datasets.py へ FragGPT 用コーパス出力を統合

- **Target file**: `src/make_datasets.py`
- **Changes**: 既存の `if args.sampling_num == 5:` ガード内（SAFE・PromptSMILES と同じ場所）に追加する。
  - `rffmg_tr` / `rffmg_val` の `full_fragments` を `drop_duplicates` し、1行1系列で
    `data/fraggpt/{frag_method}/normal/{train,val}.smi` に出力
    （FU-SMILES は `.` 区切りなので文字列としては妥当な SMILES。
    学習テキストを `.smi` で出す PromptSMILES の慣習に合わせる）
  - 評価の novelty 判定用に、学習分子の SMILES を
    `data/fraggpt/{frag_method}/normal/train.target` に出力
    （`.target` を分子とする RFFMG・`test.target` の慣習に合わせる。
    内容は PromptSMILES の `train.smi` と同一だが、モジュール間依存を作らないため独立に書き出す）
  - 既存に倣い `debug/`（先頭10000行）も出力
  - 件数を標準出力に記録
  - **test は出力しない**（推論時に SAFE の test split を読むため）
  - 新規の分割ロジックは追加しない（既存の `tr_smiles`/`val_smiles` と `rffmg_tr`/`rffmg_val` の再利用のみ）
- **Dependencies**: none（Step 1 と並行可）

### Step 3: 学習スクリプト

- **Target file**: `src/train_model/train_fraggpt.py`（新規）
  （命名は `gen_fraggpt.py` / `run_fraggpt.sh` と同じく**手法名のみ**。Step 0 で揃えた規則に従う）
- **Changes**: `train_promptsmiles.py` の構成・コードスタイルを踏襲した**無条件言語モデル**の学習。
  - Dataset: `<bos> frag1.frag2....fragN <eos>`、labels は全トークン（`-100` マスクなし）。
    `max_length` 超過は切り詰めずに `ValueError` を送出（既存2スクリプトと同じ方針）
  - 拡張: `--augment`（default: **True**）。有効時は `augment_fusmiles` を
    **データセット構築時に1回**適用する（seed 固定で再現可能、系列数は不変）。
    エポックごとの再ランダム化は行わない（再現性を優先）。
    default を True にする理由: 番号と断片順が任意でよいことを学習していないと、
    推論時のプロンプト（新規番号 1..k・シャッフル順）に対応できず手法として成立しないため。
  - `--frag_method {brics, rc_cms}` / `--mode {finetuning, from_scratch}` / `--pretrain entropy/gpt2_zinc_87m`
  - ハイパラ default は既存3手法に一致（epoch 50 / LR 1e-4 / batch 32 / warmup 10000 /
    eval・save 5000 / save_total_limit 5 / EarlyStopping 15 / seed 42 / max_length 256）
  - 出力: `models/fraggpt/gpt/{mode}/{frag_method}/`、`best_model/` を保存
  - 型ヒント・Google style docstring 必須
- **Dependencies**: after Step 1, Step 2

### Step 4: 学習実行シェルスクリプト

- **Target file**: `src/train_model/run_fraggpt.sh`（新規）
- **Changes**: `run_promptsmiles.sh` / `run_rffmg.sh` の作法を踏襲。
  リポジトリルートへ cd、`conda activate env_fraggpt`、`CUDA_VISIBLE_DEVICES=0`、
  `WANDB_MODE=offline` と `WANDB_DIR="wandb/fraggpt/gpt/${MODE}/${FRAG_NAME}"` を設定して
  `train_fraggpt.py` を実行。`FRAG_NAME` / `MODE` を上部変数で切替。
- **Dependencies**: after Step 3

### Step 5: 生成本体

- **Target file**: `src/func/generation_fraggpt_func.py`（新規）
- **Changes**: 学習済み GPT2 をロードして FU-SMILES を beam search し、組立て分子にする。
  - 入力: `datasets.load_from_disk(f'{BASEPATH}/data/safe/{frag_method}/normal')` の test split。
    正解分子 = `smiles` 列、プロンプトの元 = `pass_fragments` 列
    （`gen_promptsmiles.py` の `read_test_split` と同一の読み方）
  - 各行: `renumber_open_attachments` で `*` に 1..k を付与 → 断片順をシード付きでシャッフル →
    プロンプト `<bos> frag1.frag2....fragk.`（末尾の `.` が「次の断片を書け」の合図）
  - `model.generate(do_sample=False, num_beams=50, num_return_sequences=50, ...)`。
    プロンプト以降を `.` 分割して生成断片とし、プロンプト断片と合わせて `assemble_fragments`
  - 組立に失敗した候補は空文字列にする。**行は絶対に落とさない**
    （RFFMG・SAFE・PromptSMILES と行対応を維持するため。失敗は validity に正直に反映される）
  - 出力:
    - `results/fraggpt/gpt/{model_ver}/{frag_method}/beam/normal/predictions.csv`
      （列 `target`, `prediction_1` .. `prediction_N`、T5Chem 形式）
    - `data/fraggpt/{frag_method}/normal/test.source`（与断片、素の `*` 形式）
    - `data/fraggpt/{frag_method}/normal/test.target`（正解分子）
  - 3ファイルの行数・行順が常に一致することを assert で担保
  - 失敗理由別の件数（パース失敗 / ダミー残存 / 複数成分 / sanitize 失敗）を集計し、
    標準出力と `LogFile` の両方へ出力する。**例外を黙って飲み込まないこと**
  - 型ヒント・Google style docstring 必須、`set_seed` で再現性を担保
- **Dependencies**: after Step 1, Step 3

### Step 6: 生成ラッパー

- **Target file**: `src/gen_mols/gen_fraggpt.py` + `src/gen_mols/gen_fraggpt.sh`（新規）
- **Changes**: `gen_rffmg.py` / `gen_rffmg.sh` の作法を踏襲。
  - `func.generation_time.run_and_record_time` で `generation_fraggpt_func.py` を subprocess 実行し、
    `generation_time.json` を出力（既存2手法と同じ計測方法に揃える）
  - 引数: `--frag_method {brics, rc_cms}`, `--model_ver {finetuning, from_scratch}`,
    `--n_samples 50`, `--num_beams 50`, `--batch_size`, `--max_length 256`, `--random_seed 42`
  - 出力先: `results/fraggpt/gpt/{model_ver}/{frag_method}/beam/normal/`
  - `.sh` は既存の `gen_*.sh` と異なり、**リポジトリルートへ cd して `conda activate env_fraggpt` を行う**
    （ユーザー決定）。生成は `datasets` に依存するため、環境を取り違えると `ModuleNotFoundError` になる。
    `run_*.sh` と同じ作法に揃えることで、どのシェルからでもそのまま実行できるようにする
- **Dependencies**: after Step 5, Step 9

### Step 7: 評価パイプラインへの統合

- **Target file**: `src/evaluation.py`
- **Changes**:
  - `--model_name` の choices に `fraggpt` を追加
  - 分岐を追加: `str_name, model_dir, arc_name, gen_method = 'fraggpt', 'gpt', 't5chem', 'beam'`
  - `tr_file_name = f'{BASEPATH}/data/fraggpt/{frag_method}/normal/train.target'`
  - `testInputfile = f'{BASEPATH}/data/fraggpt/{frag_method}/{additional_path}/test.source'`
  - `outfd` は既存の `results/{str_name}/{model_dir}/{model_ver}/{frag_method}/{gen_method}/{additional_path}`
    のパターンにそのまま乗る（`sampling_num` 階層を持たないため既存コードの変更は不要）
  - validity / uniqueness / novelty / フラグメント包含率を RFFMG・SAFE と**同一関数**で算出する
- **Dependencies**: after Step 6

### Step 8: 小規模検証

- **Target file**: 実行のみ（コード変更なし）
- **Changes**: `debug/` データで以下を確認する。
  - 学習が数ステップ回ること（brics × finetuning）
  - 生成が数百行通り、`predictions.csv` / `test.source` / `test.target` の行数が一致すること
  - 組立成功率と失敗理由の内訳が妥当であること
  - `evaluation.py --model_name fraggpt` が最後まで通ること
- **Dependencies**: after Step 7

### Step 9: FragGPT 専用実行環境の追加

- **Target file**: `requirements/fraggpt_requirements.txt`（新規）
- **背景**: 実装後の検証で、`t5chem` 環境に `datasets` が無いことが判明した。
  そのため当初想定の「`t5chem` 環境で完結」は成立せず、学習は `t5chem`・生成は `safe` という
  環境の分裂が発生していた。また既存の3手法はいずれも
  `requirements/{手法}_requirements.txt` + 専用 env を持っており、FragGPT だけ無い状態だった。
- **Changes**: `promptsmiles_requirements.txt` に倣いバージョンを固定して記述する。
  FragGPT は本家パッケージを使わず自前実装のため、手法固有のパッケージは含まれない。

  ```
  rdkit==2025.3.6
  transformers==4.45.2
  datasets==4.8.2
  pandas==3.0.1
  ```

  torch は既存2環境と同様 requirements 外で個別導入する。想定 conda 環境名は **`env_fraggpt`**。
- **Dependencies**: none

## 実行順序と実行時間の見積もり

- Step 0・1・2 は並行可（Step 0 は既存 PromptSMILES の改名のみで FragGPT 実装とは独立）。
  Step 3 以降は順に依存する。
- **学習の実行はユーザーが行う**。本計画のスコープは実装 + 小規模検証まで。
- 学習は 2 frag_method × 2 mode = **4本**。
  1エポックあたりの step 数は brics 約 53,600 / rc_cms 約 207,800（batch 32）。
  `num_train_epochs=50` に到達する前に EarlyStopping（検証15回連続 = 75,000 steps 改善なし）で
  停止する想定。GPU は既存手法と同じ `CUDA_VISIBLE_DEVICES=0`。

## スコープ外（今回やらないこと）

- **de novo 生成（MOSES ベンチマーク）**: 本研究の比較軸は断片条件付き生成のため含めない。
- **paired labeling**（与断片の結合情報を与える版）: ユーザー決定により実装しない。
- **ADMET 条件付き生成 / 強化学習**: FragGPT 論文の付加機能であり、比較軸に含まれない。
- **`<sep>` 区切り**: `.` を採用したため実装しない。

## 既存コードの既知の不整合（今回は修正しない・要確認事項）

FragGPT の実装とは独立に、調査中に見つかった既存の問題を記録する。
いずれも本計画では触れないが、別途対応するか判断が必要。

1. **`src/evaluation.py` の RFFMG パスが古い**:
   `data/rffmg/{frag}/normal/train.target` と `data/rffmg/{frag}/{additional_path}/test.source`
   を参照しているが、実際のデータは `data/rffmg/{frag}/{N}times_sampling/normal/` 配下にあり
   **存在しないパス**になっている。結果パス側も `{N}times_sampling` 階層を持たないため
   `gen_rffmg.py` の出力先と一致しない。
2. **`src/gen_frags/safe_frags.py` のパスが古い**:
   `data/rffmg/{frag}/full_dataset.csv`（`{N}times_sampling` 階層なし）を読んでいる。
   既存の `safe_smiles.csv` は 5times 由来であることを行数一致で確認済み。
3. **SAFE と RFFMG の分子分割の不一致**:
   SAFE・PromptSMILES・FragGPT は 5times の分割に固定される一方、
   `run_rffmg.sh` は現在 `SAMPLING_NUM=10` のため、RFFMG-10times だけ別の分子分割になっている。
   RFFMG-5times とは行単位で一致する。
4. **rc_cms における入力情報量の非対称**:
   `trimRonRing=True` の影響で、RFFMG の `test.source` は環上の結合点が H に置換されて消えているが、
   SAFE の `pass_fragments` には残っている。したがって rc_cms では
   SAFE・PromptSMILES・FragGPT は RFFMG より多くの結合点を受け取る。
   これは既存の PromptSMILES ベースラインにも同じく当てはまる。

## 未確定事項

なし。
