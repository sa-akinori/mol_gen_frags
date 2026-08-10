# Plan: fix_evaluation_paths

- **Date**: 2026-08-06
- **Status**: completed (rev.2 — ユーザー指示により Step 1 / Step 4 を修正、旧 Step 5 を削除。2026-08-06 に Step 1-6 実装・検証済み)

## Overview

`src/evaluation.py` が組み立てるデータパス・結果パスが、生成側スクリプトの実際の出力先とずれており、
現状では `t5chem` / `gpt` / `safe_gpt` の評価がファイル未検出で失敗する。

本計画で、4 表現 5 モデル (`t5chem`, `gpt`, `safe_gpt`, `promptsmiles`, `fraggpt`) すべてが
同じ `evaluation.py` で評価できる状態にする。あわせて、手法ごとに異なる余分な列
(`prompt_safe` / `sampler`) を評価パイプラインに持ち込まないよう `loadGenSmiles` の戻り値を正規化する。

### 調査で確定した不整合

結果パス:

| model_name | 生成側の出力先 | evaluation.py が読む場所 | 判定 |
|---|---|---|---|
| t5chem / gpt | `results/rffmg/{m}/{ver}/{frag}/{N}times_sampling/{gen}/{add}/` | `results/rffmg/{m}/{ver}/{frag}/{gen}/{add}/` | NG: `{N}times_sampling` 欠落 |
| safe_gpt | `results/safe/gpt/{ver}/{frag}/beam/` | `results/safe/gpt/{ver}/{frag}/beam/normal/` | NG: 生成側に `normal` がない |
| promptsmiles | `results/promptsmiles/gpt/{ver}/{frag}/{gen}/normal/` | 同左 | OK |
| fraggpt | `results/fraggpt/gpt/{ver}/{frag}/{gen}/normal/` | 同左 | OK |

データパス:

| 用途 | 実在するパス | evaluation.py が読む場所 | 判定 |
|---|---|---|---|
| t5chem/gpt 学習分子 | `data/rffmg/{frag}/{N}times_sampling/normal/train.target` | `data/rffmg/{frag}/normal/train.target` | NG |
| t5chem/gpt テスト入力 | `data/rffmg/{frag}/{N}times_sampling/{add}/test.source` | `data/rffmg/{frag}/{add}/test.source` | NG |
| frag_order の id | `data/rffmg/{frag}/{N}times_sampling/frag_order/random_get_ids.pkl` | `data/rffmg/{frag}/frag_order/random_get_ids.pkl` | NG |
| promptsmiles / fraggpt | `data/{m}/{frag}/normal/` | 同左 | OK |

predictions.csv の列:

| 手法 | 列 | 評価に使う列 |
|---|---|---|
| t5chem / gpt / fraggpt | `target`, `prediction_1..N` | `fragment`(test.source 由来), `target`, `prediction_*` |
| safe_gpt | `(index)`, `target`, `prompt_safe`, `fragment`, `prediction_1..N` | 同上 (`prompt_safe` は不要) |
| promptsmiles | `target`, `sampler`, `prediction_1..N` | 同上 (`sampler` は不要) |

`evaluation_func()` は `fragment` / `target` / `all_smiles` しか参照しないため計算結果には影響しないが、
`sc3_check_genmol_results` が書き出す `curated_data.tsv` / `gen_samples_visualize.tsv` に余分な列が混ざる。

### 設計判断 (rev.2 でユーザーが決定)

- **safe_gpt は生成側に `normal` を追加する**。`evaluation.py` の
  `results/safe/gpt/{ver}/{frag}/{gen}/{additional_path}` という組み立ては他手法と揃っており正しいので変えず、
  `gen_safe.py` / `generation_safe_func.py` の出力先に `normal` を足して階層を揃える。
  既存の `results/safe/gpt/finetuning/brics/beam/predictions.csv` は不要 (SAFE は再生成する)。
- **promptsmiles / fraggpt の `additional_path` は固定しない**。渡された値をそのままパスに使う。
- `--sampling_num` の既定値は **5** (`data/` にも `results/` にも 5times_sampling が揃っているため)。
  `gen_rffmg.py` の既定値は 10 だが、評価側は 5 を既定とする。

## Plan

### Step 1: `--sampling_num` 引数の追加

- **Target file**: `src/evaluation.py`
- **Changes**: 引数定義ブロックに `--sampling_num` を追加する
  (`type=int, default=5, choices=[5, 10]`, help に `data/rffmg/<frag>/<N>times_sampling` を明記)。
  RFFMG (`t5chem` / `gpt`) 以外では使わない旨を help に書く。
- **Dependencies**: none

### Step 2: `--model_name` の既定値を help に合わせる

- **Target file**: `src/evaluation.py`
- **Changes**: `--model_name` に `default='t5chem'` を付ける。現状 help は `(default: t5chem)` と書いてあるが
  argparse に default がなく、省略時は `None` が最後の `else` 分岐に落ちて偶然 t5chem として動いている。
- **Dependencies**: none

### Step 3: RFFMG のデータ・結果パスに `{N}times_sampling` を挿入

- **Target file**: `src/evaluation.py`
- **Changes**:
  - `sampling = f'{args.sampling_num}times_sampling'` を用意する。
  - `t5chem` / `gpt` 分岐の `tr_file_name` を `data/rffmg/{frag_method}/{sampling}/normal/train.target` に、
    `testInputfile` を `data/rffmg/{frag_method}/{sampling}/{additional_path}/test.source` に変更。
  - `outfd` を手法ごとに組み立てる。RFFMG のときだけ `{frag_method}` と `{gen_method}` の間に
    `{sampling}` を挟む (`results/rffmg/{model_dir}/{model_ver}/{frag_method}/{sampling}/{gen_method}/{additional_path}`)。
    他の手法は現行どおり `results/{str_name}/{model_dir}/{model_ver}/{frag_method}/{gen_method}/{additional_path}`。
  - `frag_order` ブロック (現 `evaluation.py:102-107`) の `outfd` と `datafd` にも同じく `{sampling}` を挟む。
    `datafd` は `data/{str_name}/{frag_method}/` というハードコードだが、frag_order は RFFMG 専用条件なので
    `data/rffmg/{frag_method}/{sampling}/` に固定する。
- **Dependencies**: after Step 1

### Step 4: SAFE 生成の出力先に `normal` を追加

- **Target file**: `src/gen_mols/gen_safe.py`, `src/func/generation_safe_func.py`
- **Changes**: 両ファイルの `output_dir` (`gen_safe.py:24,28` と `generation_safe_func.py:106,110`) の末尾に
  `normal` を追加し、`results/safe/gpt/{model_ver}/{frag_method}/beam/normal/` に出力するようにする。
  `generation_safe_func.py` の冒頭 docstring の「出力」記述も新パスに更新する。
  これで `evaluation.py` 側は safe_gpt について変更不要になる (現行の `additional_path='normal'` 強制のまま一致する)。
  既存の `results/safe/gpt/finetuning/brics/beam/predictions.csv` は旧パスに残るが、不要なので触らない
  (SAFE は再生成する)。`generation_safe_func_old.py` は参照用の旧版なので変更しない。
- **Dependencies**: none

### Step 5: `loadGenSmiles` の戻り値を評価に使う列だけに正規化

- **Target file**: `src/func/evaluation_func.py`
- **Changes**: `loadGenSmiles` (現 `evaluation_func.py:56-73`) の末尾で、返す DataFrame を
  `['fragment', 'target'] + [prediction_* 列]` に絞る。これにより `safe_gpt` の `prompt_safe` と
  `promptsmiles` の `sampler` が下流に流れなくなり、全手法で同一列構成になる。
  docstring (Google style) を追加し、返す列名を明記する。
  `safe_gpt` 分岐の time_out/error 行フィルタは列を絞る前に実行する (フィルタが prediction 列を見るため)。
- **Dependencies**: none (Step 1-4 と独立)

### Step 6: パス解決の検証

- **Target file**: なし (検証のみ)
- **Changes**: 各 `model_name` について、`evaluation.py` が組み立てる `tr_file_name` /
  `testInputfile` / `outfd` が実在するファイルを指すことを確認する。
  重い評価本体は回さず、パス組み立て部分だけを切り出して `os.path.exists` で確認する。
  - `t5chem` / `gpt`: `--frag_method brics --sampling_num 5 --model_ver finetuning` で
    `results/rffmg/{m}/finetuning/brics/5times_sampling/beam/normal/predictions.csv` に解決されること。
  - `safe_gpt` / `promptsmiles` / `fraggpt`: 生成物が新パスに未作成のため、
    組み立てたパス文字列が生成側スクリプトの出力先と文字列として一致することを確認する。
- **Dependencies**: after Step 5

## Notes (実装対象外・報告のみ)

- Step 4 により SAFE の生成結果は新パスに出るため、SAFE の再生成が必要。
- `results/train_physic_property.csv` は最初の評価実行時にキャッシュされ、以降どの `model_name` /
  `frag_method` でも再利用される。学習分子集合は 4 手法で共通なので手法間では問題ないが、
  `brics` と `rc_cms` をまたいでも再利用される点は留意が必要。今回は変更しない。
- `promptsmiles` の `default_gen_method` は `sampling` だが `gen_promptsmiles.sh` の既定は `beam`。
  `--gen_method` で明示すれば済むため今回は変更しない。
