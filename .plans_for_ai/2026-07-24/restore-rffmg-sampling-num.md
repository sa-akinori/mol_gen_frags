# Plan: rffmg_frags.py の sampling_num 引数化と保存先変更の復元

- **Date**: 2026-07-24
- **Status**: pending-approval

## Overview

`src/gen_frags/rffmg_frags.py` が旧版に巻き戻り、以下2点の修正が失われている:

1. `nSamplingTrialsPerFragset` がハードコード (`=5`) に戻っている
2. 出力先が `data/rffmg/{frag}/full_dataset.csv` で、`{N}times_sampling` 階層が無い

一方、下流の `src/train_model/run_rffmg.sh`（`SAMPLING_NUM=10` → `data/rffmg/${FRAG}/${N}times_sampling/normal` を参照）および `train_gpt.py`（`--sampling_num`）は既に新レイアウト前提。
実データ `data/rffmg/{brics,rc_cms}/{5,10}times_sampling/full_dataset.csv` も新レイアウトで存在する。
git 全履歴・全 worktree・dangling オブジェクトを探索したが復元版は残っておらず、手で書き直す必要がある。

ユーザー決定: 引数名は `--sampling_num`、**default=5**（明示指定しない限り従来挙動）。

## Plan

### Step 1: 関数 `sc1_make_sentences_for_training` に sampling 引数を追加

- **Target file**: `src/gen_frags/rffmg_frags.py`
- **Changes**:
  - シグネチャに `nSamplingTrialsPerFragset: int = 5` を追加（型ヒント必須）。
  - `Smi2SentenceOpt(...)` 内の `nSamplingTrialsPerFragset=5`（44行目）を、引数 `nSamplingTrialsPerFragset` を渡す形に変更。
  - Google style docstring を追加/更新（引数・戻り値・返す DataFrame のカラム名 `sentence`, `full_fragments`, `pass_fragments` を明記）。
- **Dependencies**: none

### Step 2: `__main__` に `--sampling_num` 引数を追加し関数へ伝播

- **Target file**: `src/gen_frags/rffmg_frags.py`
- **Changes**:
  - `parser.add_argument('--sampling_num', type=int, default=5, help='number of sampling trials per fragment set (data/rffmg/<frag>/<N>times_sampling)')` を追加。
  - `frags_df = sc1_make_sentences_for_training(fd, smilesPath, frag_method, nSamplingTrialsPerFragset=args.sampling_num, debug=False)` として伝播。
- **Dependencies**: after Step 1

### Step 3: 保存先を `{N}times_sampling` 階層に変更

- **Target file**: `src/gen_frags/rffmg_frags.py`
- **Changes**:
  - 86–87行目を以下に変更:
    ```python
    out_dir = f'{fd}/rffmg/{frag_method}/{args.sampling_num}times_sampling'
    os.makedirs(out_dir, exist_ok=True)
    frags_df.to_csv(f'{out_dir}/full_dataset.csv')
    ```
  - これにより `run_rffmg.sh` / `train_gpt.py` が参照するパスと一致する。
- **Dependencies**: after Step 2

### Step 4: 動作確認（read-only）

- **Target file**: なし（確認のみ）
- **Changes**:
  - `python -c` で argparse とパス組み立てのみを検証（`--sampling_num` 未指定→5, `=10`→`10times_sampling`）。
  - 重い分子生成本体は実行しない（既存データを上書きしないため）。
- **Dependencies**: after Step 3

## Notes / 未確定事項

- ログ出力先 `outfd = f'{fd}/t5chem'`（17行目）は今回のスコープ外（変更しない）。実データの `data/rffmg/sentences_logs.txt` は旧世代の産物と判断。必要ならユーザー指示で別途対応。
- worktree はユーザー方針（no-worktrees-unless-asked）によりメインで直接作業。
