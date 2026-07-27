# Plan: gen_rffmg の `pretrained` 選択肢を削除する

- **Date**: 2026-07-27
- **Status**: approved（ユーザー指示「pretrainedはないので気にしないでください。configに書かれている場合は消しといて」による）

## Overview

`src/gen_mols/gen_rffmg.py` の `--model_ver` は `pretrained` を選べるが、
選ぶと必ず存在しないパスを組み立てるため、実質的に死んだ選択肢になっている。

```
組み立てるパス: models/rffmg/{model_name}/pretrained/{frag}/{sampling}/best_model
実物          : models/rffmg/t5chem/pretrained/{config.json, pytorch_model.bin, ...}  ← フラット
```

事前学習モデルは断片化手法にもサンプリング数にも依存しないためフラット構成が正しく、
学習側 `run_rffmg.sh:31` も `--pretrain models/rffmg/{model}/pretrained` とフラットに参照している。
RFFMG の生成に事前学習モデルを使う予定はないため、選択肢自体を削除する。

## スコープ外（触らないもの）

- `src/gen_mols/gen_safe.py` / `gen_safe.sh`: SAFE の `pretrained` は実在し、
  `if model_ver == 'pretrained':` の分岐で正しく扱われている（`results/safe_gpt/pretrained/` に実績あり）。
- `src/evaluation.py:16`: `--model_name` が `t5chem` / `safe_gpt` / `gpt` を共有しており、
  `safe_gpt` の `pretrained` 評価に必要。ここから消すと SAFE 側が壊れる。
- `src/train_model/run_rffmg.sh:31`: finetuning の初期値として事前学習モデルを指す正当な参照。

## Plan

### Step 1: `gen_rffmg.py` の choices から `pretrained` を外す

- **Target file**: `src/gen_mols/gen_rffmg.py`
- **Changes**: L17-18 の `--model_ver` を
  `choices=['finetuning', 'from_scratch', 'pretrained']` から
  `choices=['finetuning', 'from_scratch']` に変更する。
  `default='finetuning'` と help 文言（`Model version (default: finetuning)`）はそのまま。
- **Dependencies**: none

### Step 2: `gen_rffmg.sh` のコメントから `pretrained` を外す

- **Target file**: `src/gen_mols/gen_rffmg.sh`
- **Changes**: L6 のコメントを
  `MODEL_VER="finetuning" # "finetuning", "from_scratch", "pretrained"` から
  `MODEL_VER="finetuning" # "finetuning", "from_scratch"` に変更する。
  コメントの書式（値の直後に半角1つ + `#`、ダブルクォート付きの列挙）は維持する。
- **Dependencies**: after Step 1

### Step 3: 動作確認

- **Target file**: 変更なし（確認のみ）
- **Changes**:
  - `python -m py_compile src/gen_mols/gen_rffmg.py`
  - `python src/gen_mols/gen_rffmg.py --help` で `--model_ver` が `{finetuning,from_scratch}` になること。
  - `python src/gen_mols/gen_rffmg.py --model_ver pretrained` が argparse のエラーで
    即座に弾かれること（従来は実行が進んでから FileNotFoundError になっていた）。
  - `bash -n src/gen_mols/gen_rffmg.sh`
  - `gen_safe.py --help` の `--model_ver` に `pretrained` が残っていること（巻き添え変更がないことの確認）。
- **Dependencies**: after Step 2
