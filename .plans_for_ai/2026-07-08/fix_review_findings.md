# Plan: レビュー指摘（再構成に伴う破損・不整合）の一括修正

- **Date**: 2026-07-08
- **Status**: pending-approval

## Overview

モデルディレクトリ再構成（`models/{repr}/{model}/{mode}/{slice}`）に伴い、旧パスを参照する生成・評価スクリプトと
README が破損/不整合になっている。レビューで挙がった全項目を修正する。

### 旧→新パスの対応（正本）

| 対象 | 旧 | 新 |
|------|----|----|
| safe モデル | `models/safe_gpt/{ver}/safe/{slice}/best_model` | `models/safe/gpt/{ver}/{slice}/best_model` |
| safe pretrained | `models/safe_gpt/pretrained/` | `models/safe/gpt/pretrained/` |
| t5chem モデル | `models/t5chem/{ver}/rffmg/{slice}/best_model` | `models/rffmg/t5chem/{ver}/{slice}/best_model` |
| t5chem pretrained | `models/t5chem/pretrained/` | `models/rffmg/t5chem/pretrained/` |
| results（safe） | `results/safe_gpt/{ver}/safe/{slice}/...` | `results/safe/gpt/{ver}/{slice}/...` |
| results（t5chem） | `results/t5chem/{ver}/rffmg/{slice}/...` | `results/rffmg/t5chem/{ver}/{slice}/...` |
| mode 語彙 | `trained` | `finetuning`（＋ `from_scratch` を選択肢に追加、`pretrained` は据え置き） |

**方針判断**:
- results パスも models と並行して新レイアウトへ移行（元コードは models/results が同一構造で対になっていたため、対を維持）。
- `--model_ver` の選択肢を `trained`→`finetuning` に置換し、`from_scratch` を追加。
- pretrained は best_model サブディレクトリを持たないため、pretrained 分岐は `.../pretrained/` を直接指す。

## Plan

### Step 1: gen_safe_denovo.py のパス修正
- **Target**: `src/gen_safe_denovo.py`（`__main__`, 171-172行）
- **Changes**: `model_path = fd/'models'/'safe'/'gpt'/'pretrained'`、
  `output_dir = fd/'results'/'safe'/'gpt'/'pretrained'/'denovo'` に修正。

### Step 2: gen_safe.py のパス修正
- **Target**: `src/gen_safe.py`（`__main__`）
- **Changes**: `model_ver='trained'`→`'finetuning'`、
  `model_path=f'{fd}/models/safe/gpt/{model_ver}/{slice_name}/best_model'`、
  `output_dir=f'{fd}/results/safe/gpt/{model_ver}/{slice_name}/{gen_method}/'`。

### Step 3: gen_mols/gen_safe.py のパス + 選択肢修正
- **Target**: `src/gen_mols/gen_safe.py`
- **Changes**: `--model_ver` choices を `['finetuning','from_scratch','pretrained']`（既定 `finetuning`）に。
  分岐を pretrained と（finetuning/from_scratch）に整理し、
  model_path/output_dir を新レイアウト `models/safe/gpt/...`, `results/safe/gpt/...` に。

### Step 4: gen_mols/gen_t5chem.py のパス + 選択肢修正
- **Target**: `src/gen_mols/gen_t5chem.py`
- **Changes**: `--model_ver` choices を `['finetuning','from_scratch','pretrained']`（既定 `finetuning`）に。
  `model_path=f'{BASEPATH}/models/rffmg/t5chem/{model_name}/{frag_method}/best_model/'`、
  `output_dir=f'{BASEPATH}/results/rffmg/t5chem/{model_name}/{frag_method}/{gen_method}/{additional_path}'`。

### Step 5: gen_t5chem.py（dummy スクリプト）のパス修正
- **Target**: `src/gen_t5chem.py`（12-13行）
- **Changes**: `models/t5chem/trained/dummy/...`→`models/rffmg/t5chem/finetuning/dummy/...`、
  `results/t5chem/trained/dummy/...`→`results/rffmg/t5chem/finetuning/dummy/...`。
  ※ dummy/scratch スクリプト。使用状況不明のため整合目的の最小修正のみ。

### Step 6: evaluation.py の選択肢 + パス修正
- **Target**: `src/evaluation.py`
- **Changes**: `--model_ver` choices の `trained`→`finetuning`（既定も）。
  ファイル内で `models/{...}` や `results/{...}` を旧レイアウトで組み立てている箇所があれば新レイアウトへ更新
  （実装時に全体を確認して整合）。

### Step 7: README.md / README_ja.md の更新
- **Target**: `README.md`, `README_ja.md`
- **Changes**:
  - モデル配置手順: `models/t5chem/pretrained`→`models/rffmg/t5chem/pretrained`、
    `models/safe_gpt/pretrained`→`models/safe/gpt/pretrained`（mkdir/wget/tar/git clone のパス）。
  - 学習例: `--output_dir models/t5chem/trained/rffmg/rc_cms --pretrain models/t5chem/pretrained` を
    新レイアウト（`models/rffmg/t5chem/finetuning/rc_cms`, `--pretrain models/rffmg/t5chem/pretrained`）に。
    SAFE 学習セクションのパスも同様に更新。
  - wandb セクション: 「run_rffmg.sh が `export WANDB_PROJECT` で指定」の記述を実態に合わせ、
    「スクリプトは WANDB_PROJECT を設定せず、既定 `T5Chem` にフォールバック（setdefault により任意で上書き可）」へ修正。

## Notes / 確認事項

- results/ の旧データ（`results/safe_gpt/*`, `results/t5chem/*`）はそのまま残る（新規実行は新パスへ）。
- site-packages（t5chem run_trainer.py）編集の再インストール消失については、README の該当箇所に注意書きを追記するか要相談（本計画では任意）。
- HuggingFace からの `models/*` 一括ダウンロード手順は著者リポジトリの構造依存のため本計画では変更しない。
