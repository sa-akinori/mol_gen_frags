# Checklist Review: train_gpt

- **Date**: 2026-07-09
- **Checklist source**: check_list.txt
- **Target**: `src/train_model/train_gpt.py`（GPT2 で RFFMG source→target を学習）。比較基準: `src/train_model/run_safe.sh`（safe-gpt HP）, `src/train_model/run_rffmg.sh`（起動元/WANDB_DIR）, safe-train/t5chem パッケージ, `src/func/utility.py`
- **Generated**: 2026-07-09
- **Status**: pending-approval

## Summary

| Result | Count |
|--------|-------|
| PASS | 3 |
| PARTIAL | 3 |
| FAIL | 0 |
| N/A | 0 |

Low confidence items: 0（全判定は grep/コード確認済みで HIGH。未確定は「方針判断」であり検出確信度ではない）

## Items

### [C001] from_scratch と fine-tuning が適切に分かれ、fine-tuning 時に事前学習モデルが適切に呼ばれる

- **Result**: PASS
- **Confidence**: HIGH
- **Checked location**: `train_gpt.py:98-101, 137, 143-149`
- **Rationale**: `mode` は `choices=["finetuning","from_scratch"]` で分岐。finetuning=`GPT2LMHeadModel.from_pretrained(args.pretrain)` で事前学習重みをロード、from_scratch=`GPT2Config.from_pretrained(args.pretrain)` で同一 config を取りつつ `GPT2LMHeadModel(config)` でランダム初期化。tokenizer は両モード共通で ZINC からロードし config/tokenizer が一致。
- **Evidence**: `if args.mode == "finetuning": model = GPT2LMHeadModel.from_pretrained(args.pretrain)` / `else: config = GPT2Config.from_pretrained(args.pretrain); model = GPT2LMHeadModel(config)`
- **Notes**: run_rffmg.sh は `--pretrain` 未指定で既定 `entropy/gpt2_zinc_87m` を使用（Hub モデル初期値）。safe-gpt がローカル `models/safe/gpt/pretrained` を使うのと設計が異なるが意図通り。

### [C002] 学習率などのパラメータが safe-gpt と一致（max_length のみ異なってよい）

- **Result**: PARTIAL
- **Confidence**: HIGH
- **Checked location**: `train_gpt.py:106-121, 156-171` vs `run_safe.sh:40-51`
- **Rationale**: 学習に影響する主要 HP はすべて一致（num_train_epochs=50 / learning_rate=1e-4 / warmup_steps=10000 / per_device_train_batch_size=32 / eval_strategy=steps / eval_steps=5000 / save_strategy=steps / save_steps=5000 / save_total_limit=5 / load_best_model_at_end=True）。weight_decay・lr_scheduler_type・adam・fp16 等も双方 transformers 既定で一致。**差分1点**: train_gpt.py は `per_device_eval_batch_size=32` を明示、safe 側は未指定で既定 **8**。ただし評価スループットのみに影響し学習結果（重み更新）には無影響。
- **Evidence**: `per_device_eval_batch_size=args.per_device_train_batch_size,  # =32 ; safe側は既定8`
- **実トレース確認（改変後 safe パッケージ現物）**: safe-train は `cli.py:408-409` の `HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))`+`parse_args_into_dataclasses()` で HP 解決。`per_device_eval_batch_size` を train と同値にする代入は cli.py に無し（grep ゼロ）。run_safe.sh の `--config models/safe/gpt/pretrained/config.json` は **モデルアーキテクチャ config**（`ModelArguments.config`→`AutoConfig.from_pretrained` cli.py:246）で**学習HP config ではない**。→ safe 実効 eval batch=**8** で確定（「config で調整」ではない）。
- **Notes**: 「max_length 以外は完全一致」を厳密に満たすなら eval batch を未指定（既定8）に。実運用は許容範囲、ユーザー判断。

### [C003] "A>>B" のうち "A>>" を入力、"B" のみ出力し損失計算・学習（プロンプトマスク）

- **Result**: PASS
- **Confidence**: HIGH
- **Checked location**: `train_gpt.py:53-58`（RFFMGDataset）
- **Rationale**: `input_ids = [bos] + prompt_ids + target_ids + [eos]`、`labels = [-100]*(1+len(prompt_ids)) + target_ids + [eos]`。両者同一長で位置一致。HF GPT2LMHeadModel は内部で labels をシフトするため、最初の target トークン loss は bos+prompt を見た位置から予測されオフバイワン無し。プロンプト（`<bos> source ">>"`）は -100 マスクされ、loss は target と最終 eos のみ。`">>"` は `source+">>"` としてまとめてトークン化され生成時運用と自己整合。
- **Evidence**: `prompt_ids = tokenizer(source + ">>", add_special_tokens=False)["input_ids"]` / `labels = ([-100]*(1+len(prompt_ids)) + target_ids + [eos_id])[:max_length]`
- **Notes（縁ケース）**: `[:max_length]` の末尾切り詰めで `1+len(prompt) >= max_length` の超過例は labels 全 -100（学習信号ゼロ・eos 欠落）になり得る。max_length=256 にデータ長が収まれば実害なし。分布不明なら要確認。主要件は満たすため PASS。

### [C004] 再現性が担保されている

- **Result**: PARTIAL
- **Confidence**: HIGH
- **Checked location**: `train_gpt.py:20, 134, 169` vs `src/func/utility.py:26-42`
- **Rationale**: `transformers.set_seed(args.seed)` + `TrainingArguments(seed=args.seed)` で乱数固定・データシャッフル決定性は確保。ただしプロジェクト標準 `func.utility.set_seed`（cudnn.deterministic / use_deterministic_algorithms / CUBLAS_WORKSPACE_CONFIG / PYTHONHASHSEED を設定）ではなく `transformers.set_seed`（random/numpy/torch/cuda のみ、deterministic=False 既定）のため、**GPU 演算のビット単位決定性は未保証**。CLAUDE.md「乱数シード明示」は満たすが標準関数の完全決定化には未達。
- **Evidence**: `from transformers import ..., set_seed` / `set_seed(args.seed)`（cudnn 等未設定）
- **Notes**: `func.utility.set_seed` を使う場合 import パス解決が必要。`use_deterministic_algorithms(True)` は一部演算で例外/速度低下の可能性。**【ユーザー承認済み】** → `func.utility.set_seed` 化を実施（Step 2 確定）。

### [C005] 学習過程が wandb の適切なフォルダに保存（モデルも保存）

- **Result**: PASS
- **Confidence**: HIGH
- **Checked location**: `train_gpt.py:131, 170, 182-184` vs `run_rffmg.sh:23-25`
- **Rationale**: `report_to=["wandb"]` + `WANDB_MODE` を `setdefault("offline")`。run_rffmg.sh が `export WANDB_DIR="wandb/rffmg/${MODEL_NAME}/${MODE}/${FRAG_NAME}"` と `mkdir -p`・`WANDB_MODE=offline` を行うため、想定経路では repr/model/mode/slice 別の適切なフォルダに保存。モデルは `trainer.save_model(output_dir/best_model)` + `tokenizer.save_pretrained` で重み/config/tokenizer が揃い、`save_steps` の途中 checkpoint も保存。
- **Evidence**: `os.environ.setdefault("WANDB_MODE", "offline")` / `report_to=["wandb"]` / `trainer.save_model(best_model_dir); tokenizer.save_pretrained(best_model_dir)`
- **Notes（要確認）**: train_gpt.py 自身は WANDB_DIR を設定しないため、run_rffmg.sh を経由せず直接実行すると `./wandb` に書かれ分離されない。想定運用では問題なし。自己完結性を上げるなら train_gpt 内で WANDB_DIR 導出も可。

### [C006] early_stopping の設定が入っている（詳細は safe-gpt や t5chem と同じ）

- **Result**: PARTIAL
- **Confidence**: HIGH
- **Checked location**: `train_gpt.py:173-180`（callbacks 未指定）／ safe `trainer/cli.py:340`／ t5chem `run_trainer.py:280`
- **Rationale**: grep 事実:
  - **train_gpt.py**: `Trainer(...)` に `callbacks` 無し → EarlyStoppingCallback **無し**（`load_best_model_at_end=True` のみ）。
  - **safe-gpt（最も近い基準）**: `SAFETrainer` 生成に callbacks 無し、`safe/trainer/` に early stopping 実装 **無し**（sample.py の early_stopping は生成ビームサーチ用で学習無関係）。safe も `load_best_model_at_end` のみ。
  - **t5chem**: `run_trainer.py:280` に `callbacks=[EarlyStoppingCallback(early_stopping_patience=15)] if do_eval else []`、`metric_for_best_model="eval_loss"` を明示 → early stopping **有り**。
  → train_gpt.py の「early stopping 無し」は **safe-gpt とは一致、t5chem とは不一致**。基準が割れている。チェックリスト文言「入っている」を文字通り取ると欠落。
  - **改変後 safe パッケージ現物での再確認**: ユーザーが safe を直接改変済みのため `safe/` **全体**を `EarlyStopping|early_stopping_patience|callbacks=|add_callback` で再 grep → **No matches**。後付けの early stopping も無いことを確定。
- **Evidence**: t5chem: `callbacks=[EarlyStoppingCallback(early_stopping_patience=15)] if do_eval else []` ／ safe `SAFETrainer(...)` に callbacks 引数なし ／ train_gpt.py `Trainer(model=..., args=..., train_dataset=..., eval_dataset=..., data_collator=...)`（callbacks なし）
- **Notes（最重要・要確認）**: 「safe-gpt に合わせ early stopping 不要」か「t5chem に合わせ patience=15 を導入」か方針判断。導入時は `EarlyStoppingCallback` 追加＋`metric_for_best_model="eval_loss"` 明示が必要。

## Plan

（計画対象は PARTIAL の C002/C004/C006。多くが方針判断を伴うため承認前提。全 Step の対象は `src/train_model/train_gpt.py`。）

### Step 1: early stopping 方針の確定と Callback 追加（採用時）

- **Target file**: `src/train_model/train_gpt.py`
- **Changes**: t5chem 準拠を選ぶ場合 (a) `EarlyStoppingCallback` を import、(b) `Trainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=15)])`、(c) `TrainingArguments(..., metric_for_best_model="eval_loss")` を明示。safe-gpt 準拠（無し）を選ぶ場合は現状維持で C006 解消扱い。patience の引数化も検討。
- **Dependencies**: なし（方針決定が前提）
- **Related items**: C006

### Step 2: 再現性の強化（プロジェクト標準 set_seed）

- **Target file**: `src/train_model/train_gpt.py`（import パス調整含む）
- **Changes**: `transformers.set_seed` を `func.utility.set_seed(args.seed)` に置換し cudnn.deterministic / use_deterministic_algorithms / CUBLAS_WORKSPACE_CONFIG / PYTHONHASHSEED を設定。`TrainingArguments(seed=...)` は維持。`use_deterministic_algorithms(True)` の例外/速度影響があるため完全決定化を要求するか要確認。
- **Dependencies**: なし
- **Related items**: C004

### Step 3: eval batch size の safe-gpt 整合（任意）

- **Target file**: `src/train_model/train_gpt.py`
- **Changes**: 「max_length 以外は完全一致」を厳密に満たすなら `per_device_eval_batch_size` の明示を外し safe 既定(8)に合わせる。学習結果に無影響のため現状維持でも可。
- **Dependencies**: なし
- **Related items**: C002

### Step 4: truncation 縁ケースのロバスト化（任意）

- **Target file**: `src/train_model/train_gpt.py`（RFFMGDataset）
- **Changes**: `1 + len(prompt_ids) >= max_length` で target 全マスク/eos 欠落になる超過例をフィルタ（スキップ+ログ）、または target 優先の切り詰めに変更。max_length=256 にデータ長が収まるなら不要。
- **Dependencies**: なし
- **Related items**: C003（PASS だが縁ケース補強のため任意）

## Next Actions

### Approval pending

計画を確認し「OK／進めて」で承認すると implementer に委譲して実装する。承認までコードは変更しない。

### User review recommended（方針判断）

- [C006] early stopping: **safe-gpt 準拠（無し）** か **t5chem 準拠（patience=15）** か。← 最重要
- [C004] 再現性: `func.utility.set_seed` に切替え完全決定化するか（速度/例外リスクとのトレードオフ）。
- [C002] eval batch を safe 既定(8)に厳密一致させるか（学習結果に無影響）。
- [C003] truncation 縁ケースを補強するか（データ長が 256 内なら不要）。

### Scope review

N/A は 0 件。対象範囲は適切。
