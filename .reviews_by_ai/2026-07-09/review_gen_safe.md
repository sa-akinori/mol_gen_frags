# Checklist Review: gen_safe

- **Date**: 2026-07-09
- **Checklist source**: check_list.txt
- **Target**: `src/gen_safe.py`（SAFE-GPT 直接生成(direct)版）。対比参照: `src/func/generation_safe_func.py`, `src/gen_frags/safe_frags.py`, `src/func/fragment_for_safe.py`, `src/func/utility.py`
- **Generated**: 2026-07-09
- **Status**: pending-approval

## Summary

| Result | Count |
|--------|-------|
| PASS | 2 |
| PARTIAL | 0 |
| FAIL | 1 |
| N/A | 0 |

Low confidence items: C003 内の機序 C・E（LOW）。修正計画の Step 2 が該当。

## Items

### [C001] 学習済みモデルを読み込んで生成ができている

- **Result**: PASS
- **Confidence**: HIGH
- **Checked location**: `src/gen_safe.py:100-105, 121-123, 139-146`
- **Rationale**: `SAFEDoubleHeadsModel.from_pretrained` + `SAFETokenizer.from_pretrained(...).get_pretrained()` でモデル/トークナイザを読み、device 転送・`eval()`・test split をループして `model.generate` → `safe.decode`+RDKit → DataFrame → `predictions.csv`/`error_logs.csv` 出力まで機構として一貫。実測でトークナイザ挙動（[CLS]/[SEP]）も確認済みで、ロード＋生成の経路は実際に動く。
- **Evidence**: `model = SAFEDoubleHeadsModel.from_pretrained(model_path)` / `outputs = model.generate(**model_inputs, ...)` / `gen_df.to_csv(...)`
- **Notes**: 「生成が機構として成立」は「生成が良質」（=C003）とは別。C001 は機構の成立のみを問う。ロード対象 `models/safe/gpt/finetuning/brics/best_model` は現状ローカル未作成（run_safe.sh の学習成果物）。

### [C002] 分子生成コードに再現性がある

- **Result**: PASS
- **Confidence**: MEDIUM
- **Checked location**: `src/gen_safe.py:118, 55`, `src/func/utility.py:26-42`
- **Rationale**: 各行の生成前に `set_seed(random_seed)`（118行）。`set_seed` は PYTHONHASHSEED / CUBLAS_WORKSPACE_CONFIG / random・numpy・torch シード / cudnn.deterministic / use_deterministic_algorithms / transformers.set_seed を設定。さらに `do_sample=False`（55行）でビーム探索は決定的なので RNG 依存がそもそも無い。再現性は構造的に担保。
- **Evidence**: `set_seed(random_seed)` → `outputs = model.generate(..., do_sample=False, early_stopping=True)`
- **Notes（HIGH でない理由）**: 行ごとの `signal.alarm(timeout_sec=60)` は壁時計依存。負荷次第で同じ行が片方の実行で TimeoutError → `["time_out"]*n_samples` になり、run 間で出力が変わりうる非決定境界。正常完了時は完全決定的。→ タイムアウト発生の記録、または正典 run はタイムアウト無しを推奨。`to_csv` が既定 index を書く点は無害な雑音。

### [C003] 生成精度が非常に低い — その原因となりそうなコードのミスの探索

- **Result**: FAIL
- **Confidence**: MEDIUM
- **Checked location**: `src/gen_safe.py:47-68, 121-122`; 対比 `src/func/generation_safe_func.py:104-121`; `src/func/fragment_for_safe.py:27-76`; `src/gen_frags/safe_frags.py:40-48`
- **Rationale**: 低精度の主因は **条件付け方式の分布ミスマッチ**、増幅要因が複数。以下、機序ごとに確信度を付す。
  - **機序A（主因, MEDIUM）**: `pass_safe` を生の生成プレフィックスにして続きを生成（`_generate_valid_smiles(prefix=pass_safe)`）。しかし `pass_safe = convert2safe(p_frags, smiles)` と `full_safe = convert2safe(f_frags, smiles)` は独立に正規化され（`safe_frags.py:40-41`）、`convert2safe` はリング閉環番号を再割当（`fragment_for_safe.py:50-65`）。よって `pass_safe` は `full_safe` の接頭辞にならない（実測反例: full=`C34=O.c13ccc5cc1C…` / pass=`c13ccc5cc1C…`、先頭フラグ欠落）。学習系列にこの接頭辞は存在せず OOD → 続きが不安定で無効/誤生成。従来の `generation_safe_func.py` は `pass_fragments`(SMILES, `*`) を `SAFEDesign.scaffold_decoration/morphing` で条件付けし、接続点・閉環をライブラリ内で処理している。
  - **機序B（MEDIUM）**: プレフィックス内の開いた閉環番号（missing フラグメントへの接続）を、続きが正しい番号で閉じる必要があるが、番号対応が調整されていない → 未閉環残存 → `safe.decode` 失敗 → 'invalid' 多発。
  - **機序C（LOW）**: `early_stopping=True` かつ `min_length/min_new_tokens` 未指定で、プレフィックス直後に早期 EOS → 未完（開環残存）。A/B を増幅。実行確認が要るため LOW。
  - **機序D（MEDIUM）**: `num_return_sequences == num_beams == 50`（全ビーム返し）で候補がほぼ重複 → 実効多様性が激減、1つの悪いプレフィックスで 50 本全滅。
  - **機序E（LOW/MEDIUM）**: `max_length=200` は「プレフィックス込み総長」上限。長い scaffold で生成が途中打ち切り → 未閉環 → invalid。`max_new_tokens` が安全。
  - **機序F（HIGH, 指標の妥当性）**: `n_valid = sum(s not in ("invalid","time_out","error"))`（133行）は重複 SMILES も valid 計上。D と併せ、報告される validity が「精度/一意性」を反映しない。低精度の原因ではないが指標が不正確。
- **Evidence**: `gen_smiles = _generate_valid_smiles(model, tokenizer, pass_safe, ...)`（生の pass_safe をプレフィックス化）
- **バグ否定（前提事実に基づく）**: `v[:, :-1]` は `[SEP]`(EOS) のみ除去し `[CLS]`+scaffold は保持 → 正しい。`safe.decode(..., fix=True, remove_dummies=True, ignore_errors=True)` は誤り耐性処理であり、失敗を invalid に変換して露見させるだけで根本原因ではない。
- **Notes**: 根因は条件付け（A+B）。C/E/D が増幅、F は指標の注意点。実行して定量順位付けができないため全体 MEDIUM。

## Plan

（計画対象は FAIL の C003。全 Step の Target file は `src/gen_safe.py`。Step 1 が決定打、Step 2 は LOW 確信度のため要ユーザー確認。）

### Step 1: 条件付け方式の修正（根因 A+B）

- **Target file**: `src/gen_safe.py`（`_generate_valid_smiles` / `generate_from_model`）
- **Changes**: 生の `pass_safe` プレフィックス方式をやめ、次のいずれかをユーザーと決定して採用:
  - (a) `pass_fragments` を `SAFEDesign.scaffold_decoration`(単一) / `scaffold_morphing`(複数) で条件付け（`generation_safe_func.py:104-121` に倣う）。接続点・閉環番号をライブラリに委ねる。
  - (b) prefix-continuation を維持するなら、学習時と同一の `convert2safe(full)` 由来の「保持部分」から prefix を構築し、閉環番号を target と整合させる。
- **Dependencies**: none（主修正）
- **Related items**: C003

### Step 2: 生成長・終了条件の制約（機序 C, E｜LOW 確信度・要確認）

- **Target file**: `src/gen_safe.py`
- **Changes**: `max_length=200`(総長) を `max_new_tokens` に置換し prefix 長非依存の生成予算を確保。`min_new_tokens` 追加（または `early_stopping=True` 除去）で早期 EOS による未閉環を防ぐ。
- **Dependencies**: after Step 1
- **Related items**: C003（機序 C・E は LOW。ハードニング目的。含めるかユーザー確認）

### Step 3: 候補多様性の回復（機序 D）

- **Target file**: `src/gen_safe.py`
- **Changes**: `num_return_sequences` を `num_beams` から分離（top-k << beams）、または `do_sample=True`+temperature/top-p に切替（`set_seed` 済みで再現性維持）。n_samples 本を真に異なる試行にする。
- **Dependencies**: after Step 1（sampling 採用時は C002 決定性を再確認）
- **Related items**: C003

### Step 4: 精度指標の是正（機序 F）

- **Target file**: `src/gen_safe.py`
- **Changes**: `n_valid`（および報告精度）を **一意** valid canonical SMILES で算出。任意で target 回収フラグ（いずれかの予測が canonical で `target` と一致か）を追加。
- **Dependencies**: after Step 3
- **Related items**: C003

### Step 5（任意）: タイムアウト再現性（C002 の注意点）

- **Target file**: `src/gen_safe.py`
- **Changes**: 行ごとのタイムアウト発生をログ化、または正典 run はタイムアウト無しにして壁時計変動で結果が変わらないようにする。
- **Dependencies**: none
- **Related items**: C002

## Next Actions

### Approval pending

計画を確認し、「OK」「進めて」等で承認すると実装（implementer 委譲）を開始する。

### User review recommended (PARTIAL or low confidence)

- [C003] Step 2（機序 C・E）は LOW 確信度。実行して確認しないと支配的か不明なため、計画に含めるかを確認してください（決定打は Step 1）。
- [C003] Step 1 の方式 (a)/(b) はユーザー選択が必要。
- [C002] タイムアウトによる非決定境界の扱い（Step 5 任意）。

### Scope review

N/A は 0 件のため対象範囲は適切。
