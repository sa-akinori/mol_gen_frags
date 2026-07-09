# Checklist Review: gen_safe（最終版 rev3）

- **Date**: 2026-07-08
- **Checklist source**: check_list.txt
- **Target**: `src/gen_safe.py`（＋生成本体 `src/func/generation_safe_func.py`）
- **Generated**: 2026-07-08（rev3: 会話を踏まえた最終結論を反映）
- **Status**: concluded（実装は行わない／ユーザー判断で保留）

## 最終結論（TL;DR）

**分子生成コードの核心ロジックは「間違っていない」。** 低い有効SMILES率の主因はコードのバグではなく、SAFEライブラリ＋モデル側の限界。

- ✅ **実バグは1点のみ**: `generation_safe_func.py:100` の `kwargs` を生成呼び出しに渡していない（+`num_beams=args.n_samples` の誤用、`num_beam_groups=0.0` float）。ユーザーも自認済み。**効くのは「多様性」で、無効SMILESの主因ではない**。
- ✅ **2段階生成（morphing/decoration → 残った `*` を埋める）は意図的で妥当な設計**。`scaffold_morphing` が複数フラグメントで `*` を残すのはライブラリ内部の挙動で、それを2段目で埋める発想は正しい。**バグではない**。
- ⚠️ **`safe_invalid` / None / タイムアウトの多さ = 手法・モデル側の限界**（荷電・複数アタッチメント分子で SAFE 生成/デコードが失敗しやすい、生成が遅い）。コードの誤りではない。
- 🩹 `MACHINE_ID` 未定義・旧称パス（our_slice/pretrained）はコミット版が古いだけで、**実行環境では是正済み**のため実害なし。
- 🔁 **撤回（実測で否定）**: 「list への numpy 配列 indexing でクラッシュ」／「`add_dot` が原因」。いずれも誤り。断定を急いで2回外した点はお詫び。

## 調査で確認した事実（実データ・実行）

### 実データ統計
出力 `mol_gen_frags_oxygen/results/safe_gpt/trained/safe/brics/beam/normal/predictions_*.csv`（25,000分子×50スロット=125万）:

| 値 | 割合 |
|----|------|
| `safe_invalid` | 38.6% |
| `time_out` | 25.5% |
| `Can't generate`（旧版コード産物） | 20.2% |
| 有効SMILES様 | 15.6% |
| `error` | ~0% |

先頭500分子: 平均ユニーク有効 5.2／分子、**40.8%（204/500）が有効ゼロ**。
※ `Can't generate` は現行コード・safeライブラリに無い文字列 → この結果は旧版コードの産物。

### 実行診断（brics trained model、ラベル前の生出力を分類）
- `scaffold_morphing`/`scaffold_decoration` が未充填 `[*:N]` を残す分子を頻出（`*` は1個とは限らず複数あり得る）。
- 2段目 `scaffold_decoration(how='greedy')` の失敗機構:
  - **1個の `*`**: モデルが greedy 生成中に自分で `.` と別フラグメントを付加 → 複数フラグメント → `safe_invalid`。
    - 例: `Nc1ccc(S(=O)(=O)c2nc(N[*:4])no2)cc1` → `CCNc1noc(...)n1.CC[C@@H](C)OC`
  - **2〜3個の `*`**: 軒並み None（この小テストでは全滅）。
- **`add_dot=True/False` は出力が完全に同一** → 余分フラグメントは `add_dot` の末尾ドットではなく、モデル自身の greedy 生成が原因（＝add_dot 仮説は否定）。
- 単フラグメント decoration で `_decode_safe(ignore_errors=True)` が None を返す（デコード不能SAFE）ケース頻出（一例で20/18が None）。

## Items（チェック項目ごとの最終判定）

### [C001] 学習済みモデルを読み込んで生成ができている。
- **Result**: PASS（実運用）／コミット版は要整備
- **Confidence**: HIGH
- **判定理由**: 実行環境では `trained/` の正しいモデル・データで生成できている（ユーザー確認）。コミット版のみ旧称パス・`MACHINE_ID` 未定義で未整備。症状の原因ではない。

### [C002] 分子生成コードに再現性がある。
- **Result**: PASS
- **Confidence**: MEDIUM
- **判定理由**: `set_seed` が random/numpy/torch(cuda)/cudnn/`use_deterministic_algorithms` を網羅設定。各分子ループで seed 設定、`do_sample=False` の決定的デコード。堅牢。

### [C003] 生成精度が低い原因（コードのミスか）
- **Result**: コード起因は限定的（実バグは kwargs の1点のみ）
- **Confidence**: HIGH
- **判定理由**: 低有効率の主因は SAFEライブラリ/モデルの生成・デコード失敗とタイムアウト（手法側の限界）。2段階充填の設計自体は妥当。当初挙げた indexing・add_dot は実測で否定。

## 直すなら（任意・強制しない）

ユーザー方針により「無理に直さない」。参考として、コード起因で明確に直せるのは以下のみ。

- **kwargs 伝播の修正**（多様性向け）: `generation_safe_func.py:100,103-119`。`how`/`num_beams=args.num_beams`/`do_sample` を実際に渡す。`num_beam_groups` は int。
- （任意）コミットファイル整備: パス（pretrained→trained, our_slice→rc_cms, dataset に `/normal/`）、`MACHINE_ID`→`args.machine_id`。
- （任意・ユーザー自己修正予定）ハードコード: `range(25000)`（76行）、`n_samples_per_trial=50`（105,115行）、`signal.alarm(60)`（95行）。

有効率そのものの改善は手法・モデル側の課題で、コード修正だけでは限界がある（実測で確認済み）。

## 備考
- このレビューは read-only 調査。src/ への変更は行っていない。
- 実行に用いた診断スクリプトは一時領域（`/tmp/`）に作成、リポジトリは未変更。
