# Next Plan

- **Updated**: 2026-07-30
- **Status**: pending-user-verification

## 最優先: promptsmiles / fraggpt データの生成と検証

`src/make_datasets.py` の promptsmiles/fraggpt 出力ブロックは 2026-07-30 に復元済み（コミット `f5a04f8`）。
ただし **`data/promptsmiles/` と `data/fraggpt/` はまだ生成されていない**。生成はユーザー側で実施する。

### 実行コマンド

```bash
python src/make_datasets.py --frag_method brics  --sampling_num 5
python src/make_datasets.py --frag_method rc_cms --sampling_num 5
```

`sampling_num` が既定値(5)のときだけ SAFE / PromptSMILES / FragGPT が生成される
(`make_datasets.py` の `if args.sampling_num == 5:` ブロック)。10times では生成されない。

### 実行前に確認すること

実行すると **既存の `data/rffmg/{frag}/5times_sampling/normal/*` と `data/safe/{frag}/normal` も上書きされる**。
分割は `random.seed(0)` で決定的であり、2026-07-30 の検証で現ディスク上のデータが
`full_dataset.csv` からの再現結果と完全一致することを確認済みなので内容は変わらないが、
2026-07-30 21:06 に転送されたばかりのファイルを書き直すことになる点は認識しておく。

### 生成後に検証すること（期待値は 2026-07-30 の検証で確定した分割サイズ）

| ファイル | brics | rc_cms |
|---|---|---|
| `data/promptsmiles/{frag}/normal/train.smi` | 1,717,908 行 | 1,772,900 行 |
| `data/promptsmiles/{frag}/normal/val.smi` | 45,208 行 | 46,655 行 |
| `data/promptsmiles/{frag}/normal/test.smi` | 20,000 行 | 20,000 行 |
| `data/fraggpt/{frag}/normal/train.target` | 1,717,908 行 | 1,772,900 行 |
| `data/rffmg/{frag}/5times_sampling/normal/test.target` のユニーク分子 | 20,000 | 20,000 |
| `data/safe/{frag}/normal` test split のユニーク分子 | 20,000 | 20,000 |

- `data/fraggpt/{frag}/normal/{train,val}.smi` は `full_fragments` の重複排除後なので行数は事前確定できない。
- `data/fraggpt/{frag}/normal/test.*` は**意図的に生成されない**。`gen_fraggpt.py` は SAFE test split を
  プロンプト源にする設計のため（`generation_fraggpt_func.py:96`）。
- 生成後、RFFMG test と SAFE test の分子集合が一致すること（2026-07-30 時点では一致を確認済み）を再確認する。

### 学習・生成の前提

`models/promptsmiles/` と `models/fraggpt/` は未作成。データ生成 → 学習 → 生成 → 評価の順。

- 学習: `src/train_model/run_promptsmiles.sh`, `run_fraggpt.sh`
- 生成: `src/gen_mols/gen_promptsmiles.sh`, `gen_fraggpt.sh`
- 評価: `python src/evaluation.py --model_name promptsmiles`（既定 `--gen_method sampling`）
  / `--model_name fraggpt`（既定 `beam`）

## 保留中の項目（今日は着手しないと判断したもの）

### evaluation.py 系のパスに sampling_num 階層がない

`evaluation.py:66-67, 99` は `data/rffmg/{frag_method}/normal/train.target` を読むが、実データは
`data/rffmg/{frag}/{N}times_sampling/normal/` 配下にある。同形の問題が以下にもある。

- `src/analyze_predictions.py:280-281`
- `src/figure.py:344-347`
- `src/check_reproducibility.py`

ユーザー判断で保留中。RFFMG の評価を回すときに必ず踏むので、そのタイミングで対応する。

### dummy(旧 rffmg) を参照する残置コード

- `src/figure.py:304` → `data/dummy/{slice}/{const}/target_frags.pkl`
- `src/check_reproducibility.py:20` → `data/dummy/{method}/full_dataset.csv`

`data/dummy/` は存在しない（`rffmg` へ改名された際の取り残し）。ユーザー指示により当面残置。
実行すれば `FileNotFoundError` になる。

### results/ の命名混在

`results/js_divergence/physic_properties/{frag}/beam/normal/` と
`results/physic_properties/{beam,rc_cms}/` が同階層に異種の名前を持つ。計 108KB、
該当コード（`evaluation.py` の js-divergence 部分）はコメントアウト済みのため当面 OK と判断。

## 10times_sampling について（対応不要と確認済み）

`sampling_num` ごとに `full_dataset.csv` の `unique()` 順が変わるため、10times の分割は
5times（および SAFE/PromptSMILES/FragGPT）と別物になる。実測値:

| | RFFMG test ∩ SAFE test | SAFE test ∩ RFFMG train |
|---|---|---|
| 5times | 20,000（完全一致） | 0 |
| brics/10times | 243 | 18,969 |
| rc_cms/10times | 295 | 18,790 |

10times は比較目的ではなく別の理由で作成しているため問題なし、とユーザーが判断済み。
ベースライン比較表を作る際は 5times を使う。

## 参考: 2026-07-30 に完了した作業

- `data/rffmg` の test 分割が旧コード版だった問題 → 転送により解消（test = 20,000分子、≤5行/分子）
- mtime 保持転送で失われた `make_datasets.py` / `evaluation.py` の promptsmiles/fraggpt 接続部分を復元
- 未コミットだった PromptSMILES/FragGPT 実装 1,644行をコミット（`084c1a4`）
- `.gitignore` に `__pycache__/` を追加、追跡済み .pyc 16個を追跡解除（`285b732`）
- 旧レイアウトの残骸を削除（約 37GB 回収）: `data/` の旧 debug と `t5chem/`、
  `models/` の sampling階層なし重複6系統と `t5chem/`、`wandb/` の同6系統、`results/safe_gpt/`
- `src/gen_t5chem.py` を削除（`gen_rffmg` に集約済み）、`make_datasets.py` の未使用変数 `add_path` を削除

データ転送は mtime を保持するため、更新検出には `find -newerct`（ctime）を使う。
mtime は生成元マシンの時刻なので、同一ディレクトリ内で矛盾していても異常とは断定できない。
世代の判定は必ず内容で行う（分割の再現、行数・ユニーク分子数、集合の包含関係）。
