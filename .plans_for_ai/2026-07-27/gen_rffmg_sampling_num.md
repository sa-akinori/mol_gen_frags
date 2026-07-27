# Plan: gen_rffmg を {5,10}times_sampling に対応させる

- **Date**: 2026-07-27
- **Status**: approved

## Overview

`src/gen_mols/gen_rffmg.py` のパスは sampling_num 導入前のままで、実データレイアウトとずれている。

| | 現在のコード | ディスク上の実態 |
|---|---|---|
| data | `data/rffmg/{frag}/{additional}` | `data/rffmg/{frag}/{N}times_sampling/{additional}` |

そのため `data/rffmg/rc_cms/normal` は存在せず、既定引数のままでは生成が走らない。
学習側（`src/train_model/run_rffmg.sh`, `src/train_model/train_gpt.py`）と
データ生成側（`src/make_datasets.py`, `src/gen_frags/rffmg_frags.py`）は既に
`--sampling_num` / `{N}times_sampling` に対応済みなので、**生成側だけが取り残されている**状態。
この既存の規約にそのまま合わせる。

### 既存の規約（踏襲するもの）

- CLI 引数名: `--sampling_num`（`int`）
- パス片: `f'{args.sampling_num}times_sampling'`
- 学習の出力先: `models/rffmg/{model_name}/{mode}/{frag}/{sampling}`（`run_rffmg.sh:19`, `train_gpt.py:159`）
- shell 変数: `SAMPLING_NUM=10 # 5 or 10 (uses the data/rffmg/<frag>/<N>times_sampling slice)`（`run_rffmg.sh:14`）

### 既定値と選択肢

`--sampling_num` は `type=int, choices=[5, 10], default=10` とする
（`train_gpt.py:107-108` と同じ。`run_rffmg.sh` の `SAMPLING_NUM=10` とも揃う）。

### 3つのパスすべてに sampling を入れる

`dataset_dir` だけを直すと、10times で学習したモデルに 5times のデータを
食わせるといった取り違えが起きるため、`model_path` と `output_dir` にも入れる。

| | 変更前 | 変更後 |
|---|---|---|
| `dataset_dir` | `data/rffmg/{frag}/{additional}` | `data/rffmg/{frag}/{sampling}/{additional}` |
| `model_path` | `models/rffmg/{model}/{ver}/{frag}/best_model` | `models/rffmg/{model}/{ver}/{frag}/{sampling}/best_model` |
| `output_dir` | `results/rffmg/{model}/{ver}/{frag}/{gen}/{additional}` | `results/rffmg/{model}/{ver}/{frag}/{sampling}/{gen}/{additional}` |

`model_path` / `output_dir` の階層位置は学習側（`{frag}/{sampling}`）に合わせる。

## Plan

### Step 1: `gen_rffmg.py` に `--sampling_num` を追加し、3つのパスに反映する

- **Target file**: `src/gen_mols/gen_rffmg.py`
- **Changes**:
  - Model parameters のグループに引数を追加（`--model_ver` の直後）:
    ```python
    parser.add_argument('--sampling_num', type=int, default=10, choices=[5, 10],
                        help='Fragment-sampling multiplier N, selecting the data/rffmg/<frag>/<N>times_sampling slice (default: 10)')
    ```
  - 既存の変数取り出しブロックに `sampling = f'{args.sampling_num}times_sampling'` を追加。
  - `model_path` / `output_dir` / `dataset_dir` を上表の「変更後」に差し替える。
  - `run_and_record_time` の `params` に `"sampling_num": args.sampling_num` を追加し、
    生成時間の記録からどのスライスを使ったか分かるようにする。
- **Dependencies**: none

### Step 2: `gen_rffmg.sh` に `SAMPLING_NUM` を追加する

- **Target file**: `src/gen_mols/gen_rffmg.sh`
- **Changes**:
  - 他の設定変数と同じ形式で1行追加（コメント文言は `run_rffmg.sh:14` と揃える）:
    ```bash
    SAMPLING_NUM=10 # 5 or 10 (uses the data/rffmg/<frag>/<N>times_sampling slice)
    ```
  - `python` 呼び出しに `--sampling_num ${SAMPLING_NUM}` を追加する。
    引数の並びは他と同様、モデル系（`--model_name` … `--model_ver`）の後・
    生成系（`--additional_path` 以降）の前に置く。
- **Dependencies**: after Step 1

### Step 3: 動作確認

- **Target file**: 変更なし（確認のみ）
- **Changes**:
  - `python -m py_compile src/gen_mols/gen_rffmg.py`
  - `python src/gen_mols/gen_rffmg.py --help` で `--sampling_num` が `{5,10}` 付きで出ること。
  - `bash -n src/gen_mols/gen_rffmg.sh` で shell 構文チェック。
  - 5 / 10 それぞれで組み立てられる `dataset_dir` が実在することを確認する
    （`--frag_method brics` は 5・10 とも `normal` が存在。
    `rc_cms` は 5 のみ存在＝下記 Notes 参照）。
  - GPU 生成そのものは重いので走らせない。
- **Dependencies**: after Step 2

## Notes / 判断が必要な点

1. **既存の学習済みモデルは新しい `model_path` では見つからない**
   - ディスク上の実態: `models/rffmg/{t5chem,gpt}/{finetuning,from_scratch}/{brics,rc_cms}/best_model`
     （sampling の階層なし＝sampling_num 導入前に学習したもの）
   - 変更後のコードが探す場所: `.../{frag}/{sampling}/best_model`
   - 現在の `run_rffmg.sh` は `models/rffmg/{model}/{mode}/{frag}/{sampling}` に出力するので、
     **今後学習するモデルは新しい場所に入る**。既存モデルで生成したい場合は
     手動で移動（例: `mv .../brics/best_model .../brics/10times_sampling/`）が必要。
   - 見つからない場合は `FileNotFoundError` で明示的に落ちるため、黙って誤ったモデルを
     読むことはない。
2. **`rc_cms` の 10times_sampling はまだ空**
   - `data/rffmg/rc_cms/10times_sampling/` には `full_dataset.csv` のみで、
     `normal/` 等の分割データがない（`make_datasets.py --frag_method rc_cms --sampling_num 10` が未実行）。
   - `brics` は 5・10 とも揃っている。既定の `--sampling_num 10` で `rc_cms` を
     生成しようとすると t5chem 側がデータを見つけられず失敗する。
3. **既存の結果は上書きされない**
   - 変更後は `results/rffmg/{model}/{ver}/{frag}/{sampling}/{gen}/{additional}/` に出力されるため、
     既存の `results/rffmg/t5chem/finetuning/{frag}/beam/{additional}/predictions.csv` はそのまま残る。
     ただし旧パスを見ている下流スクリプトは新しい結果を拾わない。
4. **スコープ外**: `src/evaluation.py:46-47` も
   `data/rffmg/{frag_method}/{additional_path}` と sampling なしのパスを組み立てており、
   同じずれを抱えている。今回の指示は `gen_rffmg.py` / `gen_rffmg.sh` のみなので触らない。
