# Plan: 事前学習の学習曲線（SAFE-GPT / RFFMG-T5）

- **Date**: 2026-07-08
- **Status**: approved

## Overview

SAFE-GPT と RFFMG-T5 の学習過程を可視化するため、**loss（縦軸）vs steps（横軸）**の学習曲線を描く。

重要な前提（調査で判明）:
- ダウンロードした「pretrained」モデル（Zenodo / datamol-io）には学習ログが無い。ログが残るのは**ユーザーが実行した学習run**（wandb オフライン）のみ。したがって本タスクの「事前学習の学習曲線」= これら学習runの曲線として描く。
- 両モデルとも wandb に `train/loss` / `eval/loss` / `train/global_step` を記録済み。**eval行にも `train/global_step` が入る**ため、train・eval とも横軸を step に揃えられる。
- 損失データは wandb バイナリ（`run-*.wandb`）から `wandb.sdk.internal.datastore` で抽出できることを検証済み。wandb は conda 環境 `safe`（0.25.1）/ `t5chem`（0.25.0）の両方に存在。
- run名の旧称対応: `dummy`=`rffmg`、`our_slice`=`rc_cms`。history item のキーは `item.key or '/'.join(item.nested_key)` で取得（nested_key形式）。

### ユーザー確定事項
- フラグメント手法: **rc_cms + brics 両方**
- T5の種別: **trained + from_scratch 両方**（※SAFEには from_scratch run が無く trained のみ）
- 図の構成: **モデルごとに別ファイル**
- 損失: **train + eval loss 両方**を1枚に重ねる

### 出力（合計6枚、`figures/learning_curves/` 配下、各PNGに train_loss・eval_loss の2本）
- `rffmg_t5_trained_brics.png`
- `rffmg_t5_trained_rc_cms.png`
- `rffmg_t5_from_scratch_brics.png`
- `rffmg_t5_from_scratch_rc_cms.png`
- `safe_gpt_brics.png`
- `safe_gpt_rc_cms.png`

### 採用run（同一構成に複数runあり→ step数最大＝最も学習が進んだものを採用）
| 出力名 | wandbディレクトリ | max step |
|--------|-------------------|----------|
| rffmg_t5_trained_brics | `wandb/offline-run-20250907_114610-c54mbj4m` | 3,315,000 |
| rffmg_t5_trained_rc_cms | `wandb/offline-run-20250903_155930-1bb137m9` | 3,130,000 |
| rffmg_t5_from_scratch_brics | `wandb/offline-run-20250925_104927-gf6oysoc` | 2,005,000 |
| rffmg_t5_from_scratch_rc_cms | `wandb/offline-run-20250925_104947-sa7dxu6r` | 2,375,000 |
| safe_gpt_brics | `wandb/offline-run-20250920_111843-lxs32jiy` | 715,000 |
| safe_gpt_rc_cms | `wandb/offline-run-20250913_185038-i0hyt8qz` | 2,420,000 |

## Plan

### Step 1: wandb import を追加

- **Target file**: `src/figure.py`（先頭のサードパーティimport群、13行目 `from rdkit import Chem` 付近）
- **Changes**: 以下を追加。
  ```python
  from wandb.sdk.internal import datastore
  from wandb.proto import wandb_internal_pb2 as wandb_pb
  ```
- **Dependencies**: none

### Step 2: wandbからloss履歴を抽出する関数を追加

- **Target file**: `src/figure.py`（module末尾の helper 群、`frag_num_analyze`（79行目付近）の直後、`if __name__ == "__main__":` の直前）
- **Changes**: 関数 `read_wandb_loss_history` を追加。型ヒント・Google style docstring・返すDataFrameのカラム名を明記。
  - 引数 `run_path: str`（`run-*.wandb` の絶対パス）
  - `datastore.DataStore().open_for_scan(run_path)` で走査し、`WhichOneof('record_type')=='history'` のレコードを収集。
  - 各 history item のキーは `item.key or '/'.join(item.nested_key)`。値は `float(value_json)`。
  - `train/global_step` を step とし、`train/loss` / `eval/loss` を各行から拾って dict のリスト→DataFrameへ。
  - 返り値 DataFrame カラム: `step`（int）, `train_loss`（float, 無い行はNaN）, `eval_loss`（float, 無い行はNaN）。step昇順ソート。
- **Dependencies**: after Step 1

### Step 3: 学習曲線を描画・保存する関数を追加

- **Target file**: `src/figure.py`（Step 2 の関数の直後）
- **Changes**: 関数 `plot_learning_curve` を追加。型ヒント・Google style docstring。
  - 引数 `history: pd.DataFrame`, `save_path: str`, `title: str`
  - `train_loss` / `eval_loss` それぞれ `dropna()` した上で、`plt` で step（x）vs loss（y）を2本の折れ線として描画（train=実線, eval=マーカー付き等で区別、凡例つき）。
  - 軸ラベル `Steps` / `Loss`、`title` をタイトルに設定。`os.makedirs(os.path.dirname(save_path), exist_ok=True)` 後に `savefig`→`plt.close()`。
  - eval が空でも train のみで描けるようガード。
- **Dependencies**: after Step 2

### Step 4: 実行ブロックをファイル末尾に追加

- **Target file**: `src/figure.py`（`if __name__ == "__main__":` 内の**最終ブロック**として末尾に追加）
- **Changes**: 新規 `if 1:` ブロックを追加し、Overviewの「採用run」表の対応辞書（出力名→wandbディレクトリ相対パス）を定義。
  - `out_dir = f'{fd}/figures/learning_curves'`
  - 各runについて `run_file = glob(f'{fd}/{run_dir}/run-*.wandb')[0]` でファイル解決 → `read_wandb_loss_history` → `plot_learning_curve(hist, f'{out_dir}/{name}.png', title=name)`。
  - `fd` は既存 `__main__` 冒頭の `fd = os.path.dirname(os.path.dirname(__file__))` を流用。
- **Dependencies**: after Step 3

### Step 5: 既存アクティブブロックを無効化（単一ブロック運用の踏襲）

- **Target file**: `src/figure.py`（205行目 `if 1:`）
- **Changes**: 現在アクティブな 205行目の `if 1:` を `if 0:` に戻す（figure.py は「実行したいブロックのみ `if 1:`」という運用のため、新ブロックだけが動くようにする）。
- **Dependencies**: after Step 4

## Verification

1. 実行: `conda run -n safe python src/figure.py`
2. `ls -l figures/learning_curves/` で6枚のPNG生成を確認。
3. 各PNGを開き、横軸=Steps・縦軸=Loss で train/eval 2本が描かれ、step範囲が採用run表と整合していることを目視確認。
4. 任意: `read_wandb_loss_history` を1runで単体実行し、末尾 `eval_loss` が既知値（rc_cms SAFE ≈ 0.477、trained rc_cms T5 ≈ 0.149）に近いことを確認。

## Notes

- コードスタイル規約（`.claude/CLAUDE.md`）遵守: 型ヒント、Google style docstring、DataFrame返却関数はカラム名明記、import順（標準→サードパーティ→ローカル）。
