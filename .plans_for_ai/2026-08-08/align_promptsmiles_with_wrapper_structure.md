# Plan: PromptSMILES を他手法と同じラッパ構造に揃える

- **Date**: 2026-08-08
- **Status**: approved

## Overview

PromptSMILES だけが `generation_time.json` を出しておらず、生成速度を手法間で比較できない。
原因は構造の違いで、他3手法が「薄いラッパ + 生成本体」の2ファイル構成なのに対し、
PromptSMILES は `gen_mols/gen_promptsmiles.py`（380行）が生成本体そのものになっている。

| 手法 | ラッパ | 生成本体 |
|---|---|---|
| RFFMG | `gen_mols/gen_rffmg.py` | 外部 CLI `t5chem predict` / `train_model/generate_gpt.py` |
| SAFE | `gen_mols/gen_safe.py` | `func/generation_safe_func.py` |
| FragGPT | `gen_mols/gen_fraggpt.py` | `func/generation_fraggpt_func.py` |
| **PromptSMILES** | **なし** | `gen_mols/gen_promptsmiles.py`（本体が直接ここにある） |

ラッパは `func.generation_time.run_and_record_time` で生成本体をサブプロセスとして起動し、
**モデルロードを含む実行時間**を計測して `generation_time.json` を書く。PromptSMILES にも
同じ構造を与えることで、測定基準を揃えると同時にコードの不揃いも解消する。

### 計測基準の差（実測）

`generation_time.json` には python の起動と import の時間も含まれる。promptsmiles 環境で
`import torch, transformers, promptsmiles, pandas, datasets` にかかるのは **1.17秒**。
82,441行を16時間かける実行では `sec_per_molecule` に 0.00001秒しか効かないが、
**基準を揃える**ことを優先してラッパ構造にする（ユーザー判断）。

### `--additional_path` を追加する理由

ラッパは生成本体と**同じ `output_dir` を計算する必要がある**（JSON の書き出し先と
`predictions.csv` の行数カウントのため）。現在 PromptSMILES は `normal` を直書きしているので、
そのままだと2ファイルに同じ直書きが並ぶ。これは `gen_fraggpt.py` で実際に起きていた不具合
（出力先が食い違い、timing JSON が別 run を上書きする）と同じ形なので、
FragGPT と同様に `--additional_path` を引数にして両者が同じ値を使うようにする。

### 既存の生成結果への影響

`results/promptsmiles/` には `predictions.csv` がまだ無い（生成未実施）。
また `joblib` 未導入で生成スクリプト自体が起動しないため、**この変更で失われる結果はない**。

## スコープ外

- `requirements/promptsmiles_requirements.txt` への `joblib` 追加（ユーザーが別途対応）
- 生成ロジックの変更（移動のみ。**1行も書き換えない**）
- `--n_samples` を変えると結果が変わる件（修正に25倍のコストがかかるため対応しない）

## Plan

### Step 1: 生成本体を `func/generation_promptsmiles_func.py` に移す

- **Target file**: `src/func/generation_promptsmiles_func.py`（新規）
- **Changes**:
  - `src/gen_mols/gen_promptsmiles.py` の**全内容をそのまま移す**。
    - モジュール docstring、import、`MIN_LINK_FRAGMENTS`、`log_line`、`parse_fragments`、
      `select_prompt_fragments`、`to_prediction_row`、`format_run_summary`、
      `GPT2PromptSampler`、`build_prompter`、`parse_args`、`if __name__ == "__main__":` ブロック
  - **`if __name__ == "__main__":` ブロックの中身を `main()` 関数に切り出す**。
    `func/generation_fraggpt_func.py` と同じ形（`def main() -> None:` +
    末尾に `if __name__ == "__main__": main()`）にすること。
  - `parse_args` に `--additional_path`（`type=str, default="normal"`）を追加する。
    help は `gen_fraggpt.py:26` と同じ文言に揃えること。
  - `output_dir` の `normal` 直書きを `{args.additional_path}` に変える。
  - **生成ロジックは1行も変更しない。** 移動と `main()` への切り出し、
    `--additional_path` の追加だけ。
  - モジュール docstring の冒頭を、`func/generation_fraggpt_func.py` /
    `func/generation_safe_func.py` と同じ体裁（このファイルが生成本体であることが分かる形）に
    整えること。既存の説明本文は残す。
- **Dependencies**: none

### Step 2: `gen_mols/gen_promptsmiles.py` をラッパに書き換える

- **Target file**: `src/gen_mols/gen_promptsmiles.py`
- **Changes**:
  - 中身を全部捨て、`src/gen_mols/gen_fraggpt.py` と**同じ構造**のラッパにする。
  - モジュール docstring は `gen_fraggpt.py:1-6` と同じ体裁で、参照先を
    `src/func/generation_promptsmiles_func.py` にする。
  - 引数は生成本体の `parse_args` と**同じもの**を受け取り、そのまま子プロセスに渡す。
    現在の PromptSMILES の引数構成に合わせること（`--frag_method`, `--model_ver`,
    `--gen_method`, `--additional_path`, `--n_samples`, `--max_length`, `--num_beams`,
    `--random_seed` など。実際の `parse_args` を読んで漏れなく渡すこと）。
  - `output_dir` は
    `f'{BASEPATH}/results/promptsmiles/gpt/{model_ver}/{frag_method}/{args.gen_method}/{args.additional_path}'`。
    **生成本体が計算するものと完全に一致させること。**
  - `model_path` は `f'{BASEPATH}/models/promptsmiles/gpt/{model_ver}/{frag_method}/best_model'`。
  - `run_and_record_time` に渡す `params` は `gen_fraggpt.py:57-67` と同じキー構成にする。
    PromptSMILES に無いキー（`batch_size` など）は生成本体の引数に合わせて調整すること。
  - **`os.environ["CUDA_LAUNCH_BLOCKING"] = "1"` は付けない。**
    これは FragGPT の GPU beam クラッシュを回避するための措置で、PromptSMILES では
    その問題が確認されていない。付けると生成が遅くなり、その時間が計測に入る。
- **Dependencies**: after Step 1

### Step 3: `gen_promptsmiles.sh` を確認する

- **Target file**: `src/gen_mols/gen_promptsmiles.sh`
- **Changes**:
  - 呼び出し先は `${SCRIPT_DIR}/gen_promptsmiles.py` のままで変わらない
    （ラッパが同じパスに置かれるため）。
  - 生成本体に `--additional_path` が増えたので、`.sh` から渡す必要があるか確認する。
    他の `gen_*.sh` に倣い、**渡さない（既定の `normal` を使う）**なら変更不要。
  - `conda activate promptsmiles` と `CUDA_VISIBLE_DEVICES` は同日の別計画で追加済み。
    重複させないこと。
- **Dependencies**: after Step 2

### Step 4: README の記述を更新する

- **Target file**: `README.md`, `README_ja.md`
- **Changes**:
  - 生成セクションの PromptSMILES の説明に、他手法と同じく
    `generation_time.json` が出力される旨が読み取れる状態にする。
    ただし**手法ごとに個別の注記を増やさないこと** — 現在の記述で
    「4手法とも同じ形式で出力する」と読めるなら追加は不要。実際の文面を読んで判断すること。
  - 生成本体のパスに言及している箇所があれば
    `src/func/generation_promptsmiles_func.py` に直す。
- **Dependencies**: after Step 2

### Step 5: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 構文チェックと import 確認（**`joblib` 未導入のため import は失敗する見込み。
    その場合は `joblib` を一時的に stub して確認すること**）
  - `gen_promptsmiles.py --help` と `generation_promptsmiles_func.py --help` の
    引数が一致すること
  - **ラッパと生成本体が計算する `output_dir` が完全に一致すること**
    （`--additional_path` を `normal` 以外にしても一致すること）
  - `run_and_record_time` に渡す `params` のキーが他手法の
    `generation_time.json` と揃っていること
  - **生成ロジックが移動前と同一であること**（`git show HEAD:src/gen_mols/gen_promptsmiles.py` と
    新ファイルを、docstring・`main()` 切り出し・`--additional_path` 追加を除いて差分比較）
  - 実モデルで少数行を生成し、`predictions.csv` と `generation_time.json` の両方が
    書かれること（`joblib` 導入後に実施。**未導入なら「未検証」と明記して報告すること**）
- **Dependencies**: after Step 4
