# Plan: 旧版 src/gen_safe.py の削除

- **Date**: 2026-07-31
- **Status**: pending-approval

## Overview

`src/gen_safe.py`（ルート直下の direct 生成版）は現行の `src/gen_mols/gen_safe.py` に置き換えられ、
どこからも参照されなくなった旧版である。以下の理由から削除する。

- 最終更新は 2026-07-09 (`ef1b5ab`)。以降の改修（生成時間記録・マシン分散・CLI 引数化・pretrained 対応）は
  すべて `src/gen_mols/gen_safe.py` 側にのみ入っている。
- beam ブランチ（L176-188）は `src/func/generation_safe_func.py` に存在しない
  `--slice_name` / `--gen_method` を渡しており、実行すると argparse エラーで落ちる（既に動作しない）。
- `.plans_for_ai/2026-07-27/gen_inference_time.md` L20 で明示的にスコープ外とされたまま放置されている。

### 参照調査の結果（削除の安全性）

| 対象 | 結果 |
|---|---|
| `import gen_safe` / `from gen_safe import` | なし |
| `generate_from_model()` の外部呼び出し | なし（同ファイル内のみ） |
| `.sh` からの呼び出し | なし（`gen_mols/gen_safe.sh` は `${SCRIPT_DIR}/gen_safe.py` = `gen_mols/` 側を指す） |
| `README_ja.md` の記載 | なし（L153 は `src/gen_mols/gen_safe.sh`） |
| `timeout_handler` の共有 | なし（`src/func/generation_safe_func.py` が独自に定義） |

## Plan

### Step 1: `src/gen_safe.py` を削除する

- **Target file**: `src/gen_safe.py`（削除）
- **Changes**: `git rm src/gen_safe.py` でファイルを削除する。git 履歴には残るため必要になれば復元可能。
- **Dependencies**: none

### Step 2: 削除後の健全性確認

- **Target file**: なし（検証のみ）
- **Changes**:
  - `grep -rn "gen_safe" --include="*.py" --include="*.sh" --include="*.md" .` を再実行し、
    生きているコード／ドキュメントに壊れた参照が残っていないことを確認する。
  - `python -m py_compile src/gen_mols/gen_safe.py src/func/generation_safe_func.py` が通ることを確認する。
- **Dependencies**: after Step 1

## Out of scope（今回は触らない）

- `src/gen_safe_denovo.py`: 同様に `gen_inference_time.md` でスコープ外とされているが、
  今回のユーザー指示は `src/gen_safe.py` のみ。要否は別途確認する。
- `.plans_for_ai/` / `.reviews_by_ai/` 配下の過去の計画・レビュー文書:
  作成時点の記録なので書き換えない。
- コミットするかどうかはユーザーの指示を待つ。
