# Plan: 推論(分子生成)時間を保存する

- **Date**: 2026-07-27
- **Status**: approved

## Overview

現在、どの生成エントリポイントも推論にかかった時間を一切保存していない
(`results/` 配下に時間を記録したファイルはゼロ)。
推論1回ごとのウォールクロック時間を、`predictions.csv` と同じ出力ディレクトリに
JSON で保存できるようにする。

### 対象エントリポイント

| エントリポイント | バックエンド | 出力 |
|---|---|---|
| `src/gen_mols/gen_rffmg.py` | 外部 `t5chem predict` CLI / `generation_rffmg_func.py` (GPT2) | `generation_time.json` |
| `src/gen_mols/gen_safe.py`  | `generation_safe_func.py` (machine_id でシャーディング) | `generation_time_{machine_id}.json` |

`src/gen_safe.py`(ルート直下の direct 版)・`src/gen_safe_denovo.py`・`src/gen_t5chem.py`
は今回のスコープ外(必要なら同じヘルパーを後から呼ぶだけで拡張できる設計にする)。

### 設計方針

1. **計測は subprocess の外側(wrapper 側)で行う**。t5chem は外部インストール済み CLI で
   変更できないため、両バックエンドで共通に取れるのは「生成プロセス全体の経過時間」のみ。
   モデルロード時間を含む点は仕様として JSON に明記する。
2. **共通ヘルパーは新規モジュール `src/func/generation_time.py` に置く**。
   `utility.py` はタブインデント・型ヒントなしのレガシー集積所であり、
   `.claude/CLAUDE.md` のコードスタイル(型ヒント必須・Google style docstring)に合わない。
   `generation_rffmg_func.py` / `generation_safe_func.py` と同じ「生成関連は専用モジュール」
   の既存パターンにも沿う。
3. **分子数は入力ファイルではなく出力 `predictions*.csv` の行数から数える**。
   理由: `gen_rffmg.py` の `dataset_dir` (`data/rffmg/{frag}/{additional}`) は
   ディスク上の実レイアウト (`data/rffmg/{frag}/{n}times_sampling/{additional}`) と
   ずれており、`test.source` を数える方式は 0 を返して壊れる。
   出力CSVは両バックエンドとも「1分子=1行」で安定している。
4. **subprocess が失敗した場合(`check=True` で例外)は JSON を書かない**。
   途中で落ちた実行の時間はベンチマークとして無意味なため。
5. **再現性への影響なし**。乱数シード・cmd・生成パラメータには一切触れず、
   計測は受動的な観測のみ。

### 既存作業との関係

未マージのブランチ `worktree-gen-time-logging` (commit `3877951`, 2026-07-23) に
`gen_rffmg.py` 内へ直接ヘルパーを書く形の実装が既にある。ただし
(a) `gen_rffmg` のみで SAFE 側に再利用できない、
(b) 分子数を `test.source` から数えるため上記3の理由で 0 になる、
(c) ブランチが `make_datasets.py` / `train_gpt.py` の古い状態を含み main と衝突する、
ため **cherry-pick せず main 上で作り直す**。設計思想(perf_counter + JSON + 主要キー)は踏襲する。

## Plan

### Step 1: 共通ヘルパーモジュールを新規作成する

- **Target file**: `src/func/generation_time.py` (新規)
- **Changes**:
  - import 順は 標準ライブラリ(`json`, `subprocess`, `time`, `datetime`, `pathlib`, `typing`)
    → サードパーティ(`pandas`) → ローカル(なし)。
  - 関数を2つ定義する(いずれも型ヒント + Google style docstring)。

    ```python
    def count_prediction_rows(output_dir: Path, pattern: str = "predictions*.csv") -> int | None:
        """生成結果CSVの行数(=処理した分子数)を数える。

        Args:
            output_dir: predictions CSV が置かれたディレクトリ。
            pattern: 対象を選ぶ glob パターン。シャード実行時は
                ``predictions_{machine_id}.csv`` のように自分の分だけを指定する。

        Returns:
            マッチしたCSVの合計行数。1ファイルもマッチしない場合は None。
        """
    ```

    ```python
    def run_and_record_time(
        cmd: list[str],
        output_dir: Path,
        n_samples: int,
        params: dict[str, Any] | None = None,
        record_name: str = "generation_time.json",
        predictions_pattern: str = "predictions*.csv",
    ) -> Path:
        """生成コマンドを実行し、経過時間をJSONに保存する。

        ``subprocess.run(cmd, check=True)`` を ``time.perf_counter()`` で挟み、
        完了後に ``output_dir/record_name`` へ記録を書き出す。コマンドが失敗した
        場合は例外がそのまま送出され、JSONは書かれない。

        Args:
            cmd: subprocess に渡すコマンド。
            output_dir: predictions と同じ出力ディレクトリ(無ければ作成)。
            n_samples: 1分子あたりの生成サンプル数。
            params: 記録しておく生成パラメータ(num_beams, model_path 等)。
            record_name: 出力するJSONのファイル名。
            predictions_pattern: 分子数を数えるための glob パターン。

        Returns:
            書き出したJSONファイルのパス。
        """
    ```

  - JSON のキー:
    | key | 内容 |
    |---|---|
    | `elapsed_sec` | 生成プロセス全体の経過秒(小数第3位まで丸め、モデルロード込み) |
    | `n_molecules` | 出力CSVの行数。数えられない場合 `null` |
    | `n_samples` | 1分子あたりの生成サンプル数 |
    | `sec_per_molecule` | `elapsed_sec / n_molecules`(0件・不明時は `null`) |
    | `recorded_at` | `datetime.datetime.now().isoformat(timespec="seconds")` |
    | `params` | バックエンド固有の生成パラメータ(dict) |
  - `elapsed_sec` の計測は `time.perf_counter()`(単調時計)、`recorded_at` は壁時計と使い分ける。
    `utility.GetTime()` は区切りなしの数字列(例 `2026727_1430`)でパースしづらいため使わない。
- **Dependencies**: none

### Step 2: `gen_rffmg.py` の subprocess 呼び出しを差し替える

- **Target file**: `src/gen_mols/gen_rffmg.py`
- **Changes**:
  - `import subprocess` を削除し、`from pathlib import Path` と
    `from func.generation_time import run_and_record_time` を追加(import 順は維持)。
  - 末尾の `subprocess.run(cmd, check=True)` を以下に置き換える。

    ```python
    json_path = run_and_record_time(
        cmd,
        Path(output_dir),
        n_samples=args.n_samples,
        params={
            "backend": model_name,
            "model_ver": model_ver,
            "frag_method": frag_method,
            "additional_path": additional_path,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "model_path": model_path,
            "random_seed": args.random_seed,
        },
    )
    print(f"Saved generation time to: {json_path}")
    ```
  - このスクリプトはシャーディングしないので `record_name` / `predictions_pattern` は既定値のまま。
- **Dependencies**: after Step 1

### Step 3: `gen_safe.py` の subprocess 呼び出しを差し替える

- **Target file**: `src/gen_mols/gen_safe.py`
- **Changes**:
  - `import subprocess` を削除し、`from pathlib import Path` と
    `from func.generation_time import run_and_record_time` を追加。
    未使用の `import itertools` / `import os` はこの機会に整理する(実際に未使用であることを確認したうえで削除)。
  - 末尾の `subprocess.run(cmd, check=True)` を以下に置き換える。

    ```python
    json_path = run_and_record_time(
        cmd,
        Path(output_dir),
        n_samples=args.n_samples,
        params={
            "backend": "safe_gpt",
            "model_ver": model_ver,
            "frag_method": frag_method,
            "num_beams": args.num_beams,
            "max_length": args.max_length,
            "model_path": model_path,
            "machine_id": args.machine_id,
            "total_machines": args.total_machines,
            "random_seed": args.random_seed,
        },
        record_name=f"generation_time_{args.machine_id}.json",
        predictions_pattern=f"predictions_{args.machine_id}.csv",
    )
    print(f"Saved generation time to: {json_path}")
    ```
  - `predictions_{machine_id}.csv` / `error_logs_{machine_id}.csv` と同じ suffix 規約に揃えることで、
    複数マシンが同一 `output_dir` に同時書き込みしても衝突しない(ロック不要)。
- **Dependencies**: after Step 1

### Step 4: 動作確認

- **Target file**: 変更なし(確認のみ)
- **Changes**:
  - `python -m py_compile src/func/generation_time.py src/gen_mols/gen_rffmg.py src/gen_mols/gen_safe.py`
  - 一時ディレクトリでの単体確認: ダミー `predictions.csv` を置き、
    `cmd = ["python", "-c", "import time; time.sleep(0.2)"]` で `run_and_record_time` を呼び、
    `elapsed_sec` ≈ 0.2 / `n_molecules` がCSV行数と一致 / `sec_per_molecule` が整合することを確認。
  - 失敗系: `cmd = ["python", "-c", "raise SystemExit(1)"]` で `CalledProcessError` が送出され、
    JSONが作られないことを確認。
  - 実際のGPU生成は重いので走らせない(既存 `predictions.csv` は上書きしない)。
- **Dependencies**: after Step 2, Step 3

## 実装時の計画からの差分

- 計画の Step 1 のコードスケッチでは docstring を日本語で書いていたが、実装では
  英語に統一した。`src/func/` の既存モジュール(`generation_rffmg_func.py`,
  `generation_safe_func.py`)がすべて英語 docstring であり、1ファイル内で
  モジュール docstring(英語)と関数 docstring(日本語)が混在するのを避けるため。
  内容(引数説明・戻り値・キー)は計画どおり。

## Notes / 今回スコープ外(別途要判断)

1. **`src/func/generation_safe_func.py` に既存バグがある**
   - L78: `args.max_` は未定義引数(`--max_dataset_num` の書き間違いと思われる) → `AttributeError`
   - L156 / L166: `MACHINE_ID` が未定義(`args.machine_id` であるべき) → `NameError`
   - このため SAFE の beam 生成は現状そもそも完走できない。今回の変更(wrapper 側のみ)とは
     独立だが、Step 3 の効果を実機確認するには先にこの修正が必要。修正するか指示がほしい。
2. **`gen_rffmg.py` の `dataset_dir` が実データレイアウトとずれている**
   - コード: `data/rffmg/{frag_method}/{additional_path}`
   - 実態: `data/rffmg/{frag_method}/{5,10}times_sampling/{additional_path}`
   - `data/rffmg/rc_cms/normal` は存在しない。これも今回のスコープ外。
3. **バックエンド横断の比較について**: `elapsed_sec` はモデルロードや外部CLIの起動を含む
   プロセス全体時間なので、t5chem と GPT2/SAFE の厳密な推論速度比較には向かない。
   分子単位・バッチ単位の内部計測が必要になったら、`generation_*_func.py` 側に
   `elapsed_sec` 列を追加する拡張が可能(下流の評価コードは列を名前で選択しているため、
   列追加で壊れないことは確認済み)。ただし t5chem は外部CLIのため同じ計測はできず、
   バックエンド間で意味の異なる数値になる点に注意。
