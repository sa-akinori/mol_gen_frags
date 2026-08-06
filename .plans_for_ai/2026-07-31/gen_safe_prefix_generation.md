# Plan: gen_safe_denovo.py を「部分SAFE→完全SAFE 直接生成」に全面書き換え

- **Date**: 2026-07-31
- **Status**: implemented (rev.5)

## 実装後の変更（rev.4〜5、ユーザー指示による）

| # | 変更 | 理由 |
|---|---|---|
| 1 | 無効デコード時のセンチネルを `""` → `'safe_invalid'` | 生成は必ず何らかの文字列を返すため、実質的な問題は「SMILES に変換できるか」だけ。変換できなかったことを明示的に記録する。旧 `generation_safe_func.py:124,135,136` と同じセンチネルで、`loadGenSmiles` の `time_out|error` には一致しないため行は残り無効カウントになる |
| 2 | 行コメントを2件のみに整理 | 「無駄なコメントは入れない」。コードから読み取れることは書かず、壊されると困る非自明な制約（左パディングの必要性、`index_col=0` 契約）だけ残す |
| 3 | `_canonicalize_smiles` を削除し `func.evaluation_func.Smi2CanSmi` を利用 | 「既に存在する関数を新しく作らない」。`Smi2Mol` が None で `print` するため `if decoded else None` のガードを付ける（450万回ループでの出力汚染を回避）。未参照になった `from rdkit import Chem` も削除 |
| 4 | 出力から `full_safe` 列を削除 | 生成に使わず、`pass_safe` は `full_safe` の接頭辞ではないため文字列比較もできず、`sc3_check_genmol_results` も `target` と `fragment` しか読まない |

| 5 | `--machine_id` / `--total_machines` とシャーディングを削除 | ユーザー判断で不要。副次的に出力名が `predictions.csv` / `error_logs.csv` になり、`evaluation.py:83` が読む名前と一致するため、シャード結合処理が存在しない問題も解消する |

最終的な出力カラム: `target, pass_safe, fragment, safe_1..N, prediction_1..N`
最終的な出力ファイル: `{output_dir}/predictions.csv`、`{output_dir}/error_logs.csv`

## Overview

`src/gen_safe_denovo.py` を全面的に上書きし、次の機能を持つスクリプトにする。

> 学習済み SAFE-GPT モデルに、test split の `pass_safe`（部分的な SAFE 文字列）を
> プレフィックスとして与え、`model.generate()` を直接呼んで完全な SAFE 文字列を生成する。

`SAFEDesign` の高レベル生成API（`linker_generation` / `scaffold_decoration` /
`scaffold_morphing` / `de_novo_generation`）は一切使わない。
**実装様式は `src/func/generation_rffmg_func.py` に揃える。**

### ユーザー確定事項

| 論点 | 決定 |
|---|---|
| 位置づけ | まず単体スクリプト。後日 `generation_safe_func.py` の置き換えに発展させる前提 |
| 出力 | 生成された完全SAFE文字列と、`safe.decode` した canonical SMILES の**両方** |
| `safe.decode()` | 使用可（禁止は `SAFEDesign` の生成APIのみ） |
| デコード方法 | beam search（`num_beams=50`, `num_return_sequences=50`, `do_sample=False`） |
| 設計案 | 案B（左パディングによるバッチ生成 + argparse + シャーディング配線済み） |
| 対象行数 | test split 全行（既に `subsample_test_split` で間引き済みの評価用セット） |
| 実装様式 | **`generation_rffmg_func.py` に揃える** |
| `max_position_embeddings` | ロード時にチェックし、`max_length` が超える場合は警告する |

### 調査で確定した事実（実装の根拠）

1. **モデルは `full_safe` のみを無条件に自己回帰学習**している（`safe-train --text_column full_safe`）。
   よって「妥当なSAFEの途中文字列を与えて続きを生成させる」推論は学習分布と整合する。
2. **`pass_safe` は `full_safe` の文字列接頭辞とは限らない**。実測（全 13,173,912 行）で
   **76.5% が非接頭辞**、15.7% が接頭辞（断片境界）、7.8% が完全一致。
   → 生成には支障ないが、**「生成SAFE == full_safe」の文字列一致での評価は成立しない**。
3. **プロンプト末尾に EOS を付けてはいけない**。`SAFETokenizer.get_pretrained()` は
   `TemplateProcessing(single="[CLS] $A [SEP]")` で BOS/EOS を自動付与する。
   → rffmg と同じく **`add_special_tokens=False` + `bos_id` 手動付与**で回避する。
4. **`safe.decode()` の既定は `ignore_errors=False`（例外送出）**。`ignore_errors=True` の明示が必要。
5. **safe ライブラリは1プロンプトずつしか生成しない**（`sample.py:1009` は文字列1本のみ受け取る）。
   バッチ化は本スクリプトの追加要素。
6. **test split 行数**: brics 82,441 / rc_cms 90,974。シャーディングを最初から配線する。

### `generation_rffmg_func.py` から踏襲する点

| 項目 | rffmg の実装 | 参照 |
|---|---|---|
| pad 設定 | `pad_token_id is None` なら `pad_token = eos_token` | L70-71 |
| padding_side | `tokenizer.padding_side = "left"` を明示（コメント付き） | L72-73 |
| トークナイズ | `tokenizer(..., add_special_tokens=False)` + `bos_id` 手動付与 | L92-93 |
| パディング | 手動で左詰め（`[pad_id] * 不足 + ids`）、`attention_mask` も手組み | L94-99 |
| 生成 | `torch.no_grad()` 内で `do_sample=False, num_beams, num_return_sequences, max_length, early_stopping=True, pad_token_id, eos_token_id` | L101-110 |
| デコード | `tokenizer.batch_decode(outputs, skip_special_tokens=True)` → `n_samples` 本ずつグループ化 | L112-116 |
| バッチループ | `tqdm(range(0, len(sources), args.batch_size))` | L86 |
| `set_seed` | `from transformers import set_seed`（`func.utility` ではない） | L19, L63 |
| `batch_size` 既定 | **24** | L55-56 |
| 構成 | `read_*` / `parse_args` / `main` のみ。CLI は argparse | 全体 |

### rffmg から**意図的に外す**点（理由つき）

| 項目 | rffmg | 本計画 | 理由 |
|---|---|---|---|
| **CSV の index** | `to_csv(..., index=False)` | **`index=False` を付けない** | `evaluation_func.loadGenSmiles` は `safe_gpt` 分岐で `pd.read_csv(file_name, index_col=0)` と読む（L45）。rffmg は T5Chem 列様式で別分岐のため `index=False` でよいが、SAFE で揃えると列が1つずれる |
| 補完部分の切り出し | `text.split(">>", 1)[1]` で後半のみ取る | **切り出さない** | 欲しいのは完全SAFE（プロンプト＋続き）そのもの。`skip_special_tokens=True` で `[CLS]`/`[SEP]`/PAD が落ち、そのまま完全SAFE文字列になる |
| シャーディング | なし | `--machine_id` / `--total_machines` を追加 | 9万行を複数マシンで分担する必要があるため。既存 `gen_mols/gen_safe.py` が渡す引数名に一致させる |
| エラー処理 | なし（例外で全体停止） | バッチ単位 `try/except` + `error_logs_{machine_id}.csv` | 数時間〜十数時間の実行で1バッチの失敗が全損になるのを避けるため |
| 出力カラム | `target` + `prediction_1..N` | `+ full_safe, pass_safe, fragment, safe_1..N` | 生SAFEの保存がユーザー要件。`fragment` は `sc3_check_genmol_results` が要求 |

## Plan

対象ファイルは **`src/gen_safe_denovo.py` のみ**（全面上書き）。他ファイルは変更しない。

### Step 1: モジュールヘッダと import を差し替える

- **Target file**: `src/gen_safe_denovo.py`（冒頭）
- **Changes**:
  - モジュール docstring を新規に書く（rffmg の docstring L1-11 と同じ粒度）。
    「`SAFEDesign` の高レベル生成APIを使わず `model.generate()` を直接呼ぶ」
    「`safe.decode` は SAFE→SMILES 変換にのみ使う」
    「プロンプトは `[CLS] pass_safe`、出力は完全SAFE文字列」を明記する。
  - `from safe.sample import SAFEDesign` を**削除**する。
  - import を追加: `argparse`, `numpy as np`, `datasets`, `tqdm.tqdm`,
    `transformers.set_seed`（rffmg に合わせ `func.utility.set_seed` から変更）,
    `transformers.PreTrainedTokenizerBase`（型ヒント用）。
  - `from func.utility import set_seed` を削除する。
  - import順は 標準 → サードパーティ → ローカル を維持する。
  - `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` は現行のまま残す。
- **Dependencies**: none

### Step 2: モデル/トークナイザのロード関数を追加する

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:

  ```python
  def load_model_and_tokenizer(
      model_path: str | Path,
      max_length: int,
  ) -> tuple[SAFEDoubleHeadsModel, PreTrainedTokenizerBase, torch.device]:
      """学習済み SAFE-GPT モデルと HuggingFace トークナイザをロードする。"""
  ```

  処理（rffmg L66-77 に対応）:
  1. `SAFEDoubleHeadsModel.from_pretrained(str(model_path))`
  2. `SAFETokenizer.from_pretrained(str(model_path)).get_pretrained()`
  3. `tokenizer.pad_token_id is None` なら `tokenizer.pad_token = tokenizer.eos_token`
  4. `tokenizer.padding_side = "left"`（rffmg L72-73 と同じくコメント付きで明示）
  5. **`max_length > model.config.max_position_embeddings` なら警告を print する**
     （位置埋め込みの上限を超える指定を検知するため）
  6. device を決めて `model.to(device)` / `model.eval()`、`print(f"Using device: {device}")`
  7. `(model, tokenizer, device)` を返す
- **Dependencies**: after Step 1

### Step 3: test split の読み込み・シャーディング関数を追加する

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:

  ```python
  def load_test_split(
      dataset_dir: str | Path,
      machine_id: int = 0,
      total_machines: int = 1,
      n_generate: int | None = None,
  ) -> pd.DataFrame:
      """test split を読み込み、担当マシン分の行だけを DataFrame で返す。

      Returns:
          pd.DataFrame: カラムは ['smiles', 'full_safe', 'pass_safe',
              'full_fragments', 'pass_fragments']。index は元の test split の行番号。
      """
  ```

  1. `datasets.load_from_disk(str(dataset_dir))["test"]`
  2. `n_generate` 指定時は先頭 `n_generate` 行に絞る（動作確認用。既定 `None` = 全行）
  3. `np.array_split(np.arange(n_rows), total_machines)[machine_id]` で担当行を決める
  4. `test_dataset.select(indices).to_pandas()` で DataFrame 化し、**index に元の行番号を設定**する
  - 注: 現行 `generation_safe_func.py` の `test_dataset['smiles'][indices]`（list を ndarray で
    添字）は `TypeError` になる。`select()` を使うことでこのバグを踏まない。
- **Dependencies**: after Step 1

### Step 4: プロンプト構築（左パディング）関数を追加する

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:

  ```python
  def encode_prefixes(
      prefixes: list[str],
      tokenizer: PreTrainedTokenizerBase,
      device: torch.device,
  ) -> tuple[torch.Tensor, torch.Tensor]:
      """部分SAFE文字列を左パディングしてモデル入力テンソルに変換する。

      Returns:
          tuple[torch.Tensor, torch.Tensor]: (input_ids, attention_mask)。
      """
  ```

  **rffmg L92-99 と同じ手順**:
  1. `tokenizer(prefixes, add_special_tokens=False)`
  2. 各系列の先頭に `tokenizer.bos_token_id`（= `[CLS]`）を付ける → `[CLS] <prefix>`
  3. バッチ内最大長に手動で左詰めし、`attention_mask` を組む
  4. `torch.tensor(..., dtype=torch.long, device=device)` で返す
  - docstring に「末尾 EOS を付けない理由」を明記する
    （safe ライブラリの `sample.py:1039-1042` の `[:, :-1]` と同じ意図であること）。
- **Dependencies**: after Step 1

### Step 5: 生成関数を追加する

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:

  ```python
  def generate_safe_sequences(
      model: SAFEDoubleHeadsModel,
      tokenizer: PreTrainedTokenizerBase,
      prefixes: list[str],
      n_samples: int,
      num_beams: int,
      max_length: int,
      device: torch.device,
  ) -> list[list[str]]:
      """部分SAFEのバッチから beam search で完全SAFE文字列を生成する。

      Returns:
          list[list[str]]: 入力プレフィックスごとに n_samples 本の完全SAFE文字列。
      """
  ```

  **rffmg L101-116 と同じ手順**:
  1. `encode_prefixes(...)`
  2. `with torch.no_grad():` で `model.generate(input_ids=..., attention_mask=...,
     do_sample=False, num_beams=num_beams, num_return_sequences=n_samples,
     max_length=max_length, early_stopping=True,
     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)`
  3. `tokenizer.batch_decode(outputs, skip_special_tokens=True)`
  4. `n_samples` 本ずつグループ化して行ごとのリストに戻す
  - **rffmg と違い `">>"` での切り出しは行わない**（欲しいのは完全SAFE全体のため）。
- **Dependencies**: after Step 4

### Step 6: SAFE→SMILES デコード関数を整理する

- **Target file**: `src/gen_safe_denovo.py`（現行 L17-43）
- **Changes**: 現行の `_canonicalize_smiles` と `_decode_safe_smiles` を**そのまま残す**
  （`safe.decode(..., ignore_errors=True, ...)` + `Chem.MolFromSmiles` の None チェック済みで、
  safe ライブラリの `_decode_safe`（`sample.py:844-853`）と同一の引数）。
  `_decode_safe_smiles` の docstring に「空文字列・デコード失敗時は None を返す」ことを追記する。
- **Dependencies**: none

### Step 7: 結果 DataFrame 構築関数を追加する

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**:

  ```python
  def build_predictions_df(
      test_df: pd.DataFrame,
      generated_safe: list[list[str]],
      n_samples: int,
  ) -> pd.DataFrame:
      """生成結果を保存用の DataFrame に組み立てる。

      Returns:
          pd.DataFrame: カラムは ['target', 'full_safe', 'pass_safe', 'fragment']
              + ['safe_1', ..., f'safe_{n_samples}']（生成された完全SAFE文字列）
              + ['prediction_1', ..., f'prediction_{n_samples}']（canonical SMILES、
              SMILES に変換できなかった場合は 'safe_invalid'）。
              index は元の test split の行番号。
      """
  ```

  | 列 | 内容 | 由来 |
  |---|---|---|
  | (index) | test split の行番号 | `load_test_split` |
  | `target` | 参照分子の SMILES | `smiles` |
  | `full_safe` | 正解の完全SAFE（参考用） | `full_safe` |
  | `pass_safe` | 入力したプレフィックス | `pass_safe` |
  | `fragment` | 断片集合（`sc3_check_genmol_results` が要求） | `pass_fragments` |
  | `safe_1..N` | **生成された完全SAFE文字列** | `generate_safe_sequences` |
  | `prediction_1..N` | 上を `_decode_safe_smiles` した canonical SMILES | 変換不可なら `'safe_invalid'` |

  - **`safe_i` という列名にするのは必須**。`safe_prediction_i` にすると
    `loadGenSmiles`（列名に `prediction` を**部分一致**で含む列を拾う）に巻き込まれる。
  - **無効時は空文字列ではなく `'safe_invalid'` を入れる**（ユーザー指示 rev.3）。
    生成は必ず何らかの文字列を返すため、実際に問題になるのは
    「その文字列を SMILES に変換できるか」だけである。変換できなかったことを
    明示的に記録する。旧 `generation_safe_func.py:124,135,136` と同じセンチネルを使う。
    `'safe_invalid'` は `loadGenSmiles` の `time_out|error` 正規表現に一致しないため、
    行は除外されず、RDKit がパースに失敗して**無効カウント**になる（旧実装と同じ挙動）。
- **Dependencies**: after Step 5, Step 6

### Step 8: `parse_args` と `main` を追加する

- **Target file**: `src/gen_safe_denovo.py`
- **Changes**: rffmg の `parse_args`（L38-59）/ `main`（L62-126）と同じ構成で追加する。

  | 引数 | 既定 | 備考 |
  |---|---|---|
  | `--model_path` | 必須 | rffmg と同じ |
  | `--dataset_dir` | 必須 | SAFE は DatasetDict なので rffmg の `--dataset_file`/`--target_file` ではなくこちら |
  | `--output_dir` | 必須 | rffmg と同じ |
  | `--n_samples` | 50 | rffmg と同じ |
  | `--num_beams` | 50 | rffmg と同じ |
  | `--max_length` | 200 | 既存 SAFE パイプライン（`gen_safe.sh`）の値。rffmg は 256 |
  | `--batch_size` | **24** | rffmg に合わせる |
  | `--random_seed` | 42 | rffmg と同じ |
  | `--n_generate` | None | 追加（動作確認用。既定は全行） |
  | `--machine_id` | 0 | 追加（`gen_mols/gen_safe.py` が渡す引数名に一致） |
  | `--total_machines` | 1 | 追加 |

  `main` の処理:
  1. `set_seed(args.random_seed)`（`transformers.set_seed`）
  2. `load_model_and_tokenizer` / `load_test_split`
  3. `tqdm(range(0, len(test_df), args.batch_size), desc='prediction')` でバッチループ
     - 各バッチを `try/except` で囲み、例外時はそのバッチ全行の `safe_i` を `"error"` で埋めて継続する
     - 例外は `error_logs` に記録する
  4. `build_predictions_df` → `{output_dir}/predictions_{machine_id}.csv` に保存する。
     **`index=False` は付けない**（`loadGenSmiles` の `index_col=0` 契約に合わせるため。
     rffmg とは異なる点なので、その理由をコメントで明記する）
  5. `error_logs` を `{output_dir}/error_logs_{machine_id}.csv` に保存する
     （カラム: `['row_index', 'target', 'pass_safe', 'error_type', 'error_message']`）
  6. 保存先を `print` する（rffmg L125-126 と同じ体裁）
- **Dependencies**: after Step 7

### Step 9: `__main__` を差し替える

- **Target file**: `src/gen_safe_denovo.py`（現行 L153-165）
- **Changes**: rffmg L129-130 と同じ形にする。

  ```python
  if __name__ == "__main__":
      main()
  ```
- **Dependencies**: after Step 8

### Step 10: 削除される要素の確認と構文チェック

- **Target file**: なし（検証のみ）
- **Changes**:
  - 削除されること: `generate_denovo_direct` / `generate_denovo_library` / `GENERATORS` /
    `run_denovo_generation` / `from safe.sample import SAFEDesign` / `from func.utility import set_seed`
  - `python -m py_compile src/gen_safe_denovo.py` が通ることを確認する
  - `grep -n "SAFEDesign\|de_novo\|linker_generation\|scaffold_" src/gen_safe_denovo.py` が
    空になることを確認する
  - 実際の生成実行（GPU・モデル要）は行わない
- **Dependencies**: after Step 9

## 未決事項

- **ファイル名**: 中身が de novo でなくなるため `gen_safe_denovo.py` という名前が実態と合わなくなる。
  「上書き」のご指示どおり据え置く前提。改名が望ましければ対応する。
- **`--max_length` の意味**: `generate(max_length=)` はパディング込みの全長を数える。
  左パディングしたバッチでは、長いプレフィックスと同居した行の生成可能長が縮む。
  rffmg も同じ扱いなので、**揃える方針に従い `max_length` のまま**とする。

## Out of scope（今回は触らない）

- `src/func/generation_safe_func.py`（実行不能なバグ3件と `**kwargs` 渡し忘れによる
  beam→random フォールバックを確認済みだが、今回は手を付けない）
- `src/gen_mols/gen_safe.py` / `gen_safe.sh` / `src/func/generation_time.py`
- `src/evaluation.py` / `src/func/evaluation_func.py`
- シャード出力（`predictions_{machine_id}.csv`）を `predictions.csv` に結合する処理
  （リポジトリ全体に存在しないことを確認済み。評価時に別途必要）
- コミット（指示があるまで行わない）
