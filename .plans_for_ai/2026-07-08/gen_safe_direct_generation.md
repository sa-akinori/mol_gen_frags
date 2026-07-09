# Plan: gen_safe.py に「学習済みモデルからの直接生成 + RDKit 妥当性判定」を実装

- **Date**: 2026-07-08
- **Status**: pending-approval

## Overview

test split のフラグメント（`pass_safe`）をプレフィックスとして学習済み SAFE モデルの
`model.generate` に **直接** 与えて分子を生成し、生成物を **RDKit で valid/invalid 判定** する。

- `SAFEDesign.scaffold_decoration` / `scaffold_morphing` などの高レベル API は使わない。
- `model.generate`（HuggingFace）を直接呼ぶ。トークナイズ→生成→デコードは
  `safe.sample.SAFEDesign._generate` の実装を参照（プレフィックスのトークナイズ、末尾 EOS 除去、
  `batch_decode`、`safe.decode` で SAFE→SMILES）。
- `src/func/generation_safe_func.py` は変更しない（参照のみ）。

### 確定した設定（ユーザー承認済み）

| 項目 | 値 |
|------|-----|
| データセット | `data/safe/{slice_name}/normal` の **test split** |
| 生成入力（プレフィックス） | 各行の **`pass_safe`**（フラグメントの SAFE 文字列）|
| slice_name / model_ver | `'brics'` / `'trained'`（`models/safe_gpt/trained/safe/brics/best_model`）|
| 生成対象件数 | test 先頭 **1000 件**（`n_generate` で調整可）|
| 生成方法 | `num_beams=50`, `num_return_sequences=50`, `max_length=200`, `do_sample=False`, `seed=42` |
| 妥当性判定 | `safe.decode` で SMILES 化 → `Chem.MolFromSmiles` が `None` でなければ valid |
| 出力 | `results/safe_gpt/{model_ver}/safe/{slice_name}/direct/predictions.csv`（+ error_logs.csv）|

### 生成・判定の流れ（参照: `_generate` / `_decode_safe`）

1. `pass_safe` を `tokenizer.get_pretrained()` でトークナイズ
2. 末尾 EOS トークンを除去（`input_ids[:, :-1]`。`token_type_ids` は除去）
3. `model.generate(..., num_beams=50, num_return_sequences=50, max_length=200)`
4. `batch_decode(skip_special_tokens=True)` で SAFE 文字列を取得
5. `safe.decode(seq, as_mol=False, fix=True, canonical=True, ignore_errors=True, remove_dummies=True)` で SMILES 化
6. `Chem.MolFromSmiles(smiles)` で妥当性判定。valid なら canonical SMILES、invalid なら `'invalid'`

## Plan

### Step 1: import の整理

- **Target file**: `src/gen_safe.py`（冒頭）
- **Changes**: de novo 用の暫定 import を、直接生成に必要な構成へ置き換える。
  - 標準: `os`, `subprocess`, `itertools`, `signal`
  - サードパーティ: `torch`, `pandas as pd`, `datasets`, `from tqdm import tqdm`,
    `import safe`, `from safe.tokenizer import SAFETokenizer`,
    `from safe.trainer.model import SAFEDoubleHeadsModel`, `from rdkit import Chem`
  - ローカル: `from func.utility import set_seed`
  - import 順（標準→サードパーティ→ローカル）を守る。
  - 注: `SAFEDesign` は不要になるため import しない。
- **Dependencies**: none

### Step 2: timeout_handler・生成ヘルパ・生成関数を実装（de novo 関数を置き換え）

- **Target file**: `src/gen_safe.py`（`__main__` の上）
- **Changes**: 暫定 `generate_de_novo` を削除し、以下を新規実装する。

  1. `timeout_handler`（`TimeoutError` を送出）
  2. `_generate_valid_smiles`（プレフィックス1件に対する直接生成＋RDKit判定。単一機能の private ヘルパ）
  3. `generate_from_model`（test を反復し CSV 出力する公開関数）

  ```python
  def timeout_handler(signum, frame) -> None:
      raise TimeoutError("Execution time exceeded the limit")


  def _generate_valid_smiles(
      model: SAFEDoubleHeadsModel,
      tokenizer,
      prefix: str,
      n_samples: int,
      num_beams: int,
      max_length: int,
      device: torch.device,
  ) -> list[str]:
      """SAFE プレフィックスからモデルで直接生成し、RDKit で妥当性判定した SMILES を返す。

      Args:
          model: 学習済み SAFE モデル。
          tokenizer: SAFETokenizer.get_pretrained() で得た HuggingFace トークナイザ。
          prefix: 生成のプレフィックスとなる SAFE 文字列（test の pass_safe）。
          n_samples: 生成本数（num_return_sequences）。
          num_beams: beam search のビーム数。
          max_length: 生成配列の最大長。
          device: 実行デバイス。

      Returns:
          list[str]: 各生成分子の canonical SMILES。妥当でなければ 'invalid'。
      """
      enc = tokenizer(prefix, return_tensors="pt")
      enc.pop("token_type_ids", None)
      model_inputs = {k: v[:, :-1].to(device) for k, v in enc.items()}  # 末尾 EOS を除去
      outputs = model.generate(
          **model_inputs,
          num_beams=num_beams,
          num_return_sequences=n_samples,
          max_length=max_length,
          do_sample=False,
          early_stopping=True,
      )
      safe_seqs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

      smiles_list = []
      for seq in safe_seqs:
          decoded = safe.decode(
              seq, as_mol=False, fix=True, remove_added_hs=True,
              canonical=True, ignore_errors=True, remove_dummies=True,
          )
          mol = Chem.MolFromSmiles(decoded) if decoded else None
          smiles_list.append(Chem.MolToSmiles(mol) if mol is not None else "invalid")
      return smiles_list


  def generate_from_model(
      model_path: str,
      dataset_dir: str,
      output_dir: str,
      n_generate: int = 1000,
      n_samples: int = 50,
      num_beams: int = 50,
      max_length: int = 200,
      timeout_sec: int = 60,
      random_seed: int = 42,
  ) -> pd.DataFrame:
      """test データの pass_safe をプレフィックスに、モデルから直接分子を生成し妥当性を判定する。

      Args:
          model_path: 学習済みモデルのパス。
          dataset_dir: load_from_disk で読む DatasetDict のパス（test split を使用）。
          output_dir: 生成結果 CSV の保存先ディレクトリ。
          n_generate: 生成対象とする test 行数（先頭から）。
          n_samples: 1 行あたりの生成本数。
          num_beams: beam search のビーム数。
          max_length: 生成配列の最大長。
          timeout_sec: 1 行あたりの生成タイムアウト（秒）。
          random_seed: 乱数シード。

      Returns:
          pd.DataFrame: 生成結果。カラムは
              ['target', 'full_safe', 'pass_safe', 'fragment', 'n_valid',
               'prediction_1', ..., f'prediction_{n_samples}']（各 prediction は canonical SMILES か 'invalid'）。
      """
      model = SAFEDoubleHeadsModel.from_pretrained(model_path)
      tokenizer = SAFETokenizer.from_pretrained(model_path).get_pretrained()
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = model.to(device)
      model.eval()
      print(f"Using device: {device}")

      test_dataset = datasets.load_from_disk(dataset_dir)["test"]
      n_generate = min(n_generate, test_dataset.num_rows)

      signal.signal(signal.SIGALRM, timeout_handler)
      results = []
      error_logs = []
      for i in tqdm(range(n_generate), desc="Generating molecules"):
          row = test_dataset[i]
          smiles, full_safe, pass_safe, fragment = (
              row["smiles"], row["full_safe"], row["pass_safe"], row["pass_fragments"],
          )
          set_seed(random_seed)
          signal.alarm(timeout_sec)
          try:
              gen_smiles = _generate_valid_smiles(
                  model, tokenizer, pass_safe, n_samples, num_beams, max_length, device,
              )
          except TimeoutError as e:
              gen_smiles = ["time_out"] * n_samples
              error_logs.append([smiles, pass_safe, "TimeoutError", str(e)])
          except Exception as e:
              gen_smiles = ["error"] * n_samples
              error_logs.append([smiles, pass_safe, type(e).__name__, str(e)])
          finally:
              signal.alarm(0)

          n_valid = sum(s not in ("invalid", "time_out", "error") for s in gen_smiles)
          results.append([smiles, full_safe, pass_safe, fragment, n_valid] + gen_smiles)

      columns = ["target", "full_safe", "pass_safe", "fragment", "n_valid"] + [
          f"prediction_{i + 1}" for i in range(n_samples)
      ]
      gen_df = pd.DataFrame(results, columns=columns)

      os.makedirs(output_dir, exist_ok=True)
      gen_df.to_csv(f"{output_dir}/predictions.csv")
      error_df = pd.DataFrame(
          error_logs, columns=["target", "pass_safe", "error_type", "error_message"],
      )
      error_df.to_csv(f"{output_dir}/error_logs.csv")

      total = n_generate * n_samples
      valid = int(gen_df["n_valid"].sum())
      print(f"Validity: {valid}/{total} ({valid * 100 / total:.2f}%) -> {output_dir}/predictions.csv")
      return gen_df
  ```
- **Dependencies**: after Step 1

### Step 3: `__main__` の修正（slice 名の変更 + gen_method 分岐）

- **Target file**: `src/gen_safe.py`（`if __name__=='__main__':`）
- **Changes**:
  - `slice_name = 'our_slice'` → `'brics'`、`model_ver = 'trained'` を新設
  - `model_path` を `f'{fd}/models/safe_gpt/{model_ver}/safe/{slice_name}/best_model'` に更新
  - `dataset_dir` を `f'{fd}/data/safe/{slice_name}/normal'` に更新
  - `gen_method` 既定を `'direct'` として分岐:
    - `if gen_method == 'direct':` → `generate_from_model(...)` を inline 実行
      （出力 `results/safe_gpt/{model_ver}/safe/{slice_name}/direct/`、`n_generate=1000` 等）
    - `else:`（従来 beam）→ 既存の subprocess 呼び出しと concat をそのまま移動（`pretrained`→`model_ver` のみ置換）
  - 末尾の暫定 de novo 呼び出しブロックは削除
  - コメントアウト済みスイープブロックは残す
- **Dependencies**: after Step 2

## 補足・留意点

- `model.generate` は best_model 同梱の `generation_config.json` を使用（`generation_config` 引数は渡さない）。
- 末尾 EOS 除去は `_generate` の実装に倣う（保持すると生成が乱れるため）。
- 実行環境は conda `safe`。`func` は editable install 済み。
- 実装後、`trained/safe/brics/best_model` で少数（例: n_generate=3）動作確認を行う。
