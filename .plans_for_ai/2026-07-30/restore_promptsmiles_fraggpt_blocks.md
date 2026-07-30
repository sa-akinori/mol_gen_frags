# Plan: promptsmiles/fraggpt のデータ生成・評価接続コードの復元

- **Date**: 2026-07-30
- **Status**: pending-approval

## Overview

2026-07-30 21:06 の他マシンからのデータ転送（mtime 保持）が `data/` だけでなく `src/` も上書きし、未コミットだった作業が失われた。

失われたもの:

| ファイル | 失われた内容 |
|---|---|
| `src/make_datasets.py` | `data/promptsmiles/` と `data/fraggpt/` を出力するブロック（328行→299行） |
| `src/evaluation.py` | `promptsmiles`/`fraggpt` の分岐と `--gen_method` 引数（HEAD に巻き戻り） |

失われていないもの（未追跡ファイルは無傷）: `src/gen_mols/gen_promptsmiles.py`, `src/gen_mols/gen_fraggpt.py`,
`src/train_model/train_promptsmiles.py`, `src/train_model/train_fraggpt.py`,
`src/func/generation_fraggpt_func.py`, `src/func/fragment_for_fraggpt.py`, および各 `.sh`。

つまり **実装本体は生きていて、データ生成側と評価側の接続部分だけが欠けている**。本計画はその接続部分を復元する。

復元内容の出典は、転送前（2026-07-30 20:2x）に読み取った両ファイルの実データ。
`src/evaluation.py` については `git diff --stat` が示した «44 ++-» と、差分内の削除12行＋追加32行＝44行が一致するため、差分は完全である。

ベースは**転送で入ってきた新版**（`SAFE_SAMPLING_NUM` 定数化済み）とし、そこに欠けたブロックを足す。

## Plan

### Step 1: make_datasets.py に promptsmiles/fraggpt 出力ブロックを復元

- **Target file**: `src/make_datasets.py`
- **Dependencies**: none

**1-a.** 105-106行のコメントを、promptsmiles/fraggpt も対象であるよう修正する。

変更前:
```python
    # SAFE datasets have no sampling_num level (they are written to data/safe/{frag}/normal),
    # so only build them for the canonical sampling_num to avoid overwriting with another split.
```

変更後:
```python
    # SAFE, PromptSMILES and FragGPT datasets have no sampling_num level (they are written to
    # data/{safe,promptsmiles,fraggpt}/{frag}/normal), so only build them for the canonical
    # sampling_num (the --sampling_num default) to avoid overwriting with another split.
```

**1-b.** 140行 `debug_os_dataset.save_to_disk(f'{fd}/safe/{frag_method}/normal/debug')` の直後（`if args.sampling_num == SAFE_SAMPLING_NUM:` ブロックの内側、インデント8スペース）に以下を挿入する。

```python

        # promptsmiles: plain SMILES (one molecule per line) reusing the split shared with RFFMG and SAFE.
        # The test set is the already subsampled te_smiles, so the evaluated molecules are identical.
        promptsmiles_dir = f'{fd}/promptsmiles/{frag_method}/normal'
        os.makedirs(f'{promptsmiles_dir}/debug', exist_ok=True)
        for smiles, name in zip([tr_smiles, val_smiles, te_smiles], ["train", "val", "test"]):
            save_file("\n".join(smiles) + "\n", f'{promptsmiles_dir}/{name}.smi')
            save_file("\n".join(smiles[:10000]) + "\n", f'{promptsmiles_dir}/debug/{name}.smi')
            print(f'promptsmiles {name}: {len(smiles)} molecules -> {promptsmiles_dir}/{name}.smi')

        # fraggpt: FU-SMILES corpus (one fragmentation pattern per line) on the same split.
        # `full_fragments` already is FU-SMILES: BRICSFragmentize / RandomFragmentize label every
        # cut bond with a pair of [i*] dummy atoms, so no extra fragmentation step is needed.
        # No test file is written here: gen_fraggpt.py prompts with the SAFE test split, so that
        # RFFMG, SAFE, PromptSMILES and FragGPT all start from identical fragment sets.
        fraggpt_dir = f'{fd}/fraggpt/{frag_method}/normal'
        os.makedirs(f'{fraggpt_dir}/debug', exist_ok=True)
        for data, name in zip([rffmg_tr, rffmg_val], ["train", "val"]):
            fusmiles = data['full_fragments'].drop_duplicates().tolist()
            save_file("\n".join(fusmiles) + "\n", f'{fraggpt_dir}/{name}.smi')
            save_file("\n".join(fusmiles[:10000]) + "\n", f'{fraggpt_dir}/debug/{name}.smi')
            print(f'fraggpt {name}: {len(fusmiles)} FU-SMILES -> {fraggpt_dir}/{name}.smi')

        # Molecules behind the training corpus, read by src/evaluation.py for the novelty check.
        # Identical to the promptsmiles train.smi, but written independently to keep the two
        # baselines free of any dependency on each other.
        save_file("\n".join(tr_smiles) + "\n", f'{fraggpt_dir}/train.target')
        save_file("\n".join(tr_smiles[:10000]) + "\n", f'{fraggpt_dir}/debug/train.target')
        print(f'fraggpt train molecules: {len(tr_smiles)} -> {fraggpt_dir}/train.target')
```

**1-c.** モジュールレベル定数 `SAFE_SAMPLING_NUM` を削除する。

理由: 参照箇所は107行の1箇所のみで、しかも同じ値 `5` が52行の `--sampling_num` の default にも
書かれていて二重定義になっている。argparse の default を正本にすれば値の定義が1箇所に集約され、
Step 1-b でこの分岐が SAFE 以外（PromptSMILES/FragGPT）も制御するようになっても名称と実態の
齟齬が生じない。

46行を削除する（前後の空行は1行だけ残す）:
```python
SAFE_SAMPLING_NUM = 5 # SAFE は sampling_num を持たず data/safe/{frag}/normal に固定出力するため、この値のときだけ生成する
```

107行を置換する。

変更前:
```python
    if args.sampling_num == SAFE_SAMPLING_NUM:
```

変更後:
```python
    if args.sampling_num == parser.get_default('sampling_num'):
```

`parser` は同じ `if __name__=='__main__':` スコープ内（50行）で定義済みなので参照可能。
削除の理由（sampling_num 階層を持たない出力のみを対象とすること）は 1-a で書き換えるコメントに
含まれるため、情報は失われない。

### Step 2: evaluation.py に promptsmiles/fraggpt 分岐と --gen_method を復元

- **Target file**: `src/evaluation.py`
- **Dependencies**: none（Step 1 とは独立）

**2-a.** 14-15行を置換する。

変更前:
```python
    parser.add_argument('--model_name', type=str, choices=['t5chem', 'safe_gpt', 'gpt'],
                        help='Model name: t5chem / safe_gpt / gpt (RFFMG-GPT) (default: t5chem)')
```

変更後:
```python
    parser.add_argument('--model_name', type=str, choices=['t5chem', 'safe_gpt', 'gpt', 'promptsmiles', 'fraggpt'],
                        help='Model name: t5chem / safe_gpt / gpt (RFFMG-GPT) / promptsmiles / fraggpt (default: t5chem)')
```

**2-b.** 21行（`--additional_path` の help 行）の直後に以下を追加する。

```python
    parser.add_argument('--gen_method', type=str, default=None, choices=['beam', 'sampling'],
                        help='Decoding scheme segment of the results path; defaults to the one the model was generated with')
```

**2-c.** 26-33行を置換する。

変更前:
```python
    # RFFMG-GPT (model_name='gpt') keeps its own results path (str_name/model_dir) but
    # reuses the T5Chem-format reader in evaluation_func (arc_name).
    if model_name == 'safe_gpt':
        str_name, model_dir, arc_name = 'safe', 'gpt', 'safe_gpt'
    elif model_name == 'gpt':
        str_name, model_dir, arc_name = 'rffmg', 'gpt', 't5chem'
    else:  # t5chem
        str_name, model_dir, arc_name = 'rffmg', 't5chem', 't5chem'
```

変更後:
```python
    # RFFMG-GPT (model_name='gpt') and PromptSMILES keep their own results path
    # (str_name/model_dir/gen_method) but reuse the T5Chem-format reader in evaluation_func
    # (arc_name). The default decoding scheme is beam search, except for PromptSMILES which is
    # published with multinomial sampling; --gen_method overrides it when both schemes were run.
    if model_name == 'safe_gpt':
        str_name, model_dir, arc_name, default_gen_method = 'safe', 'gpt', 'safe_gpt', 'beam'
    elif model_name == 'gpt':
        str_name, model_dir, arc_name, default_gen_method = 'rffmg', 'gpt', 't5chem', 'beam'
    elif model_name == 'promptsmiles':
        str_name, model_dir, arc_name, default_gen_method = 'promptsmiles', 'gpt', 't5chem', 'sampling'
    elif model_name == 'fraggpt':
        str_name, model_dir, arc_name, default_gen_method = 'fraggpt', 'gpt', 't5chem', 'beam'
    else:  # t5chem
        str_name, model_dir, arc_name, default_gen_method = 'rffmg', 't5chem', 't5chem', 'beam'
    gen_method  = args.gen_method or default_gen_method
```

**2-d.** 43行 `additional_path = 'normal'` の後、45行 `else:  # t5chem or gpt ...` の前に、以下2分岐を挿入する。

```python
    elif model_name == 'promptsmiles':
        # Plain-SMILES corpus (one molecule per line) and the prompts written by gen_promptsmiles.py.
        tr_file_name  = f'{BASEPATH}/data/promptsmiles/{frag_method}/normal/train.smi'
        testInputfile = f'{BASEPATH}/data/promptsmiles/{frag_method}/{additional_path}/test.source'

    elif model_name == 'fraggpt':
        # Molecules behind the FU-SMILES corpus and the prompts written by
        # generation_fraggpt_func.py (the shared test split, with unlabeled `*`).
        tr_file_name  = f'{BASEPATH}/data/fraggpt/{frag_method}/normal/train.target'
        testInputfile = f'{BASEPATH}/data/fraggpt/{frag_method}/{additional_path}/test.source'
```

**2-e.** 59行の結果パスの `beam` を `{gen_method}` に置換する。

変更前: `outfd = f'{BASEPATH}/results/{str_name}/{model_dir}/{model_ver}/{frag_method}/beam/{additional_path}'`

変更後: `outfd = f'{BASEPATH}/results/{str_name}/{model_dir}/{model_ver}/{frag_method}/{gen_method}/{additional_path}'`

**2-f.** 63行を `outfd` の再利用に置き換える（重複した長い f-string の解消）。

変更前: `file_name = f'{BASEPATH}/results/{str_name}/{model_dir}/{model_ver}/{frag_method}/beam/{additional_path}/predictions.csv'`

変更後: `file_name = f'{outfd}/predictions.csv'`

**2-g.** 78行（`frag_order` 分岐内）の `beam` を `{gen_method}` に置換する。

変更前: `outfd  = f'{BASEPATH}/results/{str_name}/{model_dir}/{model_ver}/{frag_method}/beam'`

変更後: `outfd  = f'{BASEPATH}/results/{str_name}/{model_dir}/{model_ver}/{frag_method}/{gen_method}'`

### Step 3: 構文チェックのみ実施（データ生成は行わない）

- **Target file**: なし（検証のみ）
- **Dependencies**: after Step 1, Step 2
- **Changes**: `python -m py_compile src/make_datasets.py src/evaluation.py` で構文を確認する。

`make_datasets.py` の実行は本計画に含めない。理由は、実行すると
`data/rffmg/{frag}/5times_sampling/normal/*` と `data/safe/{frag}/normal` も同時に書き直され、
2026-07-30 21:06 に転送されたばかりのファイルを上書きすることになるため。

実行する場合に期待される出力（今回の検証で確定した分割サイズに基づく）:

| ファイル | brics | rc_cms |
|---|---|---|
| `data/promptsmiles/{frag}/normal/train.smi` | 1,717,908 行 | 1,772,900 行 |
| `data/promptsmiles/{frag}/normal/val.smi` | 45,208 行 | 46,655 行 |
| `data/promptsmiles/{frag}/normal/test.smi` | 20,000 行 | 20,000 行 |
| `data/fraggpt/{frag}/normal/train.target` | 1,717,908 行 | 1,772,900 行 |

`data/fraggpt/{frag}/normal/{train,val}.smi` は `full_fragments` の重複排除後の行数なので事前に確定できない。

## Notes

- 本計画は `src/` 配下のみを変更する。`data/` は変更しない。
- 旧レイアウトの残骸ディレクトリ（`data/safe/{frag}/debug`, `data/rffmg/{frag}/5times_sampling/debug`）は
  2026-07-30 に削除済み（計 35MB、コードからの参照なしを確認済み）。本計画の対象外。
- `evaluation.py` の `data/rffmg/{frag}/normal/...` に sampling_num 階層が無い問題は、
  ユーザー判断により本計画では扱わない。
