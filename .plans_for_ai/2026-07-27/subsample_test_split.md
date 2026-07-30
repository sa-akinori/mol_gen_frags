# Plan: 生成評価用にテストセットを間引く（RFFMG + SAFE）

- **Date**: 2026-07-27
- **Status**: implemented（Step 1-3 実施済み / Step 4 はユーザー指示により見送り）

## Overview

`normal` の生成に brics 5times で 62 時間、RFFMG 4条件の合計で約 358 時間（15日）かかる。
原因はテストセットの行数で、`normal` だけが全量（24.8万〜50.8万行）を使い、
他のロバストネス評価用スライスは 1万〜12万行の固定サブセットになっている。

テスト分割（分子単位 2.5%）はそのまま維持し、**生成評価に使う分子と行だけを間引く**。
分割比を変えると訓練データが変わりモデル自体が別物になるため、分割には触らない。

### 方針（ユーザー決定）

1. **テスト分割の全分子から無作為に 20,000 分子を抽出**する。
   「断片化パターンが5通り以上ある分子」に候補を絞ると大きい分子に偏る
   （brics 5times で重原子数の平均が 24.4 → 29.7）ため、候補は絞らない。
2. 各分子につき **最大5行**（`min(5, 実際の行数)`）を採用する。
3. **RFFMG と SAFE に同じ 20,000 分子を適用**し、表現手法の比較の土台を揃える。
4. **SAFE のデータセット生成は `sampling_num == 5` のときだけ**実行する。
   SAFE は sampling_num の概念を持たず `data/safe/{frag}/normal` に固定出力するため、
   他の sampling_num で実行すると別の分割で上書きされてしまう。
5. 間引くのは **test のみ**。train / val は全量のまま（学習に影響させない）。

### 実測値（RFFMG）

| 条件 | 現在の行数 | 間引き後 | 推定時間 |
|---|---|---|---|
| brics 5times | 248,724 | 82,441 | 20.6 時間 |
| brics 10times | 352,920 | 85,459 | 21.3 時間 |
| rc_cms 5times | 328,116 | 90,974 | 22.7 時間 |
| rc_cms 10times | 507,879 | 92,872 | 23.2 時間 |
| **合計** | **1,437,639（358時間）** | **351,746（87.8時間 = 3.7日）** | |

推定時間は brics 5times の実績（248,724 行 = 62 時間、1行あたり 0.90 秒）からの換算。
`frag_order` は間引き後も 10,000 件の抽出に必要な候補（3断片以上の行）が
45,531〜66,488 行残るため成立する。

### 再現性の検証結果（実施済み）

`full_dataset.csv` から `make_datasets.py:63-67` と同じ手順で分割を再導出し、
ディスク上のファイルと SMILES 集合を照合した。

| | train | val | test |
|---|---|---|---|
| RFFMG brics | 1,717,908 一致 | 45,208 一致 | 45,209 一致 |
| RFFMG rc_cms | 1,772,900 一致 | 46,655 一致 | 46,656 一致 |
| SAFE rc_cms | 一致 | 一致 | 一致 |
| SAFE brics | 3,610 少ない | 5 少ない | 一致 |

- **再実行しても同じ分割になる**ことを確認済み。`tr_smiles` / `val_smiles` / `te_smiles` は
  L67 で一度だけ計算され RFFMG と SAFE が共有するため、両者で同一。
- SAFE brics の差は `drop_duplicates(subset='full_safe')`（L93-94、test には未適用）によるもの。
  欠けた 3,610 分子は全て `safe_smiles.csv` に存在しており、SAFE 変換での欠落はゼロ。
  決定的な処理なので再実行で同じ結果になる。
- **SAFE の test は分割と完全一致**（dedup 未適用のため）。よって同じ `te_smiles` を
  適用すれば RFFMG と SAFE で同一の分子集合を評価できる。

### 波及範囲

| スライス | 入力元 | 間引きの影響 |
|---|---|---|
| `frag_order` | `normal/test.source`（インデックス指定） | **作り直しになる** |
| `frag_num` / `dup_frags` / `attach_point_num` | `normal/train.source` + `unique_frags.csv` | なし |
| SAFE `data/safe/{frag}/normal` | `te_smiles` | **test が間引かれる** |

`evaluation.py:78-82` は `frag_order` の評価で `random_get_ids.pkl` を使って
`normal` の結果から同じ行を引くため、`normal` と `frag_order` は行が揃っている必要がある。
`evaluation_func.py:52` は `test.source` と `predictions.csv` を**行番号で連結**するため、
間引き後の `test.source` は対応する `predictions.csv` とセットで扱う必要がある。

`gen_rffmg.py` / `gen_safe.py` の変更は不要（パスは変わらず中身が減るだけ）。

## Plan

### Step 1: 定数とヘルパー関数、引数を追加する

- **Target file**: `src/make_datasets.py`
- **Changes**:
  - モジュールレベル（`unique_f_num` などのラムダ定義の近く、L43-44 付近）に定数を追加する。

    ```python
    SAFE_SAMPLING_NUM = 5  # SAFE は sampling_num を持たず data/safe/{frag}/normal に固定出力するため、この値のときだけ生成する
    ```

  - 既存のヘルパー関数群（`save_file` 〜 `setrffmgAtoms`）の後に、型ヒントと
    Google style docstring 付きの関数を1つ追加する。

    ```python
    def cap_rows_per_molecule(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        """1分子あたりの行数を上限まで間引く。

        Args:
            df: ``smiles`` 列を持つデータフレーム。
            max_rows: 1分子あたりに残す最大行数。

        Returns:
            各 ``smiles`` について先頭 ``max_rows`` 行だけを残したデータフレーム。
            元の行順を保持し、インデックスは振り直す。
        """
        return df.groupby('smiles', sort=False).head(max_rows).reset_index(drop=True)
    ```

  - argparse（L48-51）に引数を2つ追加する。既存の書き方（シングルクォート、1行の `help=`）に合わせる。

    ```python
    parser.add_argument('--test_mol_num', type=int, default=None, help='number of test molecules kept for generation evaluation (default: None, keep all)')
    parser.add_argument('--test_rows_per_mol', type=int, default=5, help='max fragmentation patterns kept per test molecule (default: 5)')
    ```

  - 既定値を `None` にすることで、引数なしの実行は現在と同じ全量出力になる（後方互換）。
- **Dependencies**: none

### Step 2: `te_smiles` を間引き、行数上限を RFFMG と SAFE の両方に適用する

- **Target file**: `src/make_datasets.py`
- **Changes**:
  - 分割直後（L67 の後）に、`te_smiles` そのものを間引く。ここに置くことで
    RFFMG と SAFE の両方が同じ分子集合を使う。

    ```python
    # Subsample the test molecules: generating on the full test split is prohibitively slow.
    # Applied to te_smiles itself so that RFFMG and SAFE are evaluated on the same molecules.
    if args.test_mol_num is not None:
        random.seed(0)
        te_smiles = random.sample(sorted(te_smiles), args.test_mol_num)
    ```

  - `sorted()` を挟むのは、`unique()` の順序に依存せず同じ seed で同じ分子集合を得るため。
  - `rffmg_te`（L72）の直後に行数上限を適用する。

    ```python
    if args.test_mol_num is not None:
        rffmg_te = cap_rows_per_molecule(rffmg_te, args.test_rows_per_mol)
    ```

  - `safe_te`（L95）の直後にも同様に適用する。

    ```python
    if args.test_mol_num is not None:
        safe_te = cap_rows_per_molecule(safe_te, args.test_rows_per_mol)
    ```

  - `safe_tr` / `safe_val` / `rffmg_tr` / `rffmg_val` には適用しない（train/val は全量維持）。
  - L133 で `normal/test.source` を読む `frag_order` は、間引き後のファイルから自動的に作られる。
- **Dependencies**: after Step 1

### Step 3: SAFE の生成を `sampling_num == 5` のときだけ実行する

- **Target file**: `src/make_datasets.py`
- **Changes**:
  - SAFE ブロック全体（L92 の `# safe` から L121 の debug 保存まで）を条件で囲む。

    ```python
    # SAFE datasets have no sampling_num level (they are written to data/safe/{frag}/normal),
    # so only build them for the canonical sampling_num to avoid overwriting with another split.
    if args.sampling_num == SAFE_SAMPLING_NUM:
        # safe
        safe_tr  = ...
        （以下、既存の L93-121 をインデントして格納）
    ```

  - 既存の処理内容は変更せず、インデントと条件の追加のみ。
- **Dependencies**: after Step 2

### Step 4: `run_cpu.sh` を更新する（任意）

- **Target file**: `run_cpu.sh`
- **Changes**:
  - 現状 `~/Research/mol_gen_frags_copy/src/make_datasets.py` という別リポジトリを指しており、
    `--sampling_num` も渡していない（sampling_num 導入前のまま）。実行できる状態に直したうえで、
    間引き引数を渡す。

    ```bash
    FRAG_NAME="brics"    # "brics" or "rc_cms"
    SAMPLING_NUM=5       # 5 or 10 (uses the data/rffmg/<frag>/<N>times_sampling slice)

    ~/miniconda3/envs/safe/bin/python ~/Research/mol_gen_frags/src/make_datasets.py \
        --frag_method ${FRAG_NAME} --sampling_num ${SAMPLING_NUM} --test_mol_num 20000
    ```

  - 変数の書き方とコメント文言は `src/train_model/run_rffmg.sh` に揃える。
  - **このステップは独立**。不要であれば外してよい。
- **Dependencies**: after Step 3

### Step 5: 動作確認

- **Target file**: 変更なし（確認のみ）
- **Changes**:
  - `python -m py_compile src/make_datasets.py` / `bash -n run_cpu.sh`
  - `python src/make_datasets.py --help` で新しい引数2つが出ること。
  - `SAFE_SAMPLING_NUM` の条件により、`--sampling_num 10` では SAFE ブロックが
    実行されないことをコード上で確認する。
  - **本実行はしない**（後述のとおり非常に重い）。代わりに間引きロジックだけを
    既存の `normal/test.*` と `safe_smiles.csv` に対して切り出して実行し、
    次の2点を確認する。
    - RFFMG の行数が上表と一致すること（brics 5times = 82,441 行）。
    - **SAFE の test 行数が RFFMG と一致すること**。両者は同じ分子集合・
      同じ上限で間引くため一致するはずだが、1分子あたりの行数が
      `safe_smiles.csv` と `full_dataset.csv` で揃っているかは未検証のため、
      ここで確かめる。
  - 引数なしの場合に `te_smiles` / `rffmg_te` / `safe_te` が変更されない（後方互換）ことを確認する。
- **Dependencies**: after Step 3

## 実施結果（2026-07-27）

Step 1-3 を `src/make_datasets.py` に実装。Step 4（`run_cpu.sh`）はユーザー指示により見送り
（実行はユーザー側で行うため）。Step 5 の確認は以下のとおり全て通過。

- `py_compile` 通過。`--help` に `--test_mol_num` / `--test_rows_per_mol` が出ることを確認。
- SAFE ブロックのガード範囲を行番号で確認（L123 の `if` から L156 の `save_to_disk` まで内側、
  L158 の `frag_order` 生成以降は外側）。`--sampling_num 10` では SAFE の書き出しがスキップされる。
- **間引き後の行数を実データで照合し、RFFMG と SAFE が完全一致することを確認**
  （計画時点で唯一未検証だった前提）。

  | | RFFMG test | SAFE test | |
  |---|---|---|---|
  | brics | 82,441 行（20,000 分子） | 82,441 行（20,000 分子） | 一致 |
  | rc_cms | 90,974 行（20,000 分子） | 90,974 行（20,000 分子） | 一致 |

### 再現性の追加検証（行レベル）

計画作成時の照合は SMILES 集合の比較にとどまっていたため、行単位でも確認した。
`full_dataset.csv` をファイルの行順のままなめて各 split に振り分け、
`make_datasets.py:75-77` と同じ形式でバイト列を組み立て、MD5 を比較。

**brics / rc_cms の train / val / test × source / target の計12ファイルすべてで MD5 が一致。**
再実行すれば既存ファイルとバイト単位で同一のものが再生成される。
この保証は `full_dataset.csv` が変更されないことが前提（`rffmg_frags.py` を再実行すると切れる）。

## Notes / 実行時の注意

1. **実行前にバックアップを取ること**
   - `data/safe/{brics,rc_cms}/normal` は現在の SAFE 学習済みモデルの学習データそのもの。
     再実行で上書きされる（内容は再現される見込みだが保険をかける）。
   - `data/rffmg/{frag}/{N}times_sampling/normal` も同様。
   ```bash
   cp -r data/safe/brics/normal data/safe/brics/normal.bak
   cp -r data/safe/rc_cms/normal data/safe/rc_cms/normal.bak
   ```
2. **`make_datasets.py` の再実行そのものが重い**
   - L157 / L188 / L248 で `train.source` の全行（brics 5times で 9,459,145 行）を
     `canonical_smiles` にかける処理が**3回**走る。RDKit の正準化を約2,800万回行う計算で、
     数時間規模のCPU時間がかかる。`run_cpu.sh` が PBS の CPU キュー用なのはこのため。
3. **再生成が必要になる結果**
   - RFFMG `normal`: 4条件 × 約21〜23時間 = 約88時間
   - RFFMG `frag_order`: 4条件 × 12.5時間 = 約50時間（間引きに伴い作り直し）
   - SAFE `normal`: 所要時間は未見積もり。`generation_safe_func.py` は1分子ずつ
     60秒タイムアウト付きで回す作りで、RFFMG とは時間特性が異なる。必要なら別途見積もる。
   - `dup_frags` / `frag_num` / `attach_point_num` は入力もシードも変わらないため既存結果が使える。
4. **論文への記載**
   - データ分割（95/2.5/2.5）と評価プロトコル（テスト分割から無作為抽出した 20,000 分子、
     各最大5断片化）は分けて書く。抽出はシード固定で再現可能。
   - 代表性を示すなら、抽出した20,000分子と全テスト分割の物性分布（MW / LogP / QED / TPSA）を
     `evaluation_func.py` の `calcPhysicProp`(L576) と `jensenshannon`(L648) で比較できる。
   - **5times と 10times のテスト分子集合はほぼ重ならない**（共通 2.6% / 3.4%）。
     `unique_smiles` の並び順が `full_dataset.csv` の行順に依存するため、同じ seed でも
     別の分割になる。サンプリング数の効果を論じる際は説明が要る。
5. **スコープ外**: `evaluation.py` が sampling 階層に対応していない件は今回は触らない。
   評価を回す前に別途対応が必要。
