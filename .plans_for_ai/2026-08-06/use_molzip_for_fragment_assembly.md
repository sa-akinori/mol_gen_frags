# Plan: 断片の再結合を RDKit の molzip に置き換える

- **Date**: 2026-08-06
- **Status**: approved

## Overview

`assemble_fragments_with_reason` は、ダミー原子を削除してから `AddBond` で結合を張り直している。
RDKit のキラルタグは「その原子に結合が張られた順序」を基準とする相対的な指定なので、
この手順は**不斉中心を反転させる**。

### 機構（実データ例2で追跡した結果）

断片 `[1*]C(=O)[C@@H]([2*])C[5*]` の不斉炭素で、隣接順は次のように変化する。

```
隣接順（元）   : [カルボニルC, ダミー[2*], CH2]     ← ダミーは2番目
AddBond 後     : [カルボニルC, ダミー[2*], CH2, N]  ← 置き換え先 N は末尾にしか追加できない
RemoveAtom 後  : [カルボニルC,             CH2, N]  ← N は末尾のまま
キラルタグ     : CHI_TETRAHEDRAL_CW（不変）
```

本来 N はダミーがいた2番目に入るべきだが末尾に入るため、2番目と3番目が入れ替わる。
互換1回＝奇順列なので、同じタグが逆の立体を指す（`[C@H]` → `[C@@H]`）。

原因は RDKit の API の制約で、`AddBond` は末尾にしか追加できず `RemoveAtom` は抜くだけ。
結果として「k番目にあったものを末尾へ移動」する操作になり、移動距離
`(隣接数 − 1 − k)` が奇数なら反転、偶数ならたまたま正しくなる。

### 実測した影響（学習データ各1万件を組み立て直して元の分子と比較）

| | 現行の手組み | **molzip** | SAFE（参考） |
|---|---:|---:|---:|
| brics | 84.04% | **92.58%** | 93.12% |
| rc_cms | 85.83% | **100.00%** | 100.00% |

現行の不一致の内訳は brics で「立体が減った 728 / **向きが違う 868**」、
rc_cms で「立体が減った 477 / **向きが違う 940**」。半分以上が反転（誤った立体異性体の生成）。

**rc_cms は表現の限界がゼロ**で、現在の 14% はすべて実装由来。brics の残り約7%は本物の
表現限界であり、SAFE も同じ壁（93.12%）に当たる。molzip と SAFE がほぼ一致することが裏付け。

### 他手法との比較

- **SAFE**: 原子レベルの手術をしない。`safe.decode()` は結合点の数字を環閉じ結合に戻す
  文字列レベルの変換で、RDKit は最終形を1回パースする。よって反転しない。
- **PromptSMILES**: 組み立て工程が無い。モデルが SMILES 全体を書き、`Chem.MolFromSmiles` で
  妥当性を見るだけ。
- **FragGPT だけが原子を削除・再結合している。**

### 事前検証済みの事項

- `rdmolops.molzip` は `MolzipParams.label = MolzipLabel.Isotope` で同位体ラベル付きダミーを
  結合できる。二重結合・三重結合も正しく復元される
  （`[1*]=C(C)C.[1*]=C1CCCCC1` → `CC(C)=C1CCCCC1`、`[1*]#CC.[1*]#CC` → `CC#CC`）。
- 組み立て後の `Chem.AssignStereochemistry(cleanIt=True, force=True)` は**残すべき**
  （brics 92.55% → 92.58%、rc_cms は不変）。
- **断片は個別にパースしてから `CombineMols` する現行方針を維持すること。**
  文字列を一括パースすると環閉じ数字が点をまたいで結合する
  （`C1CC.C1CC` → `CCCCCC`）。この危険は実在する。
- `CombineMols` + `molzip` 方式で brics 92.58% / rc_cms 100.00%、失敗0件を確認済み。

### スコープ外

- 評価側の立体比較（`isomericSmiles` の扱い）。表現の限界として受け入れる方針
- SAFE / PromptSMILES / RFFMG の変更

## Plan

### Step 1: 結合操作を molzip に置き換える

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**:
  - import に `from rdkit.Chem import rdmolops` を追加する。
  - モジュール冒頭に molzip の設定を1度だけ作る定数を置く。既存の定数と同じ体裁に揃えること。
    ```python
    MOLZIP_PARAMS = rdmolops.MolzipParams()
    MOLZIP_PARAMS.label = rdmolops.MolzipLabel.Isotope
    ```
  - `assemble_fragments_with_reason` の以下は**そのまま残す**。
    - 断片を個別にパースして `CombineMols` する処理（46-56行目相当）
    - `dummies_by_label` の構築と `unmatched_dummy` 判定
    - `_dummy_bond` による None 判定（`unmatched_dummy`）
    - `first_neighbor == second_neighbor` / 既に結合済み の判定（`invalid_connection`）
    - 結合次数の一致判定（`bond_order_mismatch`）
    - `Chem.SanitizeMol` と `Chem.AssignStereochemistry(cleanIt=True, force=True)`
    - `multiple_components` 判定
  - 以下を**削除**する。
    - `rwmol = Chem.RWMol(combined)`
    - `rwmol.AddBond(first_neighbor, second_neighbor, bond_type)`
    - ダミー原子を降順に削除するループとその直前のコメント
    - 一致確認後に `bond_type` を決める代入（molzip が結合次数を決めるため不要になる）
  - 検査ループは `rwmol` ではなく `combined` を参照するように変える
    （`_dummy_bond(combined, ...)` / `combined.GetBondBetweenAtoms(...)`）。
    検査は結合操作の**前**に全ラベル分を走らせ、1つでも問題があればその理由を返すこと。
  - 検査を通過したら `rdmolops.molzip(combined, MOLZIP_PARAMS)` で結合し、
    その結果に対して既存の `Chem.SanitizeMol` / `Chem.AssignStereochemistry` を適用する。
    molzip が例外を投げた場合も既存の `except` で `sanitize_failure` を返してよい。
  - docstring を更新する。
    - 現在の「Dummy atoms sharing an isotope label are the two ends of one broken bond: the atoms
      they are attached to are bonded again and the dummies are deleted」以下の説明を、
      molzip に委ねる旨に書き換える。
    - **なぜ molzip を使うのかを1行で残すこと**（ダミーを削除して結合を張り直すと
      隣接順が変わり不斉中心が反転するため）。これは非自明な制約なのでコメントに値する。
    - 結合次数についての説明は「両端の次数が一致していることを検査し、実際の結合は
      molzip が行う」旨に改める。
    - `Returns` の失敗理由一覧は変更しない（理由の集合は変わらない）。
- **Dependencies**: none

### Step 2: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 構文チェックと import 確認
  - **学習データ各1万件で、元の分子との完全一致率が brics 92.58% / rc_cms 100.00% に
    改善すること**（変更前は 84.04% / 85.83%）
  - 骨格（`isomericSmiles=False`）が変更前後で 100% 一致すること
    （立体以外は何も変わっていないこと）
  - 失敗理由が従来どおり返ること
    - `unmatched_dummy`: `[1*].[1*]`、`[1*]C.[1*]`、`C[1*]C.C[1*]C`、`[1*]CC.`
    - `invalid_connection`: `C([1*])[1*]`、`[1*]C[2*].[1*]C[2*]`
    - `bond_order_mismatch`: `[1*]=C(C)C.[1*]#CC` と `[1*]#CC.[1*]=C(C)C`（順序非依存）
    - `parse_failure`: `[1*]CC.[1*]O.C1CC`、`C1CC.C1CC`、`""`
  - 二重結合・三重結合の切断が復元されること
  - 末尾 `.` / `..` 連続を含む入力が従来どおり組み上がること
  - `label_attachment_points` の挙動が変わっていないこと
- **Dependencies**: after Step 1
