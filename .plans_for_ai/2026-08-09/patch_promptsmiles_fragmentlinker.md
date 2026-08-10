# Plan: promptsmiles の FragmentLinker のバグを修正する

- **Date**: 2026-08-09
- **Status**: **withdrawn（一度適用したのち取り下げ・完全に復元済み）**

## 取り下げの理由

一度パッチを当てて検証したのち、**上流の実装のまま使う**と決めた。

- **SAFE の既存パッチとは性質が違う。** SAFE の4箇所は「動かすための配管」
  （`best_model` の保存先、transformers の API 互換、EarlyStopping の追加）で、手法の
  アルゴリズムは変えていない。今回のものは `FragmentLinker` が断片を置くかどうかの判定、
  すなわち**手法のアルゴリズムそのもの**。
- 比較対象として使う以上、**配布されている実装のまま**であるべき。パッチを当てると
  「PromptSMILES」ではなく「修正した PromptSMILES」との比較になり、論文でその差分を
  説明する必要が生じる。
- **実測で指標に差が出なかった。** 同一断片を含む3断片以上の linking 行6行で、生成結果は
  5/6行で変わったが、プロンプト断片の充足数は **19/19 → 19/19 で不変**。
  同じ断片が2つある場合、1つスキップされても部分構造照合は成立するため。
- site-packages のパッチは再インストールで消え、別マシンにも適用が必要。
  実測で効果が確認できない変更のためにこのリスクを負う理由がない。

**復元の確認**: `samplers.py` はバックアップと `diff -q` で完全一致、
796行目・865行目とも `frag_indexes` に戻っている。README への追記も削除済み。

以下は調査記録として残す。

## Overview

外部ライブラリ `promptsmiles` の `FragmentLinker._batch_sample` に、**指定した断片が
最終分子から抜け落ちるバグ**がある。`gen_promptsmiles.py` は `batch_prompts=True` で
このコードパスを通るため、生成結果が直接影響を受ける。

### FragmentLinker の処理

複数の断片 A・B・C をつなぐとき、断片を1つずつ順にプロンプトへ足していく。

```
1. 断片A をプロンプトにして生成
2. 次は断片B。その前に「モデルがもう B に相当する構造を作っていないか」を確認する
   - 作っていれば B をプロンプトに入れない（スキップ）
   - 作っていなければ B をプロンプトに足して続きを生成
3. 断片C も同様
```

この確認では、生成分子の中で B と一致した部分構造が **本当に新しく作られた B** なのか、
**すでに置いた断片A の一部を誤認しているだけ** なのかを区別する必要がある。
そのために「その原子はすでに断片として計上済みか」を調べている。

### バグの内容

`_single_sample`（単一プロンプト経路）では `frag_indexes` が整数のフラットなリスト。

```python
frag_indexes = list(range(len(prompt_tokens)))                      # samplers.py:621
if not any([atom_map[aidx] in frag_indexes for aidx in match]):     # :649, :701  正しい
```

`_batch_sample`（バッチ経路）では **行ごとのリストのリスト**になる。

```python
frag_indexes = []                                                   # samplers.py:747
frag_indexes.append(list(range(len(f0_tokens))))                    # :756
if not any([atom_map[aidx] in frag_indexes for aidx in match]):     # :796, :865  バグ
```

整数をリストのリストから探すため `in` は**必ず False**、`not any(...)` は**常に True**。
「計上済みの原子は1つも無い」と常に判定される。

```
2 in [[0,1,2],[3,4]]     -> False    バッチ版が書いている比較
2 in [[0,1,2],[3,4]][0]  -> True     本来やるべき比較
```

同じ関数の `:818` では正しく `existing_indexes`（`zip` で取り出した行ごとのリスト）を
使っており、書き間違いと見られる。

### 何が起きるか

**すでに置いた断片を「新しく生成された断片」と誤認し、本来プロンプトに入れるべき断片を
スキップする。** 結果、指定した断片が最終分子に含まれない。

例: 断片が「ベンゼン環A」「ベンゼン環B」「リンカーC」のとき、A を置いた後に B を探すと
A のベンゼン環がヒットする。正しくは「その原子は A として計上済み → B はまだ無い」と
判定して B をプロンプトに足すべきだが、常に「B はもうある」と判定してスキップしてしまう。
ベンゼン環が1つしかない分子ができる。

評価では `validfragratio`（断片をすべて含むか）に効く。

### 影響範囲（実測）

`detect_existing=True` かつ `batch_prompts=True` の経路のみ。この分岐は2断片目以降で
回るので、3断片以上の linking に効く。

```
brics test 82,441行のうち
  3断片以上の linking 行     : 528
    完全に同じ断片を含む行   : 197   ← 最も踏みやすい
    一方が他方の部分構造の行 :  69
```

### 修正箇所

`FragmentLinker._batch_sample` の2箇所。`_single_sample` の2箇所（`:649`, `:701`）は
正しいので**触らない**。

| 行 | 分岐 | 囲むループ | 直し方 |
|---|---|---|---|
| 796 | `optimize_prompts=True` 側 | `for bi, (smiles, fragments, existing_indexes) in enumerate(zip(..., frag_indexes))` | `frag_indexes` → `existing_indexes` |
| 865 | `optimize_prompts=False` 側（`else:`） | `for bi, (smiles, fragments) in enumerate(zip(batch_smiles[0], batch_fragments))` — `existing_indexes` を取っていない | `frag_indexes` → `frag_indexes[bi]` |

### 前例

SAFE でも同じく site-packages に手を入れており、`README.md:77-112` /
`README_ja.md:75-110` にパッチ一覧が記載されている。同じ形で PromptSMILES の項を足す。

### 環境

`import promptsmiles` が解決するのは
`/home/sato/miniconda3/envs/promptsmiles/lib/python3.12/site-packages/promptsmiles`。
`lib/python3.1` は `python3.12` へのシンボリックリンクなので**実体は1つ**。

## スコープ外

- `promptsmiles` の他のバグ調査
- 生成のやり直し（修正後に改めて判断する）

## Plan

### Step 1: README にパッチ内容を追記する

- **Target file**: `README.md`, `README_ja.md`
- **Changes**:
  - 「Modifications to Virtual Environments」/「仮想環境の変更点」の節に、
    SAFE / T5Chem と同じ体裁で **PromptSMILES** の項を追加する。
  - 内容は `promptsmiles/samplers.py` の2箇所。既存の項と同じく
    「変更前をコメントアウトし、変更後を示す」形式で書くこと。
    ```python
    # Error in promptsmiles/samplers.py (FragmentLinker._batch_sample, line 796)
    # if not any([atom_map[aidx] in frag_indexes for aidx in match]):
    if not any([atom_map[aidx] in existing_indexes for aidx in match]):

    # Error in promptsmiles/samplers.py (FragmentLinker._batch_sample, line 865)
    # if not any([atom_map[aidx] in frag_indexes for aidx in match]):
    if not any([atom_map[aidx] in frag_indexes[bi] for aidx in match]):
    ```
  - **なぜ必要かを1〜2行で添えること**（`_batch_sample` では `frag_indexes` が
    リストのリストになるため比較が常に False になり、既に置いた断片を新規生成と誤認して
    断片をスキップする）。**それ以上は書かない。**
  - 節の見出しは既存に合わせる（`## PromptSMILES` + `### ...`）。
- **Dependencies**: none

### Step 2: 検証（メインエージェントが実施。ライブラリ本体の修正もここで行う）

- **Target file**: 実行のみ（`site-packages` はリポジトリ外なのでメインエージェントが直接適用する）
- **Changes**:
  - `samplers.py:796` と `:865` にパッチを当てる
  - 修正前後で `not any(...)` の判定が変わることを、実際の値で確認する
  - `_single_sample` の `:649` / `:701` が**変更されていない**ことを確認する
  - 3断片以上・同一断片を含む linking 行を実モデルで生成し、修正前後で結果が変わるかを見る
  - `python -c "import promptsmiles"` が通ること
- **Dependencies**: after Step 1
