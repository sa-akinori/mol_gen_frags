# PromptSMILES が断片集合全体を条件にできない理由と、その根拠となるコード

- **Date**: 2026-08-09
- **対象**: `promptsmiles` 1.7.2
- **実体パス**: `/home/sato/miniconda3/envs/promptsmiles/lib/python3.12/site-packages/promptsmiles/`
- **目的**: `src/func/generation_promptsmiles_func.py` の `is_promptsmiles_expressible` が
  どのコードを根拠にしているかを記録する。生成対象から外す行の判定はすべてここに帰着する。

## 除外の基準

**「ライブラリのどの経路でも、指定された断片集合全体を条件として単一の分子を生成できない」場合のみ除外する。**

サンプラーは2つしかないので（`__init__.py` の `DeNovo` / `FragmentLinker` / `ScaffoldDecorator`。
`DeNovo` は断片を指定できない）、根拠は**2本足**になる。片方だけでは除外の理由にならない。

| | 根拠の性質 |
|---|---|
| **A-1. FragmentLinker** | `assert` で**実行を拒否**する（例外が出て1分子も生成されない） |
| **A-2. ScaffoldDecorator** | **実行はされる**が、集合全体を単一分子に組み上げる経路がコード上存在しない |

「性能が落ちる」「実行時に取りこぼす」は除外の根拠にしない（後述 C）。

## 調査方法

`samplers.py` と `utils.py` の `assert` / `raise` / `logger.warn` をすべて列挙し、各条件を実際に
投げて例外が出ることを確認した。A-2 については各段階の関数を単体で通し、出力を記録した。
「コードを読んでそう見える」ではなく、実行結果を載せている。

---

## A. 断片集合全体を条件にできない（＝除外の根拠）

対象は **複数断片で、いずれかの断片が attachment point を2個以上（または0個）持つ集合**。

| | 該当行 | 全行 | 割合 |
|---|---|---|---|
| brics | 74,631 | 82,441 | 90.53% |
| rc_cms | 76,690 | 90,974 | 84.30% |

### A-1. FragmentLinker は実行を拒否する

**該当コード**: `samplers.py:569-573`（`FragmentLinker.__init__`）

```python
for frag in fragments:
    aidx = utils.get_attachment_points(frag)
    assert (
        len(aidx) == 1
    ), f"Fragment {frag} should only have one attachment point"
```

- **例外**: `AssertionError: Fragment <smi> should only have one attachment point`
- **発生時点**: コンストラクタ。**生成前に落ちる**
- **「ちょうど1個」であり「1個以下」ではない**。0個でも落ちる

実測（ダミーの `sample_fn` / `evaluate_fn`）:

```
['*CCO', '*c1ccccc1']         -> 構築OK
['*CCO', '*c1ccc(*)cc1']      -> AssertionError: Fragment *c1ccc(*)cc1 should only have one attachment point
['*CCO', 'CCO']               -> AssertionError: Fragment CCO should only have one attachment point
['*CCO', '*CC*', '*c1ccccc1'] -> AssertionError: Fragment *CC* should only have one attachment point
```

**なぜ結合1本しか作れないのか**: 2個目以降の断片は、括弧枝として文字列に挿入される
（`samplers.py:664-671` / `:803-829`）。

```python
tsmi = "".join(smiles_tokens[: i + 1] + ["("] + [fi_smi] + [")"] + smiles_tokens[i + 1 :])
```

SMILES の括弧枝は結合を1本しか作らない。結合点2個の断片を置くには両端に閉環番号を張る必要が
あるが、この挿入方式にその機能は無い。assert は後付けの制限ではなく、アルゴリズムの構造から
来る制約。

### A-2. ScaffoldDecorator は実行されるが単一分子を作れない

`ScaffoldDecorator` には結合点の数に関する制約が**まったく無い**（1個/2個/4個いずれも構築OK）。
ドット連結した断片集合をそのまま `scaffold` に渡しても**例外は出ない**。
したがって「実行を拒否する」という根拠は使えず、以下5点のコードが根拠になる。

#### ① 生成はプロンプト文字列の末尾への追記でしかない — `samplers.py:349-352`

```python
prompts = [v.strip_smiles for v in batch_variants]
smiles = self.sample_fn(prompt=prompts, batch_size=batch_size, **self.sample_fn_kwargs)
```

`sample_fn` の契約は `smiles.startswith(prompt)`（`samplers.py:624`, `:773`, `:848` の assert）。
SMILES で末尾に足した文字は最後の原子にしか結合できず、`.` の向こう側の成分と結合するには
閉環番号の対が要る。

#### ② 閉環番号を新規に発行するコードが存在しない — `utils.py` の環番号関数はすべて振り直し

| 関数 | 役割 | 呼び出し元 |
|---|---|---|
| `utils.py:330` `_reverse_ring_numbers` | 反転時に既存の閉環を振り直す | `reverse_smiles` |
| `utils.py:349` `_check_ring_numbers` | 重複した閉環番号を振り直す | `root_smiles`, `randomize_smiles` |
| `utils.py:650` `correct_fragment_ring_numbers` | 衝突回避の振り直し | **FragmentLinker のみ**（`samplers.py:659`, `:709`, `:808`, `:872`） |

いずれも**既存の閉環を振り直すだけで、新しい閉環は作らない**。
ScaffoldDecorator の経路からは `correct_fragment_ring_numbers` すら呼ばれない。
成分間の結合を生む手段がコード上存在しない。

#### ③ `*` は単純な文字列置換で消される — `utils.py:539`

```python
smi = smi.replace("(*)", "").replace("*", "")
```

結合は作られず、どこに何が付くべきかの情報も残らない。プロンプトは `.` を含んだ
多成分文字列のままになる。

```
'c1ccccc1(*).OCC(*)'  ->  prompt 'c1ccccc1.OCC'
```

#### ④ 指定した結合点を末尾に持ってこられない — `utils.py:166`

```python
new_smi = Chem.MolToSmiles(mol, rootedAtAtom=rootAtom)
```

RDKit の `rootedAtAtom` は**成分をまたいだ並べ替えをしない**。実測で、どちらの結合点を
指定しても同じ文字列が返る。

```
入力 scaffold: *c1ccccc1.*CCO   (get_attachment_points -> [0, 5])

rootedAtAtom=0 -> *CCO.*c1ccccc1
rootedAtAtom=7 -> *CCO.*c1ccccc1        同じ

root_smiles(reverse=True):
  at_pt=0 -> 'c1ccccc1(*).OCC(*)'
  at_pt=5 -> 'c1ccccc1(*).OCC(*)'       同じ。結合点を選べていない
```

#### ⑤ 反転で SMILES 構文が壊れる — `utils.py:175` → `utils.py:281` `reverse_smiles`

```python
if reverse:
    new_smi = _check_ring_numbers(new_smi)
    new_smi = reverse_smiles(new_smi)
```

`.` をまたいで文字列全体を反転するため、実データではプロンプト自体が RDKit で読めなくなる。

```
実データの unsupported 集合:
  *C(=O)C=*.*N1CCN(*)CC1.*c1ccc(*)cc1.*c1nc(N)c2cc(*)c(*)cc2n1

  at_pt= 0 -> prompt 'n1c2ccccc2c(N)nc1.c3ccccc3.C4CNCCN4.=CC(=O)'   RDKitで読める: False
  at_pt= 2 -> prompt 'n1c2ccccc2c(N)nc1.c3ccccc3.C4CNCCN4.O=CC='     RDKitで読める: False
```

#### 実際の生成結果

上の①〜⑤の帰結として、実モデルで生成すると**例外は出ないが単一分子にならない**。

```
断片集合: *C(=O)C=*.*N1CCN(*)CC1.*c1ccc(*)cc1.*c1nc(N)c2cc(*)c(*)cc2n1
  FragmentLinker    : AssertionError: Fragment *C(=O)C=* should only have one attachment point
  ScaffoldDecorator : 例外なし。生成 3 分子
      成分数=-1  c1ccccc1.=CC=O.C2CNCCN2.c3cc4c(N)ncnc4cc3N=2      RDKitで読めない
      成分数=-1  c12ccccc2c(N)ncn1.=C(C(=O)).C3NCCNC3.c4ccccc4-1   RDKitで読めない
      成分数=-1  c12ccccc2c(N)ncn1.=C(C(=O)).C3NCCNC3.c4ccccc4N=3  RDKitで読めない

断片集合: *CCCCC(*)=O.*OC.*c1ccc(*)cc1.*c1nc(N)c2cc(*)c(*)cc2n1
  ScaffoldDecorator : 例外なし
      成分数=4   CCCCC(=O).c1cc2ncnc(N)c2cc1.c3ccccc3.OC          4成分のまま
```

**我々の扱い**: `sampler = "unsupported"` として記録し、生成しない。予測列はすべて `INVALID_SMILES`。

---

## B. 実行を拒否するが、実データでは該当しない／経路分けで回避しているもの

### B-1. FragmentLinker: 断片に `X` を含む

**該当コード**: `samplers.py:556-557`

```python
if any(["X" in f for f in fragments]):
    raise NotImplementedError("FragmentLinker does not support X substitution yet.")
```

- **例外**: `NotImplementedError`（実測で確認）。判定は SMILES 文字列の単純な部分一致
- **該当行数**: brics 0 / rc_cms 0
- **扱い**: 該当が無いため個別の分岐は設けていない

### B-2. ScaffoldDecorator: scaffold が str でも list でもない

**該当コード**: `samplers.py:157`

```python
assert isinstance(scaffold, (str, list)), "Scaffold must be a SMILES string or list of SMILES strings."
```

- **例外**: `AssertionError`（実測で確認）
- **扱い**: 常に `str` を渡すのでコード上発生しない

### B-3. FragmentLinker: 断片が1個

- **該当コード**: **明示的なチェックは存在しない。** `samplers.py:614` の `f0 = fragments.pop(i)`
  の後、残りが空のまま次の `fragments.pop(i)` に到達して落ちる
- **例外**: `IndexError: pop from empty list`（実測で確認）。結合点が何個あっても起きる
- **注意**: 設計された拒否ではなく**未処理のクラッシュ**
- **扱い**: 断片1個の集合は `ScaffoldDecorator` に回すためこの経路に入らない。
  分岐の直前に理由をコメントで残してある

---

## C. 除外の根拠にならないもの

ライブラリは入力を受け付け、**単一分子の出力も得られる**。性能が落ちるだけ、または実行時に
取りこぼすだけ。

### C-1. 3断片以上での `scan` 強制 — `samplers.py:560-564`

```python
if len(fragments) > 2 and not self.scan:
    logger.warn("Scan must be used for more than two fragments, Scan will be enabled.")
    self.scan = True
```

- `scan` は `FragmentLinker.__init__` の**正式な公開パラメータ**（`scan: bool = False`）で
  docstring にも記載がある。2断片の「単純連結」（`samplers.py:872` の `smiles = smiles + fi_smi`）
  に対し、3断片以上では「断片を括弧枝として全位置に挿入し NLL 最小を選ぶ」方式に切り替わる
- **除外しない理由**: 例外は出ず出力も得られる。実測した断片充足率は 2断片 62.7% に対し
  3断片以上 42.7% で、**外すと PromptSMILES に有利に働く**（苦手な行だけ除かれる）。
  性能差は指標として出すべきもので、除外して隠すべきではない
- **該当行数**: brics 528 / rc_cms 457

### C-2. `*` を持たない scaffold への結合点の捏造 — `samplers.py:159-161` → `utils.py:697-730`

`ScaffoldDecorator` に渡した scaffold に `*` が無いと、空き原子価を持つ全ての重原子に `*` を
付けた超構造に置き換えられる。例外は出ない。

- **該当行数**: brics 0 / rc_cms 0（結合点0個の断片を含む行は存在しない）

### C-3. 実行時のフォールバック（断片が最終分子に入らない）

いずれも例外を出さず、**断片集合から静的に判定できない**。同じ集合でもサンプルごとに成否が変わる。

| 箇所 | 内容 |
|---|---|
| `samplers.py:838-840` | 挿入位置が1つも無いと `# Don't add fragment` で断片を捨てる |
| `samplers.py:786-802` | `detect_existing` が「もう生成済み」と判定したら断片を足さない |
| `samplers.py:391-420` | プロンプト再配置に失敗すると `# Skip position` で結合点を飛ばし、さらに失敗すると `# Stop here` で装飾を打ち切る |

- **除外しない理由**: 事前に判定できないため除外リストが作れない。加えて `Skip position` は
  断片1個の scaffold 行でも発火する（実測15行×5サンプルで12回）
- **扱い**: 断片が入っていない分子は評価の `validfragratio` が0点にするので指標にそのまま現れる。
  除外して見えなくするより正確

### C-4. 生成 SMILES がプロンプトで始まることの検査 — `samplers.py:624`, `:773`, `:848`

```python
assert smiles.startswith(prompt), f"Sampled SMILES {smiles} does not start with prompt {prompt}, why not?"
```

入力への制約ではなく、**我々が渡す `sample_fn` の契約**に対する検査。
`GPT2PromptSampler.sample` は `prompt + completion` を組み立てて返すので構造的に必ず成立する。

---

## D. 我々のコードでの実装

`src/func/generation_promptsmiles_func.py`

```python
def is_promptsmiles_expressible(fragment_set: str) -> bool:
    fragments = fragment_set.split(".")
    return len(fragments) == 1 or all(frag.count("*") == 1 for frag in fragments)
```

- `len(fragments) == 1` → `ScaffoldDecorator` に回す（B-3 の `IndexError` を避ける）
- `all(count("*") == 1)` → **A-1 の assert がそのまま条件になっている**

False の行は A-1（FragmentLinker が拒否）と A-2（ScaffoldDecorator が単一分子を作れない）の
両方に該当するため、生成しない。

生成対象:

| | 断片1個 | 全断片1点 | 生成対象 | 全行 |
|---|---|---|---|---|
| brics | 2,690 | 5,120 | **7,810** | 82,441 |
| rc_cms | 9,473 | 4,811 | **14,284** | 90,974 |

---

## E. 再検証の手順

`is_promptsmiles_expressible` を変更したとき、または `promptsmiles` を更新したときは以下を確認する。

1. `assert` / `raise` / `logger.warn` を再列挙し、A・B・C の表と一致するか確認する。

```bash
grep -n "assert \|raise \|logger.warn" \
  "$(python -c 'import promptsmiles,os;print(os.path.dirname(promptsmiles.__file__))')"/samplers.py
```

2. A-1 の assert が実際に発火することを、ダミーの `sample_fn` / `evaluate_fn` で確認する。
3. A-2 の④について、`utils.root_smiles` にドット連結の SMILES と各結合点を渡し、
   **返る文字列が結合点ごとに変わらない**ことを確認する（変わるようになっていれば
   ScaffoldDecorator 側の前提が崩れる）。
4. `utils.py` の環番号関数が振り直しのみで、新規の閉環を作らないことを確認する（A-2 の②）。
5. test 分割に対する該当行数が D の表と一致することを確認する。
