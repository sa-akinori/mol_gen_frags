# Plan: FragGPT の augment_fusmiles を文字列置換に簡素化

- **Date**: 2026-08-03
- **Status**: pending-approval
- **調査元セッション**: PromptSMILES ベースライン実装セッション（`.plans_for_ai/2026-07-28/add_promptsmiles_baseline.md`）
- **引き継ぎ理由**: 対象ファイルが FragGPT セッションの担当のため、調査結果のみを渡す

## Overview

`src/func/fragment_for_fraggpt.py` の `augment_fusmiles()` は、結合点の付番をランダム置換するために
**フラグメントを RDKit の Mol に変換して `SetIsotope` で書き換え、`MolToSmiles` で書き戻している**。

調査の結果、この Mol 変換は **論文にも本家実装にも根拠がなく、不要**であることが分かった。
正規表現による文字列置換で置き換えられ、**約30倍高速**になる。

## 調査結果（すべて実測・一次情報）

### 1. 論文が定める augmentation は2つだけ

FragGPT 論文（Yue et al., *Chem. Sci.* 2024, DOI: 10.1039/d4sc03744h）の記述:

> "randomly transforming the **i** value within (1 ~ n), so that the break point identification
> of each fragment has a probability of any value in (1 ~ n)"

> "by randomly **shuffling** molecular fragments, allowing the fragments to be in any position
> in the sequence"

> "Both data augmentations are carried out simultaneously to further enrich the diversity of data"

→ **①付番のランダム置換 と ②フラグメント順のシャッフル のみ**。
**再カノニカル化については論文に一切記述がない。**

### 2. 本家実装も文字列置換を使っている

公開リポジトリ: `https://github.com/pengbingxin/FragGPT-Interface`
（学習側コードは非公開。`task/__init__.py` が import する `datasets/` はリポジトリに含まれない。
推論側の `generate_frag_pocket.py` に同等の処理がある）

```python
# generate_frag_pocket.py 257-259行
pat = r'\[(\d+)\*\]'
new_frags_list_1 = []
for m, frag in enumerate(frags_list_1):
    new_frags_list_1.append(re.sub(pat, lambda match: f"[{int(match.group(1))+1}*]" if int(match.group(1)) > select_id else match.group(0), frag))
```

```python
# generate_frag_pocket.py 245, 278, 327, 364行（フラグメント順シャッフル）
np.random.shuffle(ori_frags_list)
```

→ **付番は `re.sub` による文字列置換、順序はリストの `shuffle`。Mol への変換は行っていない。**

### 3. Mol 変換は化学的に何も追加していない（実測）

`data/fraggpt/brics/normal` の train split 2,000集合 × seed 3通り = 6,000件で検証:

| 検証 | 結果 |
|---|---|
| 現行 Mol版 と 文字列置換版 が化学的に同一か | **6,000件すべて一致**（不一致0） |
| 現行 Mol版 と「文字列置換+再カノニカル化」が文字列として一致するか | **6,000件すべて一致**（不一致0） |

つまり Mol 変換の唯一の効果は `MolToSmiles` による**再カノニカル化**（原子の書き順の変化）であり、
これは論文にも本家にもない副作用。

### 4. 速度（実測）

| 実装 | 2,000集合 | 全1,714,298集合の推定 |
|---|---|---|
| 現行 Mol版 | 0.65 秒 | **9.3 分** |
| 文字列置換 + 再カノニカル化 | 0.55 秒 | 7.9 分 |
| **文字列置換のみ** | **0.02 秒** | **0.3 分** |

## Plan

### Step 1: `augment_fusmiles` を文字列置換に置き換える

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**: 現行の Mol ベース実装（192-222行付近）を以下に置き換える。

  ```python
  ATTACHMENT_LABEL = re.compile(r"\[(\d+)\*\]")

  def augment_fusmiles(fragments: list[str], rng: random.Random) -> list[str]:
      labels = sorted({int(m) for f in fragments for m in ATTACHMENT_LABEL.findall(f)})
      new_labels = list(range(1, len(labels) + 1))
      rng.shuffle(new_labels)
      relabeling = {str(o): str(n) for o, n in zip(labels, new_labels)}
      augmented = [ATTACHMENT_LABEL.sub(lambda m: f"[{relabeling[m.group(1)]}*]", f) for f in fragments]
      rng.shuffle(augmented)
      return augmented
  ```

- **実装上の必須要件（いずれも実データで検証済み）**:
  - **同時置換であること**。逐次 `str.replace` だと `1→2` の後に `2→1` を適用して壊れる。
    `re.sub` にコールバックを渡す形なら1パスで置換されるため安全。
  - **正規表現は `\[(\d+)\*\]`**（`\]` まで含める）。含めないと `[10*]` を `[1*]` と誤マッチする。
  - **フラグメント順のシャッフルは現行のまま維持**（`rng.shuffle(augmented)`）。
    ここは本家とも一致しており変更不要。
- **Dependencies**: none

### Step 2: 不要になった依存の整理

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**: `augment_fusmiles` が `parse_fragments` / `DUMMY_ATOMIC_NUM` / `Chem` を使わなくなるため、
  他から参照されていなければ import や定数を整理する。
  **他の関数がまだ使っている場合は残すこと**（`fragment_for_fraggpt.py` 内の他関数を要確認）。
- **Dependencies**: after Step 1

### Step 3: docstring の更新

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**: docstring の `Raises: ValueError: If RDKit cannot parse one of the fragments.` を削除する
  （パースしなくなるため）。教える2つの不変性の説明はそのまま維持。
- **Dependencies**: after Step 1

## 判断が必要な点

**SMILES の検証（`parse_fragments`）がなくなる。**

- 本家も検証していない
- 入力は `make_datasets.py` が生成した `data/fraggpt/{frag}/normal` の `full_fragments` 列で、
  上流でキュレーション済み
- PromptSMILES 側でも同じ理由で `Chem.MolFromSmiles` の None チェックを省く判断をしている
  （`.plans_for_ai/2026-07-28/add_promptsmiles_baseline.md` 参照）

→ 同じ前提で問題ないと考えるが、FragGPT セッション側で判断されたい。

## 注意

- 本計画は**調査結果の引き継ぎ**であり、調査元セッションでは実装していない。
  対象ファイルが FragGPT セッションの担当であり、並行編集による競合を避けるため。
- 実装する場合は、上記「実装上の必須要件」の3点を必ず満たすこと。
