import os
from pathlib import Path

import torch
import pandas as pd
import safe
from safe.sample import SAFEDesign
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel
from rdkit import Chem

from func.utility import set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _canonicalize_smiles(smi: str | None) -> str | None:
    """SMILES を RDKit で妥当性判定し、canonical SMILES を返す。

    Args:
        smi: 判定対象の SMILES（None や空文字列は無効扱い）。

    Returns:
        str | None: 妥当な場合は canonical SMILES、妥当でなければ None。
    """
    mol = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _decode_safe_smiles(seq: str) -> str | None:
    """SAFE 文字列をデコードし、RDKit で妥当性判定した canonical SMILES を返す。

    Args:
        seq: デコード対象の SAFE 文字列。

    Returns:
        str | None: 妥当な場合は canonical SMILES、妥当でなければ None。
    """
    decoded = safe.decode(
        seq, as_mol=False, fix=True, remove_added_hs=True,
        canonical=True, ignore_errors=True, remove_dummies=True,
    )
    return _canonicalize_smiles(decoded)


def generate_denovo_direct(
    designer: SAFEDesign,
    n_samples: int,
    max_length: int,
) -> list[str | None]:
    """model.generate を直接呼んで de novo 生成し、canonical SMILES のリストを返す。

    Args:
        designer: 生成に用いる SAFEDesign インスタンス。
        n_samples: 生成本数（num_return_sequences）。
        max_length: 生成配列の最大長。

    Returns:
        list[str | None]: 各生成分子の canonical SMILES。妥当でなければ None。
    """
    tokenizer = designer.tokenizer.get_pretrained()
    outputs = designer.model.generate(
        inputs=None,
        generation_config=designer.generation_config,
        do_sample=True,
        num_return_sequences=n_samples,
        max_length=max_length,
        early_stopping=True,
    )
    safe_seqs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [_decode_safe_smiles(seq) for seq in safe_seqs]


def generate_denovo_library(
    designer: SAFEDesign,
    n_samples: int,
    max_length: int,
) -> list[str | None]:
    """SAFEDesign.de_novo_generation で de novo 生成し、canonical SMILES のリストを返す。

    Args:
        designer: 生成に用いる SAFEDesign インスタンス。
        n_samples: 生成本数（n_samples_per_trial）。
        max_length: 生成配列の最大長。

    Returns:
        list[str | None]: 各生成分子の canonical SMILES。妥当でなければ None。
    """
    generated = designer.de_novo_generation(
        n_samples_per_trial=n_samples,
        sanitize=False,
        how="random",
        max_length=max_length,
    )
    return [_canonicalize_smiles(smi) for smi in generated]


def compute_validity(smiles_list: list[str | None]) -> dict[str, float | int]:
    """生成 SMILES リストから Validity（妥当率）を計算する。

    Args:
        smiles_list: 生成 SMILES のリスト（妥当でない要素は None）。

    Returns:
        dict[str, float | int]: 以下のキーを持つ辞書。
            - 'n_total' (int): 生成本数（リスト長）。
            - 'n_valid' (int): 妥当な（None でない）SMILES 数。
            - 'validity' (float): n_valid / n_total（n_total が 0 なら 0.0）。
    """
    n_total = len(smiles_list)
    n_valid = sum(smi is not None for smi in smiles_list)
    validity = n_valid / n_total if n_total else 0.0
    return {"n_total": n_total, "n_valid": n_valid, "validity": validity}


def run_denovo_comparison(
    model_path: str | Path,
    output_dir: str | Path,
    n_samples: int = 1000,
    max_length: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """direct と library の 2 経路で de novo 生成し、Validity を比較する。

    Args:
        model_path: 学習済み SAFE-GPT モデルのパス。
        output_dir: 生成結果・比較サマリの保存先ディレクトリ。
        n_samples: 各手法での生成本数。
        max_length: 生成配列の最大長。
        random_seed: 乱数シード（各手法の生成前に設定）。

    Returns:
        pd.DataFrame: 手法ごとの比較サマリ。カラムは
            ['method', 'n_total', 'n_valid', 'validity']。
    """
    model = SAFEDoubleHeadsModel.from_pretrained(str(model_path))
    tokenizer = SAFETokenizer.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    designer = SAFEDesign(model=model, tokenizer=tokenizer)

    output_dir = Path(output_dir)
    methods = {"direct": generate_denovo_direct, "library": generate_denovo_library}
    summary = []
    for method, generate_fn in methods.items():
        set_seed(random_seed)
        smiles_list = generate_fn(designer, n_samples, max_length)

        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        gen_df = pd.DataFrame(
            list(enumerate(smiles_list)), columns=["raw_index", "smiles"],
        )
        gen_df.to_csv(method_dir / "generated.csv", index=False)

        metrics = compute_validity(smiles_list)
        summary.append({"method": method, **metrics})

    summary_df = pd.DataFrame(summary, columns=["method", "n_total", "n_valid", "validity"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "comparison.csv", index=False)
    print(summary_df)
    return summary_df


if __name__ == '__main__':
    fd = Path(__file__).resolve().parent.parent
    model_path = fd / 'models' / 'safe' / 'gpt' / 'pretrained'
    output_dir = fd / 'results' / 'safe' / 'gpt' / 'pretrained' / 'denovo'
    run_denovo_comparison(
        model_path=model_path,
        output_dir=output_dir,
        n_samples=1000,
        max_length=200,
        random_seed=42,
    )
