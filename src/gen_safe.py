import os
import subprocess
import itertools
import signal

import torch
import pandas as pd
import datasets
from tqdm import tqdm
import safe
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel
from rdkit import Chem

from func.utility import set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


if __name__=='__main__':
    fd = os.path.dirname(os.path.dirname(__file__))
    gen_method = 'direct'
    slice_name = 'brics'
    model_ver = 'finetuning'
    model_path = f'{fd}/models/safe/gpt/{model_ver}/{slice_name}/best_model'
    output_dir = f'{fd}/results/safe/gpt/{model_ver}/{slice_name}/{gen_method}/'
    dataset_dir = f'{fd}/data/safe/{slice_name}/normal'

    if gen_method == 'direct':
        generate_from_model(
            model_path=model_path,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            n_generate=1000,
            n_samples=50,
            num_beams=50,
            max_length=200,
            random_seed=42,
        )
    else:
        # beam-search
        cmd = [
            "python", f"{fd}/src/func/generation_safe_func.py",
            "--model_path", model_path,
            "--dataset_dir", dataset_dir,
            "--slice_name", slice_name,
            "--n_samples", "50",
            "--gen_method", "beam",
            "--max_length", "200",
            "--num_beams", "50",
            "--random_seed", "42",
            "--output_dir", f'{output_dir}'
        ]
        subprocess.run(cmd, check=True)

        # concat
        from glob import glob
        import pandas as pd
        error_logs = [pd.read_csv(p, index_col=0) for p in glob(f'{output_dir}/error_logs_*.csv')]
        error_logs_csv = pd.concat(error_logs).reset_index(drop=True)
        predictions = [pd.read_csv(p, index_col=0) for p in glob(f'{output_dir}/predictions_*.csv')]
        predictions_csv = pd.concat(predictions).reset_index(drop=True)
        error_logs_csv.to_csv(f'{output_dir}/error_logs.csv')
        predictions_csv.to_csv(f'{output_dir}/predictions.csv')

    # if gen_method == 'random':
    #     temperatures = [0.01, 0.1, 0.5, 1.0, 1.5]
    #     for temperature in temperatures:
    #         cmd = [
    #             "python", f"{fd}/src/func/generation_safe_func.py",
    #             "--model_path", model_path,
    #             "--dataset_dir", dataset_dir,
    #             "--slice_name", slice_name,
    #             "--n_samples", "50",
    #             "--gen_method", "random",
    #             "--max_length", "200",
    #             "--temperature", str(temperature),
    #             "--output_dir", f'{output_dir}/temperature_{temperature}'
    #         ]
    #         subprocess.run(cmd, check=True)

    # elif gen_method == 'beam':
    #     num_beam_blocks = [1, 2, 5]
    #     div_penalties   = [0.0, 0.3, 0.7, 1.2, 1.5]
    #     for num_beam_block, div_penalty in list(itertools.product(num_beam_blocks, div_penalties)):
    #         if (num_beam_block == 1 and div_penalty != 0.0) or (num_beam_block != 1 and div_penalty == 0.0):
    #             continue
            
    #         cmd = [
    #             "python", f"{fd}/src/func/generation_safe_func.py",
    #             "--model_path", model_path,
    #             "--dataset_dir", dataset_dir,
    #             "--slice_name", slice_name,
    #             "--n_samples", "50",
    #             "--gen_method", "beam",
    #             "--max_length", "200",
    #             "--num_beams", "50",
    #             "--num_beam_groups", str(num_beam_block),
    #             "--diversity_penalty", str(div_penalty),
    #             "--output_dir", f'{output_dir}/beam_groups_{num_beam_block}/div_penalty_{div_penalty}'
    #         ]
    #         subprocess.run(cmd, check=True)

