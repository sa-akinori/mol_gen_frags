"""学習済み SAFE-GPT モデルでビームサーチによる分子生成を行う。

``safe.SAFEDesign`` の高レベル生成API（``super_structure`` 等）は使わず、
``SAFEDoubleHeadsModel.generate()`` を直接呼ぶ。高レベルAPIは内部で独自にプロンプトを
組み立てるため、与えたフラグメントがそのまま先頭に来る保証がないためである。

プロンプトは共通テスト分割（``data/safe/{frag_method}/normal`` の test）の
``pass_fragments`` 列を ``safe.SAFEConverter(slicer=None)`` でエンコードした部分SAFE文字列。
これは RFFMG / FragGPT / PromptSMILES がプロンプトに使うフラグメント集合と同一であり、
テスト分子を再フラグメント化することはしない。

出力（``results/safe/gpt/{model_ver}/{frag_method}/beam/`` 配下）:
    - ``predictions.csv``: 列は ``target`` / ``prompt_safe`` / ``fragment`` /
      ``prediction_1`` .. ``prediction_N``。
    - ``error_logs.csv``: 生成に失敗したバッチの行を理由つきで記録する。

バッチ単位で例外を捕捉し、失敗した行も ``error`` を埋めて表に残すため、predictions.csv は
常にテスト分割と同じ行数・同じ順序になる。
"""

import argparse
import os

import datasets
import pandas as pd
import safe
import torch
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from func.evaluation_func import Smi2CanSmi
from func.utility import BASEPATH

def decode_safe_smiles(seq: str) -> str | None:
    """生成されたSAFE文字列を正準SMILESに戻す。

    Args:
        seq: モデルが生成したSAFE文字列。

    Returns:
        str | None: 正準SMILES。デコードに失敗した場合と、非連結分子（``.`` を含む）
            になった場合は None。
    """
    decoded = safe.decode(seq, as_mol=False, fix=True, remove_added_hs=True, canonical=True, ignore_errors=True, remove_dummies=True)

    if decoded:

        if len(decoded.split('.')) > 1:
            return None

        else:
            return Smi2CanSmi(decoded)

    return None

def encode_prefixes(
    prefixes: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """部分SAFE文字列を左パディングしてモデル入力テンソルに変換する。

    プロンプトは ``[CLS] <prefix>`` とし、**末尾に EOS を付けない**。
    ``SAFETokenizer.get_pretrained()`` が返すトークナイザは TemplateProcessing により
    ``[CLS] $A [SEP]`` を自動付与するため、そのままエンコードすると続きを生成する前に
    EOS を読ませてしまう。そこで ``add_special_tokens=False`` でトークナイズし、
    BOS のみを手動で先頭に付ける（safe ライブラリが末尾トークンを ``[:, :-1]`` で
    落としているのと同じ意図）。

    Args:
        prefixes: プレフィックスとして与える部分SAFE文字列のリスト。
        tokenizer: pad_token と padding_side を設定済みのトークナイザ。
        device: テンソルを載せるデバイス。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (input_ids, attention_mask)。
            いずれも shape は (len(prefixes), バッチ内の最大プロンプト長)。
    """
    pad_id = tokenizer.pad_token_id
    encoded = tokenizer(prefixes, add_special_tokens=False)
    prompt_ids = [[tokenizer.bos_token_id] + ids for ids in encoded["input_ids"]]
    max_len = max(len(ids) for ids in prompt_ids)

    input_ids = [[pad_id] * (max_len - len(ids)) + ids for ids in prompt_ids]
    attention_mask = [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in prompt_ids]
    return torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(attention_mask, dtype=torch.long, device=device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--frag_method', type=str, default='brics', choices=['brics', 'rc_cms'], help='Fragmentation method (default: brics)')
    parser.add_argument('--model_ver', type=str, default='finetuning', choices=['finetuning', 'from_scratch', 'pretrained'], help='Phase name (default: finetuning)')
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate per molecule (default: 50)")
    parser.add_argument("--num_beams", type=int, default=50, help="Number of beams for beam search (default: 50)")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum sequence length (default: 200)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--n_generate", type=int, default=None, help="Number of leading test rows to use (default: all rows)")
    args = parser.parse_args()

    dataset_dir = f'{BASEPATH}/data/safe/{args.frag_method}/normal/'

    if args.model_ver == 'pretrained':
        model_path = f'{BASEPATH}/models/safe/gpt/pretrained/'
        output_dir = f'{BASEPATH}/results/safe/gpt/pretrained/{args.frag_method}/beam/'

    else:  # finetuning / from_scratch
        model_path = f'{BASEPATH}/models/safe/gpt/{args.model_ver}/{args.frag_method}/best_model'
        output_dir = f'{BASEPATH}/results/safe/gpt/{args.model_ver}/{args.frag_method}/beam/'

    model     = SAFEDoubleHeadsModel.from_pretrained(model_path)
    tokenizer = SAFETokenizer.from_pretrained(model_path).get_pretrained()
    encoder   = safe.SAFEConverter(slicer=None)

    # Decoder-only batched generation requires left padding.
    tokenizer.padding_side = "left"

    if args.max_length > model.config.max_position_embeddings:
        raise ValueError(f"max_length={args.max_length} > max_position_embeddings={model.config.max_position_embeddings}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    test_df = datasets.load_from_disk(dataset_dir)["test"].to_pandas()

    if args.n_generate is not None:
        test_df = test_df.head(args.n_generate)

    generated_safe = []
    prompts = []
    error_logs = []
    for start in tqdm(range(0, len(test_df), args.batch_size), desc='prediction'):
        batch_df = test_df.iloc[start:start + args.batch_size]

        try:
            prefixes = [encoder.encoder(f, canonical=True, randomize=False, constraints=None, allow_empty=True) for f in batch_df['pass_fragments']]
            input_ids, attention_mask = encode_prefixes(prefixes, tokenizer, device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=args.num_beams,
                    num_return_sequences=args.n_samples,
                    max_length=args.max_length,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prompts.extend(prefixes)
            generated_safe.extend([decoded[i * args.n_samples:(i + 1) * args.n_samples] for i in range(len(prefixes))])

        except Exception as error:
            prompts.extend(["error"] * len(batch_df))
            generated_safe.extend([["error"] * args.n_samples for _ in range(len(batch_df))])
            error_logs.extend(
                [row_index, target, fragments, type(error).__name__, str(error)]
                for row_index, target, fragments
                in zip(batch_df.index, batch_df['smiles'], batch_df['pass_fragments'])
            )


    base_df = pd.DataFrame({'target': test_df['smiles'], 'prompt_safe': prompts, 'fragment': test_df['pass_fragments']}, index=test_df.index)
    smiles_df = pd.DataFrame(
        [[decode_safe_smiles(seq) or 'safe_invalid' for seq in seqs] for seqs in generated_safe],
        columns=[f'prediction_{i + 1}' for i in range(args.n_samples)],
        index=test_df.index,
    )
    predictions_df = pd.concat([base_df, smiles_df], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = f'{output_dir}/predictions.csv'
    predictions_df.to_csv(predictions_path)

    error_path = f'{output_dir}/error_logs.csv'
    error_df = pd.DataFrame(error_logs, columns=['row_index', 'target', 'pass_fragments', 'error_type', 'error_message'])
    error_df.to_csv(error_path, index=False)
    print(f"Saved SMILES predictions to: {predictions_path}")
    print("Generation completed!")
