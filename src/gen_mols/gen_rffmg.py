import os
import argparse
from pathlib import Path

from func.utility import BASEPATH
from func.generation_time import run_and_record_time

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generate molecules for the RFFMG representation (T5Chem or GPT2 backend)')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='t5chem', choices=['t5chem', 'gpt'],
                        help='Backend model: t5chem (T5) or gpt (GPT2) (default: t5chem)')
    parser.add_argument('--frag_method', type=str, default='rc_cms', choices=['rc_cms', 'brics'],
                        help='Dataset slice name (default: rc_cms)')
    parser.add_argument('--model_ver', type=str, default='finetuning', choices=['finetuning', 'from_scratch'],
                        help='Model version (default: finetuning)')
    parser.add_argument('--sampling_num', type=int, default=10, choices=[5, 10],
                        help='Fragment-sampling multiplier N, selecting the data/rffmg/<frag>/<N>times_sampling slice (default: 10)')

    # Generation parameters
    parser.add_argument('--additional_path', type=str, default='normal', choices=['normal', 'dup_frags', 'frag_num', 'frag_order', 'attach_point_num'],
                        help='Additional path (default: normal)')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples to generate per molecule (default: 50)')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length, used by the gpt backend (default: 256)')
    parser.add_argument('--num_beams', type=int, default=50,
                        help='Number of beams for beam search (default: 50)')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size (default: 24)')
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    gen_method = 'beam'
    frag_method = args.frag_method
    model_name = args.model_name
    model_ver = args.model_ver
    additional_path = args.additional_path
    sampling = f'{args.sampling_num}times_sampling'
    model_path  = f'{BASEPATH}/models/rffmg/{model_name}/{model_ver}/{frag_method}/{sampling}/best_model'
    output_dir  = f'{BASEPATH}/results/rffmg/{model_name}/{model_ver}/{frag_method}/{sampling}/{gen_method}/{additional_path}'
    dataset_dir = f'{BASEPATH}/data/rffmg/{frag_method}/{sampling}/{additional_path}'
    os.makedirs(output_dir, exist_ok=True)

    # Generate compounds (beam search). Both backends write predictions.csv with columns
    # `target`, `prediction_1..N` so the shared evaluation pipeline can read them.
    if model_name == 't5chem':
        cmd = [
            "t5chem", "predict",
            "--data_dir", f"{dataset_dir}/",
            "--model_dir", f"{model_path}/",
            "--prediction", f"{output_dir}/predictions.csv",
            "--num_beams", str(args.num_beams),
            "--num_preds", str(args.n_samples),
            "--batch_size", str(args.batch_size),
        ]
    else:  # gpt
        cmd = [
            "python", f"{BASEPATH}/src/func/generation_rffmg_func.py",
            "--model_path", model_path,
            "--dataset_file", f"{dataset_dir}/test.source",
            "--target_file", f"{dataset_dir}/test.target",
            "--output_dir", output_dir,
            "--n_samples", str(args.n_samples),
            "--max_length", str(args.max_length),
            "--num_beams", str(args.num_beams),
            "--batch_size", str(args.batch_size),
            "--random_seed", str(args.random_seed),
        ]
    json_path = run_and_record_time(
        cmd,
        Path(output_dir),
        n_samples=args.n_samples,
        params={
            "backend": model_name,
            "model_ver": model_ver,
            "frag_method": frag_method,
            "sampling_num": args.sampling_num,
            "additional_path": additional_path,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "model_path": model_path,
            "random_seed": args.random_seed,
        },
    )
    print(f"Saved generation time to: {json_path}")
