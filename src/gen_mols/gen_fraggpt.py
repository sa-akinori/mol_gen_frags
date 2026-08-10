"""Run FragGPT generation as a subprocess and record its wall-clock time.

The generation itself lives in ``src/func/generation_fraggpt_func.py``; launching it as a
subprocess is what ``gen_rffmg.py`` and ``gen_safe.py`` do too, so all baselines are timed
the same way (model loading included) by ``func.generation_time.run_and_record_time``.
"""

import argparse
import os
from pathlib import Path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from func.generation_time import run_and_record_time
from func.utility import BASEPATH

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generate molecules with a FragGPT (FU-SMILES) GPT2 model')

    # Model parameters
    parser.add_argument('--frag_method', type=str, default='brics', choices=['brics', 'rc_cms'], help='Fragmentation method (default: brics)')
    parser.add_argument('--model_ver', type=str, default='finetuning', choices=['finetuning', 'from_scratch'], help='Model version (default: finetuning)')

    # Generation parameters
    parser.add_argument('--gen_method', type=str, default='beam', choices=['beam', 'sampling'], help='Decoding scheme: beam search or multinomial sampling (default: beam)')
    parser.add_argument('--additional_path', type=str, default='normal', help='Additional path segment to append to the output dir')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples to generate per molecule (default: 50)')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length (default: 256)')
    parser.add_argument('--num_beams', type=int, default=50, help='Number of beams, used by the beam gen_method (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    frag_method = args.frag_method
    model_ver   = args.model_ver
    model_path  = f'{BASEPATH}/models/fraggpt/gpt/{model_ver}/{frag_method}/best_model'
    output_dir  = f'{BASEPATH}/results/fraggpt/gpt/{model_ver}/{frag_method}/{args.gen_method}/{args.additional_path}'
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", f"{BASEPATH}/src/func/generation_fraggpt_func.py",
        "--frag_method", frag_method,
        "--model_ver", model_ver,
        "--gen_method", args.gen_method,
        "--additional_path", args.additional_path,
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
            "backend": "gpt",
            "model_ver": model_ver,
            "frag_method": frag_method,
            "additional_path": args.additional_path,
            "gen_method": args.gen_method,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "model_path": model_path,
            "random_seed": args.random_seed,
        },
    )
    print(f"Saved generation time to: {json_path}")
