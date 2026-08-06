import argparse
from pathlib import Path

from func.utility import BASEPATH
from func.generation_time import run_and_record_time

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--frag_method', type=str, default='brics', choices=['brics', 'rc_cms'], help='Fragmentation method (default: brics)')
    parser.add_argument('--model_ver', type=str, default='finetuning', choices=['finetuning', 'from_scratch', 'pretrained'], help='Phase name (default: finetuning)')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples to generate per molecule (default: 50)')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum sequence length (default: 200)')
    parser.add_argument('--num_beams', type=int, default=50, help='Number of beams for beam search (default: 50)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    frag_method = args.frag_method
    model_ver = args.model_ver

    if model_ver == 'pretrained':
        model_path = f'{BASEPATH}/models/safe/gpt/pretrained/'
        output_dir = f'{BASEPATH}/results/safe/gpt/pretrained/{frag_method}/beam/'

    else:  # finetuning / from_scratch
        model_path = f'{BASEPATH}/models/safe/gpt/{model_ver}/{frag_method}/best_model'
        output_dir = f'{BASEPATH}/results/safe/gpt/{model_ver}/{frag_method}/beam/'
    
    cmd = [
        "python", f"{BASEPATH}/src/func/generation_safe_func.py",
        "--frag_method", frag_method,
        "--model_ver", model_ver,
        "--n_samples", str(args.n_samples),
        "--max_length", str(args.max_length),
        "--num_beams", str(args.num_beams),
        "--batch_size", str(args.batch_size),
    ]
    json_path = run_and_record_time(
        cmd,
        Path(output_dir),
        n_samples=args.n_samples,
        params={
            "backend": "safe_gpt",
            "model_ver": model_ver,
            "frag_method": frag_method,
            "num_beams": args.num_beams,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "model_path": model_path,
            "random_seed": args.random_seed,
        },
        predictions_pattern="predictions.csv",
    )
    print(f"Saved generation time to: {json_path}")
        
    