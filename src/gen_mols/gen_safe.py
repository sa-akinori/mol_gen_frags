import argparse
from pathlib import Path

from func.utility import BASEPATH
from func.generation_time import run_and_record_time

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--frag_method', type=str, default='brics', choices=['brics', 'rc_cms'],
                        help='Fragmentation method (default: brics)')
    parser.add_argument('--model_ver', type=str, default='finetuning', choices=['finetuning', 'from_scratch', 'pretrained'],
                        help='Phase name (default: finetuning)')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples to generate per molecule (default: 50)')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum sequence length (default: 200)')
    parser.add_argument('--num_beams', type=int, default=50,
                        help='Number of beams for beam search (default: 50)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--machine_id', type=int, default=0,
                        help='Machine ID (default: 0)')
    parser.add_argument('--total_machines', type=int, default=1,
                        help='Total number of machines (default: 1)')
    args = parser.parse_args()
    
    frag_method = args.frag_method
    model_ver = args.model_ver
    dataset_dir = f'{BASEPATH}/data/safe/{frag_method}/normal/'
    
    if model_ver == 'pretrained':
        model_path = f'{BASEPATH}/models/safe/gpt/pretrained/'
        output_dir = f'{BASEPATH}/results/safe/gpt/pretrained/{frag_method}/beam/'

    else:  # finetuning / from_scratch
        model_path = f'{BASEPATH}/models/safe/gpt/{model_ver}/{frag_method}/best_model'
        output_dir = f'{BASEPATH}/results/safe/gpt/{model_ver}/{frag_method}/beam/'
    
    # Molecule generation using beam-search
    cmd = [
        "python", f"{BASEPATH}/src/func/generation_safe_func.py",
        "--model_path", model_path,
        "--dataset_dir", dataset_dir,
        "--n_samples", str(args.n_samples),
        "--max_length", str(args.max_length),
        "--num_beams", str(args.num_beams),
        "--random_seed", str(args.random_seed),
        "--output_dir", output_dir,
        "--machine_id", str(args.machine_id),
        "--total_machines", str(args.total_machines)
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
            "model_path": model_path,
            "machine_id": args.machine_id,
            "total_machines": args.total_machines,
            "random_seed": args.random_seed,
        },
        record_name=f"generation_time_{args.machine_id}.json",
        predictions_pattern=f"predictions_{args.machine_id}.csv",
    )
    print(f"Saved generation time to: {json_path}")
        
    