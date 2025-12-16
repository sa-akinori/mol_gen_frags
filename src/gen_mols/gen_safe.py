import os
import subprocess
import itertools
import argparse
from func.utility import BASEPATH

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--frag_method', type=str, default='brics', choices=['brics', 'rc_cms'],
                        help='Fragmentation method (default: brics)')
    parser.add_argument('--model_ver', type=str, default='trained', choices=['trained', 'pretrained'],
                        help='Phase name (default: trained)')
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
    
    if model_ver == 'trained':
        model_path = f'{BASEPATH}/models/safe_gpt/trained/safe/{frag_method}/best_model'
        output_dir = f'{BASEPATH}/results/safe_gpt/trained/safe/{frag_method}/beam/'
        
    elif model_ver == 'pretrained':
        model_path = f'{BASEPATH}/models/safe_gpt/pretrained/'
        output_dir = f'{BASEPATH}/results/safe_gpt/pretrained/safe/{frag_method}/beam/'
    
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
    subprocess.run(cmd, check=True)
        
    