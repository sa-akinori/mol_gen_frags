import os
import subprocess
import itertools
import argparse
from transformers import GenerationConfig
from func.utility import BASEPATH

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generate molecules using T5Chem')
    
    # Model parameters
    parser.add_argument('--frag_method', type=str, default='rc_cms', choices=['rc_cms', 'brics'],
                        help='Dataset slice name (default: rc_cms)')
    parser.add_argument('--model_ver', type=str, default='trained', choices=['trained', 'pretrained'],
                        help='Model version (default: trained)')
    
    # Generation parameters
    parser.add_argument('--additional_path', type=str, default='normal', choices=['normal', 'dup_frags', 'frag_num', 'frag_order', 'attach_point_num'],
                        help='Additional path (default: normal)')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples to generate per molecule (default: 50)')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum sequence length (default: 200)')
    parser.add_argument('--num_beams', type=int, default=50,
                        help='Number of beams for beam search (default: 50)')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size (default: 24)')
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    
    gen_method = 'beam'
    frag_method = args.frag_method
    model_name = args.model_ver
    additional_path = args.additional_path
    model_path  = f'{BASEPATH}/models/t5chem/{model_name}/rffmg/{frag_method}/best_model/'
    output_dir  = f'{BASEPATH}/results/t5chem/{model_name}/rffmg/{frag_method}/{gen_method}/{additional_path}'
    dataset_dir = f'{BASEPATH}/data/rffmg/{frag_method}/{additional_path}/'
    os.makedirs(output_dir, exist_ok=True)
        
    # Generate compounds (beam search)
    cmd = [
        "t5chem", "predict",
        "--data_dir", dataset_dir,
        "--model_dir", model_path,
        "--prediction", f"{output_dir}/predictions.csv",
        "--num_beams", str(args.num_beams),
        "--num_preds", str(args.n_samples),
        "--batch_size", str(args.batch_size)
    ]
    subprocess.run(cmd, check=True)

    



