import os
import safe
from safe.sample import SAFEDesign
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel
import torch
from typing import List
import pandas as pd
import datasets
from tqdm import tqdm
import argparse
import transformers
import signal
from utility import set_seed
import logging, sys
from rdkit import rdBase
import numpy as np

# Remove existing handlers and only allow CRITICAL or higher
from safe.sample import logger as safe_logger
safe_logger.remove()
safe_logger.add(sys.stderr, level="CRITICAL")

def timeout_handler(signum, frame):
    raise TimeoutError("Execution time exceeded the limit")
    
def main():
    
    parser = argparse.ArgumentParser(description='Generate molecules using SAFE model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str,
                        help='Path to the trained model')
    parser.add_argument('--machine_id', type=int, default=0,
                        help='Machine ID (default: 0)')
    parser.add_argument('--total_machines', type=int, default=1,
                        help='Total number of machines (default: 1)')
    
    # Generation parameters
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples to generate per molecule (default: 50)')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum sequence length (default: 200)')
    parser.add_argument('--num_beams', type=int, default=50,
                        help='Number of beams for beam search (default: 50)')
    parser.add_argument('--random_seed', type=int, default=42)
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Load the trained model and tokenizer
    model = SAFEDoubleHeadsModel.from_pretrained(args.model_path)
    tokenizer = SAFETokenizer.from_pretrained(args.model_path)
    
    # Load dataset
    row_datasets = datasets.load_from_disk(f'{args.dataset_dir}')
    test_dataset = row_datasets['test']
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")
    
    # Generate molecules
    safe_generator = SAFEDesign(model=model, tokenizer=tokenizer)
    
    # Split indices
    indices = np.array_split(range(25000), args.total_machines)[args.machine_id]
    
    gen_mols_results = list()
    error_logs = list() 
    
    for gen_id, (smiles, full_safe, pass_safe, pass_frag) in tqdm(
        enumerate(
            zip(test_dataset['smiles'][indices],
            test_dataset['full_safe'][indices], 
            test_dataset['pass_safe'][indices],
            test_dataset['pass_fragments'][indices])), 
        desc='Generating molecules'
    ):
        
        # Set seed
        set_seed(args.random_seed)
        
        # Set 'time-out'
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        try:

            safe_generator.generation_config.diversity_penalty = 0.0
            kwargs = {'how':'beam', 'num_beams':args.n_samples, 'num_beam_groups':0.0}

            if len(pass_frag.split('.')) == 1:
                gen_mols = safe_generator.scaffold_decoration(
                        scaffold=pass_frag,
                        n_samples_per_trial=50, 
                        n_trials=1,
                        do_sample=False,
                        random_seed=args.random_seed,
                        max_length=args.max_length
                    )
                
            else:
                gen_mols = safe_generator.scaffold_morphing(
                        side_chains=pass_frag.split('.'),
                        n_samples_per_trial=50, 
                        n_trials=1,
                        random_seed=args.random_seed,
                        max_length=args.max_length
                    )
                
            
            gen_mols = ['safe_invalid' if gen_mol == None else gen_mol for gen_mol in gen_mols]
            
            new_gen_mols = [safe_generator.scaffold_decoration(
                            scaffold=gen_mol,
                            n_samples_per_trial=1,
                            how='greedy',
                            do_sample=False,
                            random_seed=args.random_seed,
                            max_length=args.max_length
                            )[0] if '*' in gen_mol and len(gen_mol.split('.')) == 1 else gen_mol for gen_mol in gen_mols]
            new_gen_mols = ['safe_invalid' if new_gen_mol == None else new_gen_mol for new_gen_mol in new_gen_mols]
            new_gen_mols = ['safe_invalid' if len(new_gen_mol.split('.')) != 1 else new_gen_mol for new_gen_mol in new_gen_mols]
            
        except TimeoutError as e:
            new_gen_mols = ['time_out'] * args.n_samples
            error_logs.append([smiles, full_safe, pass_safe, pass_frag, 'TimeoutError', str(e)])
            
        except Exception as e:
            new_gen_mols = ['error'] * args.n_samples
            error_logs.append([smiles, full_safe, pass_safe, pass_frag, type(e).__name__, str(e)])
        
        finally:
            signal.alarm(0) 
            gen_mols_results.append([smiles, full_safe, pass_safe, pass_frag] + new_gen_mols)
            
    # Create DataFrame
    columns = ['target', 'full_safe', 'pass_safe', 'fragment'] + [f'prediction_{i+1}' for i in range(args.n_samples)]
    gen_mols_df = pd.DataFrame(gen_mols_results, columns=columns)
    
    # Save results
    os.makedirs(f'{args.output_dir}/', exist_ok=True)
    gen_mols_df.to_csv(f'{args.output_dir}/predictions_{MACHINE_ID}.csv')
    print(f"Saved SMILES predictions to: {args.output_dir}/predictions.csv")
    print("Generation completed!")

    # Create error_log df
    if error_logs:
        error_df = pd.DataFrame(error_logs, columns=['target', 'full_safe', 'pass_safe', 'fragment', 'error_type', 'error_message'])
    else:
        # Create an empty DataFrame even if there are no errors
        error_df = pd.DataFrame(columns=['target', 'full_safe', 'pass_safe', 'fragment', 'error_type', 'error_message'])
    error_df.to_csv(f'{args.output_dir}/error_logs_{MACHINE_ID}.csv')

if __name__ == '__main__':
    main()
