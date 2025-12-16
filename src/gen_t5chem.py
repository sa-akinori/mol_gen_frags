import os
import subprocess
import itertools
from transformers import GenerationConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__=='__main__':
    fd = os.path.dirname(os.path.dirname(__file__))
    gen_method = 'beam'
    slice_name = 'our_slice'
    output_dir = f'{fd}/results/t5chem/trained/dummy/{slice_name}/{gen_method}/'
    model_path = f'{fd}/models/t5chem/trained/dummy/{slice_name}/best_model/'
    dataset_dir = f'{fd}/data/dummy/{slice_name}/'
    
    optim = False
    config = GenerationConfig.from_pretrained(model_path)
    
    # Set default generation_config
    config.temperature = 1.0
    config.do_sample   = False
    config.num_beam_groups   = 1
    config.diversity_penalty = 0.0
    
    # Set generation_config
    config.num_beam_groups   = 1
    config.diversity_penalty = 0.0
    config.save_pretrained(model_path)
    
    if optim:
        
        if gen_method == 'random':
            
            temperatures = [0.01, 0.1, 0.5, 1.0, 1.5]
            for temperature in temperatures:
                
                # Set generation_config
                config.temperature = temperature
                config.do_sample   = True
                config.save_pretrained(model_path)
                
                # Generate compounds
                cmd = [
                    "t5chem", "predict",
                    "--data_dir", dataset_dir,
                    "--model_dir", model_path,
                    "--prediction", f"{output_dir}/temperature_{temperature}/predictions.csv",
                    "--num_beams", '1',
                    "--num_preds", '50',
                    "--batch_size", '8'
                ]
                subprocess.run(cmd, check=True)
                
        
        elif gen_method == 'beam':
            
            num_beam_groups = [1] #1, 2, 5
            div_penalties   = [0.0, 0.3, 0.7, 1.2, 1.5]
            for num_beam_group, div_penalty in list(itertools.product(num_beam_groups, div_penalties)):
                
                if (num_beam_group == 1 and div_penalty != 0.0) or (num_beam_group != 1 and div_penalty == 0.0):
                    continue
                
                print(num_beam_group, div_penalty)
                # Set generation_config
                config.num_beam_groups   = num_beam_group
                config.num_beams = 50
                config.diversity_penalty = div_penalty
                config.save_pretrained(model_path)
                
                # Generate compounds
                cmd = [
                    "t5chem", "predict",
                    "--data_dir", dataset_dir,
                    "--model_dir", model_path,
                    "--prediction", f"{output_dir}/beam_groups_{num_beam_group}/div_penalty_{div_penalty}/predictions.csv",
                    "--num_beams", '50',
                    "--num_preds", '50',
                    "--batch_size", '8'
                ]
                subprocess.run(cmd, check=True)
    
    else:
        # Generate compounds
        cmd = [
            "t5chem", "predict",
            "--data_dir", dataset_dir,
            "--model_dir", model_path,
            "--prediction", f"{output_dir}/predictions.csv",
            "--num_beams", '50',
            "--num_preds", '50',
            "--batch_size", '12'
        ]
        subprocess.run(cmd, check=True)
    
    
    # Return default generation_config
    config.temperature = 1.0
    config.do_sample   = False
    config.num_beam_groups   = 1
    config.diversity_penalty = 0.0
    config.save_pretrained(model_path)
    



