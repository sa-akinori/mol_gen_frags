import os
import subprocess
import itertools

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__=='__main__':
    fd = os.path.dirname(os.path.dirname(__file__))
    gen_method = 'beam'
    slice_name = 'our_slice'
    model_path = f'{fd}/models/safe_gpt/pretrained/safe/{slice_name}/best_model'
    output_dir = f'{fd}/results/safe_gpt/pretrained/safe/{slice_name}/{gen_method}/'
    dataset_dir = f'{fd}/data/safe/{slice_name}/'
    
    # # beam-search
    # cmd = [
    #     "python", f"{fd}/src/func/generation_safe_func.py",
    #     "--model_path", model_path,
    #     "--dataset_dir", dataset_dir,
    #     "--slice_name", slice_name,
    #     "--n_samples", "50",
    #     "--gen_method", "beam",
    #     "--max_length", "200",
    #     "--num_beams", "50",
    #     "--random_seed", "42",
    #     "--output_dir", f'{output_dir}'
    # ]
    # subprocess.run(cmd, check=True)
    
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
            

