"""Build an augmented RFFMG training dataset from successful generations.

Extracts molecules that a trained model successfully generated on the robustness
evaluation scenarios (frag_num / dup_frags / attach_point_num) and appends the
(fragment set -> molecule) pairs to the existing ``normal/train.*`` files,
writing the result into a new directory. Only the training split is augmented;
val/test are copied from ``normal`` unchanged. Data preparation only (no training).

The extraction and construction logic follows ``make_datasets.py`` and lives
inline under ``if __name__ == '__main__':`` (no new named functions), reusing the
existing ``load_file`` / ``save_file`` helpers.
"""
import argparse
import ast
import os
import random

import pandas as pd

from func.utility import BASEPATH, load_file, save_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build an augmented RFFMG training dataset from successful generations')
    parser.add_argument('--frag_method', type=str, choices=['rc_cms', 'brics'], required=True, help='fragmentation method')
    parser.add_argument('--model_name', type=str, default='t5chem', help='model name (default: t5chem)')
    parser.add_argument('--model_ver', type=str, default='trained', help='model version (default: trained)')
    parser.add_argument('--scenarios', nargs='+', default=['frag_num', 'dup_frags', 'attach_point_num'],
                        help='robustness scenarios to harvest (default: frag_num dup_frags attach_point_num)')
    parser.add_argument('--n_select', type=int, default=5, help='number of successful molecules to keep per fragment set (default: 5)')
    parser.add_argument('--out_name', type=str, default='augmented', help='output directory name under data/rffmg/<frag_method> (default: augmented)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for molecule sampling (default: 0)')
    args = parser.parse_args()

    frag_method = args.frag_method
    model_name = args.model_name
    model_ver = args.model_ver
    scenarios = args.scenarios
    n_select = args.n_select
    out_name = args.out_name
    seed = args.seed

    # Setting
    data_dir = f'{BASEPATH}/data/rffmg/{frag_method}'
    rng = random.Random(seed)

    # Harvest successful generations per scenario
    added_records = list()
    scenario_counts = dict()
    for scenario in scenarios:
        results_dir = f'{BASEPATH}/results/{model_name}/{model_ver}/rffmg/{frag_method}/beam/{scenario}'
        curated = pd.read_csv(f'{results_dir}/curated_data.tsv', sep='\t', index_col=0)

        scenario_count = 0
        for _, row in curated.iterrows():
            # Skip fragment sets with no valid generation
            if row['nvalid_onfrags'] == 0:
                continue

            smis = ast.literal_eval(row['valid_smis_on_frags'])
            picked = rng.sample(smis, min(n_select, len(smis)))
            added_records.extend([(row['fragment'], smi, scenario) for smi in picked])
            scenario_count += len(picked)

        scenario_counts[scenario] = scenario_count

    added = pd.DataFrame(added_records, columns=['source', 'target', 'scenario'])

    # Remove exact duplicates among the harvested pairs
    n_before_dedup = len(added)
    added = added.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
    n_dup_removed = n_before_dedup - len(added)

    # Exclude pairs already present in normal/train
    train_source = load_file(f'{data_dir}/normal/train.source')
    train_target = load_file(f'{data_dir}/normal/train.target')
    existing = set(zip(train_source, train_target))
    n_before_existing = len(added)
    added = added[added.apply(lambda r: (r['source'], r['target']) not in existing, axis=1)].reset_index(drop=True)
    n_existing_removed = n_before_existing - len(added)

    # Create output directory
    os.makedirs(f'{data_dir}/{out_name}', exist_ok=True)

    # Save augmented train (normal train + harvested pairs)
    save_file('\n'.join(train_source + added['source'].tolist()) + '\n', f'{data_dir}/{out_name}/train.source')
    save_file('\n'.join(train_target + added['target'].tolist()) + '\n', f'{data_dir}/{out_name}/train.target')

    # Copy val/test from normal unchanged
    for split in ['val', 'test']:
        for ext in ['source', 'target']:
            save_file('\n'.join(load_file(f'{data_dir}/normal/{split}.{ext}')) + '\n', f'{data_dir}/{out_name}/{split}.{ext}')

    # Record the harvested pairs
    added.to_csv(f'{data_dir}/{out_name}/added_pairs.csv')

    # Log parameters and metrics
    n_final_train = len(train_source) + len(added)
    log_lines = [
        '# Augmentation parameters',
        f'frag_method: {frag_method}',
        f'model_name: {model_name}',
        f'model_ver: {model_ver}',
        f'n_select: {n_select}',
        f'seed: {seed}',
        f'scenarios: {scenarios}',
        '',
        '# Metrics',
        *[f'picked[{scenario}]: {count}' for scenario, count in scenario_counts.items()],
        f'duplicates_removed: {n_dup_removed}',
        f'existing_in_normal_removed: {n_existing_removed}',
        f'normal_train_size: {len(train_source)}',
        f'final_train_size: {n_final_train}',
    ]
    save_file('\n'.join(log_lines) + '\n', f'{data_dir}/{out_name}/augmentation_log.txt')
