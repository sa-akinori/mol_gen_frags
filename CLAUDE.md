# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements molecular fragment generation for machine learning using two main approaches:
- **SAFE (Sequential Attachment-based Fragment Embedding)**: Uses the safe-mol library for molecular encoding/decoding
- **T5Chem**: Uses a T5-based transformer model for chemical language modeling

The pipeline processes chemical compounds from ChEMBL31 dataset through curation, fragmentation, and dataset preparation for training generative models.

## Environment Setup

Multiple conda environments are required for different components:

### SAFE Environment
```bash
conda create -n env_safe python=3.12
conda activate env_safe
pip install safe-mol
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.46.1
```

### T5Chem Environment
```bash
conda create -n t5chem python=3.12
conda activate t5chem
pip install t5chem
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install rdkit==2025.3.3
pip install pandas
pip install joblib
```

### General Requirements
```bash
pip install rdkit>=2024 pandas numpy joblib
```

## Core Architecture

### Data Processing Pipeline
1. **Curation** (`src/curate_datasets.py`): Filters and standardizes ChEMBL compounds
   - Converts SMILES to canonical form using RDKit
   - Applies molecular weight and heavy atom filters
   - Removes structural alerts (Glaxo, PAINS filters)

2. **Fragmentation** (`src/func/fragmentation.py`): Core fragmentation algorithm
   - `RandomFragmentize()`: Breaks molecules at eligible bonds (SP3-SP3 carbons, ring-ring connections)
   - `Smi2Sentences()`: Generates fragment→molecule training pairs
   - Configurable via `Smi2SentenceOpt` dataclass

3. **Dataset Generation**:
   - **SAFE**: `src/safe_frags.py` - Encodes molecules using SAFE representation
   - **T5Chem**: `src/t5chem_frags.py` - Creates sentence-based fragment training data
   - **Splitting**: `src/make_datasets.py` - Creates train/val/test splits for both approaches

### Key Components

- **Fragmentation Engine** (`src/func/fragmentation.py:28-59`): Core molecule fragmentation logic
- **SAFE Encoding** (`src/safe_frags.py:26-31`): Converts SMILES to SAFE strings with error handling
- **Utility Functions** (`src/func/utility.py`): Logging and folder management utilities

## Common Development Tasks

### Run Complete Pipeline
```bash
# 1. Curate ChEMBL dataset
python src/curate_datasets.py

# 2. Generate SAFE fragments
python src/safe_frags.py

# 3. Generate T5Chem fragments  
python src/t5chem_frags.py

# 4. Create train/test splits
python src/make_datasets.py
```

### Training Models
- **SAFE**: Use notebooks `run_safe.ipynb` with appropriate conda environment
- **T5Chem**: Use `run_t5chem.ipynb` for training with t5chem library

## Data Structure

```
data/
├── curated/           # Processed ChEMBL compounds
├── safe/             # SAFE-encoded data
│   └── for_gen/      # Train/val/test splits
└── t5chem/           # T5Chem sentence data
    └── for_gen/      # Source/target files
```

## Key Configuration

Fragmentation parameters in `Smi2SentenceOpt` (`src/func/fragmentation.py:173-183`):
- `fragmentRatio`: Proportion of eligible bonds to cut (default: 0.6)
- `bigRingThres`: Ring size threshold for fragmentation (default: 7)
- `nFragmentPatterns`: Number of fragmentation patterns per molecule (default: 5)
- `uppMolSizeToFragSize`: Fragment-to-molecule size ratio limit (default: 1.75)