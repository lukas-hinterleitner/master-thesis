# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master thesis project analyzing gradient similarities between LIMA fine-tuning dataset samples and their paraphrased/model-generated versions. Explores whether gradient-based explanations can identify training data from similar data points.

**Primary model**: AMD-OLMo-1B-SFT (configurable via environment)

## Commands

### Installation
```bash
# Using UV (preferred)
uv sync --extra build
uv sync --extra build --extra trakfast  # For random projection support

# Activate venv
source .venv/bin/activate
```

### Main Execution
```bash
# Gradient similarity with paraphrased data
python main.py --setting paraphrased --computation-type gradient-similarity

# Dot product with model-generated data
python main.py --setting model-generated --computation-type dot-product

# With random projection
python main.py --setting paraphrased --computation-type gradient-similarity --use-random-projection

# With partitioning (for distributed processing)
python main.py --setting paraphrased --computation-type gradient-similarity --partition-start 0 --partition-end 100
```

### Dataset Preparation
```bash
python paraphrase.py --dataset-type paraphrased      # Create paraphrased dataset
python paraphrase.py --dataset-type model-generated  # Requires paraphrased dataset first
python paraphrase.py --dataset-type both             # Creates both sequentially
```

### Testing
```bash
python tests.py  # Verifies gradient computation correctness
```

### Export requirements
```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

## Architecture

### Core Modules (`src/`)

- **computation.py**: Main computation logic for gradient similarities and dot products. Entry points: `calculate_*_gradient_similarities()`, `calculate_*_layer_dot_products()`
- **dataset.py**: Dataset loading with partition support. All `get_*_dataset()` functions accept `partition_start`/`partition_end` for distributed processing
- **model.py**: Model/tokenizer loading via HuggingFace. Sets chat template automatically
- **model_operations.py**: Gradient extraction via backward pass, model output generation
- **storage.py**: Centralized path management. All I/O paths defined here
- **preprocessing.py**: Uses open-instruct submodule for dataset tokenization

### Configuration (`src/config/`)

All configuration is environment-based via `.env`:
- `HF_TOKEN`: HuggingFace API token
- `MT_MODEL_NAME`: Model name (e.g., `amd/AMD-OLMo-1B-SFT`)
- `MT_DEVICE`: Device (`cpu`, `cuda:0`, etc.)
- `MT_SAMPLE_SIZE`: Optional subset size for testing

### Data Flow

1. Original LIMA dataset → tokenized via open-instruct
2. Paraphrased/model-generated variants created via `paraphrase.py`
3. Gradients computed per sample via backward pass
4. Similarities computed layer-wise (gradient cosine similarity or dot products)
5. Results stored as JSON in `data/` directory

### Key Dependencies

- **TrAKer**: Used for random projection of gradients (dimensionality reduction)
- **open-instruct** (submodule): Provides dataset preprocessing and chat template handling

## Distributed Processing (SLURM)

Generate and submit distributed jobs:
```bash
python slurm_scripts/batch_processing/compute.py --setting paraphrased --computation-type gradient-similarity --num-partitions 10

# Dry run
python slurm_scripts/batch_processing/compute.py --setting paraphrased --computation-type gradient-similarity --num-partitions 10 --dry-run

# Combine results after completion
bash slurm_scripts/batch_processing/combine_all.sh
```

## Notes

- Random seed is set to 42 globally in `main.py` for reproducibility
- Model-generated dataset requires paraphrased dataset to exist first
- Results are stored in `data/gradient_similarity/` and `data/dot_products/`
- Analysis notebooks in `notebooks/`, primary analysis is `analysis.ipynb`
