# Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Flukas-hinterleitner%2Fmaster-thesis%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/lukas-hinterleitner/master-thesis/docker-image.yml)
[![DOI](https://zenodo.org/badge/804346968.svg)](https://doi.org/10.5281/zenodo.18346665)

This project calculates similarities between gradients of the LIMA fine-tuning dataset samples and their paraphrased or model-generated versions. It explores whether gradient-based explanations can find training data on similar data points.

The project is designed to work with the AMD-OLMo-1B-SFT model, but can be adapted for other models as well. It includes scripts for dataset preparation, analysis, and visualization of results. The project supports both local execution and distributed processing on SLURM clusters.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Distributed Processing](#distributed-processing)
- [Output Files](#output-files)
- [Analysis](#analysis)
- [Testing](#testing)
- [Author](#author)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Gradient Similarity Analysis**: Compute similarities between gradients of original and modified (paraphrased/model-generated) training samples
- **Multi-layer Analysis**: Analyze gradient similarities across different transformer layers
- **Random Projection Support**: Efficient dimensionality reduction for large-scale gradient computations using TrAKer
- **Dot Product Computations**: Alternative similarity metric using layer output dot products
- **Dataset Generation**: Automated creation of paraphrased and model-generated datasets
- **Distributed Processing**: SLURM cluster support for large-scale experiments
- **Comprehensive Analysis**: Jupyter notebooks for result visualization and interpretation
- **Docker Support**: Containerized execution environment for reproducible results
- **Multiple Model Support**: Compatible with various transformer models (AMD-OLMo, etc.)

## Prerequisites

- **Python**: 3.10 - 3.12 (tested with 3.11)
- **CUDA**: 12.x (for GPU acceleration)
- **Git**: For submodule management
- **HuggingFace Account**: For model access (requires API token)
- **OpenAI API Key**: Only required for creating new paraphrased datasets

### Core Dependencies

- PyTorch (with CUDA support)
- Transformers (HuggingFace)
- TrAKer (for random projection)
- NumPy, Pandas, scikit-learn
- Jupyter, Seaborn, Matplotlib
- OpenAI API client

## Project Structure

```
master-thesis/
├── src/                            # Core source code modules
│   ├── __init__.py                 # Package initialization
│   ├── computation.py              # Gradient and similarity computations
│   ├── dataset.py                  # Dataset loading and preprocessing
│   ├── model.py                    # Model wrapper and loading utilities
│   ├── model_operations.py         # Gradient extraction and model inference
│   ├── paraphrasing.py             # OpenAI-based text paraphrasing
│   ├── preprocessing.py            # Data preprocessing functions
│   ├── storage.py                  # File I/O and path management
│   ├── utility.py                  # General utility functions
│   └── config/                     # Configuration modules
│       ├── __init__.py
│       ├── dataset.py              # Dataset and chat template config
│       ├── device.py               # GPU/CPU device configuration
│       ├── model.py                # Model name configuration
│       └── storage.py              # Storage paths configuration
├── data/                           # Data storage directory (generated)
│   ├── datasets/                   # Generated and processed datasets
│   │   ├── paraphrased/            # Paraphrased LIMA dataset
│   │   └── model_generated/        # Model-generated responses
│   ├── gradient_similarity/        # Gradient similarity results
│   │   ├── paraphrased/            # Results for paraphrased data
│   │   ├── model_generated/        # Results for model-generated data
│   │   └── random_projection/      # Random projection results
│   └── dot_products/               # Dot product computation results
│       ├── paraphrased/            # Paraphrased data dot products
│       └── model_generated/        # Model-generated data dot products
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── analysis.ipynb              # Main comprehensive analysis
│   ├── playground.ipynb            # General experimentation
│   ├── playground_bm25.ipynb       # BM25 retrieval experiments
│   ├── playground_datasets.ipynb   # Dataset exploration
│   ├── playground_model_generation.ipynb                     # Model generation tests
│   ├── playground_preprocessing.ipynb                        # Data preprocessing workflows
│   └── playground_preliminary_analysis_gradient_dot_products.ipynb  # Preliminary analysis
├── results/                        # Analysis results and visualizations
│   ├── accuracy_per_layer/         # Layer-wise accuracy analysis
│   ├── greedy_layer_selection/     # Greedy layer selection results
│   ├── layer_comparison_full_gradient/  # Full gradient comparisons
│   ├── parameters_per_layer/       # Parameter distribution analysis
│   └── self_similarity_over_layers/     # Self-similarity analysis
├── slurm_scripts/                  # SLURM cluster job scripts
│   ├── batch_processing/           # Distributed computation scripts
│   ├── paraphrase_dataset/         # Dataset creation scripts
│   └── old_scripts/                # Legacy scripts
├── submodules/                     # Git submodules
│   └── open-instruct/              # Open-instruct framework for preprocessing
├── papers/                         # Reference papers and literature
├── slides/                         # Presentation slides
├── main.py                         # Main execution script
├── paraphrase.py                   # Dataset paraphrasing script
├── tests.py                        # Unit tests for gradient operations
├── pyproject.toml                  # Project configuration (uv/pip)
├── requirements.txt                # Python dependencies (pip)
├── uv.lock                         # UV dependency lock file
├── Dockerfile                      # Docker container definition
├── docker-bake.hcl                 # Docker build configuration
├── README.md                       # This file
└── README_SLURM.md                 # SLURM-specific documentation
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/lukas-hinterleitner/master-thesis.git
cd master-thesis
```

### 2. Initialize Git Submodules

```bash
git submodule init
git submodule update
```

### 3. Install Dependencies

#### Using UV (Recommended)

```bash
# Install base dependencies
uv sync --extra build

# Install with random projection support (TrAKer fast)
uv sync --extra build --extra trakfast
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

#### Using PIP (Alternative)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install --no-build-isolation traker[fast]==0.3.2
```

### 4. Docker Setup (Alternative)

Docker images are automatically built and pushed to [Docker Hub](https://hub.docker.com/r/lukashinterleitner/master-thesis-data-science). This image is used in SLURM scripts for distributed processing.

```bash
# Pull the latest image
docker pull lukashinterleitner/master-thesis-data-science:latest

# Run interactively
docker run -it --gpus all lukashinterleitner/master-thesis-data-science:latest
```

## Configuration

Create a `.env` file in the repository root:

```bash
# Required: HuggingFace API token for model access
HF_TOKEN=your_huggingface_token_here

# Required: Model name from HuggingFace Hub
MT_MODEL_NAME=amd/AMD-OLMo-1B-SFT

# Required: Device configuration
# Options: cpu, cuda:0, cuda:1, etc.
MT_DEVICE=cuda:0

# Optional: Limit dataset size for testing (omit for full dataset)
MT_SAMPLE_SIZE=100

# Optional: OpenAI API key (only needed for creating new paraphrased datasets)
OPENAI_API_KEY=your_openai_key_here
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token for accessing gated models |
| `MT_MODEL_NAME` | Yes | Full model name from HuggingFace Hub |
| `MT_DEVICE` | Yes | Compute device (`cpu` or `cuda:X`) |
| `MT_SAMPLE_SIZE` | No | Subset size for testing (omit for full dataset) |
| `OPENAI_API_KEY` | No | Only needed for generating paraphrased datasets |

## Dataset Preparation

The project supports two types of modified datasets. For the AMD-OLMo-1B-SFT model, pre-generated datasets are already included in the repository.

### Dataset Types

1. **Paraphrased Dataset**: Paraphrased versions of original LIMA samples using OpenAI's API
2. **Model-Generated Dataset**: Model-generated responses to paraphrased questions

**Important**: The model-generated dataset requires the paraphrased dataset to exist first.

### Creating Datasets

```bash
# Create only the paraphrased dataset
python paraphrase.py --dataset-type paraphrased

# Create only the model-generated dataset (requires paraphrased dataset)
python paraphrase.py --dataset-type model-generated

# Create both datasets sequentially
python paraphrase.py --dataset-type both
```

## Usage

The project supports two main computation types:
1. **Gradient Similarity**: Cosine similarity between gradients of original and modified samples
2. **Dot Product**: Layer output dot products between original and modified samples

### Running Analysis

```bash
python main.py --setting <setting> --computation-type <type> [options]
```

### Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--setting` | `paraphrased`, `model-generated` | Dataset variant to use |
| `--computation-type` | `gradient-similarity`, `dot-product` | Type of computation |
| `--use-random-projection` | Flag | Enable random projection (gradient-similarity only) |
| `--partition-start` | Integer | Start index for partitioned processing |
| `--partition-end` | Integer | End index for partitioned processing |

### Examples

```bash
# Gradient similarity with paraphrased samples
python main.py --setting paraphrased --computation-type gradient-similarity

# Dot products with model-generated samples
python main.py --setting model-generated --computation-type dot-product

# Gradient similarity with random projection for efficiency
python main.py --setting paraphrased --computation-type gradient-similarity --use-random-projection

# Process a specific partition (for distributed processing)
python main.py --setting paraphrased --computation-type gradient-similarity --partition-start 0 --partition-end 100
```

## Distributed Processing

For large-scale experiments, the project supports distributed processing via SLURM. See [README_SLURM.md](README_SLURM.md) for detailed instructions.

### Quick Start

```bash
# Generate and submit all distributed jobs
bash slurm_scripts/batch_processing/compute_all.sh

# Or submit specific job types
python slurm_scripts/batch_processing/compute.py \
    --setting paraphrased \
    --computation-type gradient-similarity \
    --num-partitions 10

# Combine results after all jobs complete
bash slurm_scripts/batch_processing/combine_all.sh
```

## Output Files

Results are stored in the `data/` directory with model-specific subdirectories:

```
data/
├── datasets/
│   ├── paraphrased/                    # Paraphrased dataset files
│   └── model_generated/                # Model-generated dataset files
├── gradient_similarity/
│   ├── paraphrased/<model>/            # Gradient similarities (paraphrased)
│   ├── model_generated/<model>/        # Gradient similarities (model-generated)
│   └── random_projection/<model>/      # Random projection results
└── dot_products/
    ├── paraphrased/<model>/            # Dot products (paraphrased)
    └── model_generated/<model>/        # Dot products (model-generated)
```

Results are stored as JSON files containing similarity scores between original and modified samples across different model layers.

## Analysis

Jupyter notebooks for analysis are available in the `notebooks/` directory:

| Notebook | Description |
|----------|-------------|
| `analysis.ipynb` | Main comprehensive analysis with gradient similarities, layer comparisons, and visualizations |
| `playground.ipynb` | General experimentation with model testing and gradient operations |
| `playground_bm25.ipynb` | BM25 ranking and text similarity experiments |
| `playground_datasets.ipynb` | Dataset exploration and validation |
| `playground_model_generation.ipynb` | Model text generation experiments |
| `playground_preprocessing.ipynb` | Data preprocessing and tokenization workflows |
| `playground_preliminary_analysis_gradient_dot_products.ipynb` | Preliminary gradient dot product analysis |

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

## Testing

Run the test suite to verify gradient computation correctness:

```bash
python tests.py
```

The tests verify:
- Tokenization produces different outputs for different samples
- Gradient computation is idempotent (same input → same gradients)
- Different samples produce different gradients

## Author

**Lukas Hinterleitner**

- GitHub: [@lukas-hinterleitner](https://github.com/lukas-hinterleitner)

## License

This project is part of a master's thesis. Please contact the author for licensing information.

## Acknowledgments

- **AMD** for the AMD-OLMo-1B-SFT model
- **Allen Institute for AI** for the [open-instruct](https://github.com/allenai/open-instruct) framework
- **LIMA Dataset** creators for the training data
- **TrAKer** developers for the gradient tracking and random projection functionality

## Citation

If you use this work in your research, please cite:
!!! Will be provided in the future !!!
```bibtex
@misc{hinterleitner2026selectprojectevaluatinglowerdimensional,
      title={Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations}, 
      author={Lukas Hinterleitner and Loris Schoenegger and Benjamin Roth},
      year={2026},
      eprint={2601.16651},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.16651}, 
}
```

---

## Utility Commands

### Export requirements from UV

```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

### Check Python version

```bash
python --version  # Should be 3.10-3.12
```

### Verify CUDA availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
