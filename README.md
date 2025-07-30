# Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Flukas-hinterleitner%2Fmaster-thesis%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/lukas-hinterleitner/master-thesis/docker-image.yml)

This project calculates similarities between gradients of the LIMA fine-tuning dataset samples and their paraphrased or model-generated versions. It explores whether gradient-based explanations can find training data on similar data points. 
The project is designed to work with the AMD-OLMo-1B-SFT model, but can be adapted for other models as well.
It includes scripts for dataset preparation, analysis, and visualization of results. The project supports both local execution and distributed processing on SLURM clusters.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Distributed Processing](#distributed-processing)
- [Output Files](#output-files)
- [Analysis](#analysis)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Project Structure

```
master-thesis/
├── src/                          # Core source code modules
│   ├── __init__.py              # Package initialization
│   ├── computation.py           # Gradient and similarity computations
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── model_operations.py     # Model loading and operations
│   ├── model.py                # Model wrapper classes
│   ├── paraphrasing.py         # Text paraphrasing utilities
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── storage.py              # File I/O and data persistence
│   ├── utility.py              # General utility functions
│   └── config/                 # Configuration files
├── data/                        # Data storage directory
│   ├── datasets/               # Generated and processed datasets
│   │   ├── paraphrased/        # Paraphrased LIMA dataset
│   │   └── model_generated/    # Model-generated responses
│   ├── gradient_similarity/    # Gradient similarity results
│   │   ├── paraphrased/        # Results for paraphrased data
│   │   ├── model_generated/    # Results for model-generated data
│   │   ├── random_projection/  # Random projection results
│   │   └── *.csv              # Sample size analysis files
│   └── dot_products/           # Dot product computation results
│       ├── paraphrased/        # Paraphrased data dot products
│       └── model_generated/    # Model-generated data dot products
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── analysis.ipynb          # Main analysis notebook
│   ├── datasets_playground.ipynb  # Dataset exploration
│   ├── paraphrased_dataset.ipynb  # Paraphrased data analysis
│   ├── playground_bm25.ipynb   # BM25 retrieval experiments
│   ├── playground_model_generation.ipynb  # Model generation tests
│   └── playground.ipynb        # General experimentation
├── results/                     # Analysis results and visualizations
│   ├── accuracy_per_layer/     # Layer-wise accuracy analysis
│   ├── layer_comparison_full_gradient/  # Full gradient comparisons
│   ├── parameters_per_layer/   # Parameter distribution analysis
│   └── self_similarity_over_layers/    # Self-similarity analysis
├── slurm_scripts/              # SLURM cluster job scripts
│   ├── batch_processing/       # Batch job scripts
│   ├── old_scripts/           # Legacy scripts
│   └── paraphrase_dataset/    # Dataset creation scripts
├── submodules/                 # Git submodules
│   └── open-instruct/         # Open-instruct framework
├── papers/                     # Reference papers and literature
│   ├── allen-ai/              # Allen AI research papers
│   ├── explainability/        # Explainability research
│   └── *.pdf                  # Core ML/NLP papers
├── slides/                     # Presentation slides
├── tests/                      # Test files
├── main.py                     # Main execution script
├── paraphrase.py              # Dataset paraphrasing script
├── requirements.txt           # Python dependencies (pip)
├── pyproject.toml            # Project configuration (uv/pip)
├── uv.lock                   # UV dependency lock file
├── Dockerfile                # Docker container definition
├── README.md                 # This file
└── README_SLURM.md          # SLURM-specific documentation
```

## Features

- **Gradient Similarity Analysis**: Compute similarities between gradients of original and modified (paraphrased/model-generated) training samples
- **Multi-layer Analysis**: Analyze gradient similarities across different transformer layers
- **Random Projection Support**: Efficient dimensionality reduction for large-scale gradient computations
- **Dot Product Computations**: Alternative similarity metric using layer output dot products
- **Dataset Generation**: Automated creation of paraphrased and model-generated datasets
- **Distributed Processing**: SLURM cluster support for large-scale experiments
- **Comprehensive Analysis**: Jupyter notebooks for result visualization and interpretation
- **Docker Support**: Containerized execution environment
- **Multiple Model Support**: Compatible with various transformer models (AMD-OLMo, etc.)

## Requirements

- **Python**: 3.11.* (required by open-instruct submodule)
- **GPU**: NVIDIA GPU with CUDA support (recommended for model inference)
- **Memory**: Minimum 16GB RAM (32GB+ recommended for larger models)
- **Storage**: At least 10GB free space for datasets and results
- **API Keys**: OpenAI API key (for dataset paraphrasing)
- **HuggingFace Token**: For accessing gated models

### Core Dependencies

- PyTorch
- Transformers (HuggingFace)
- TrAKer (influence function computation)
- NumPy, Pandas
- scikit-learn
- Jupyter
- OpenAI API client

## Setup

### Initialize Git Submodules
```shell
git submodule init
git submodule update
```

### Installation
All necessary packages can be installed using either [UV](https://docs.astral.sh/uv/) or [PIP](https://pypi.org/project/pip/).

#### Install necessary packages using UV
```shell
uv sync
uv pip install --no-build-isolation traker[fast]==0.3.2
```

Activate virtual environment: 
```shell
source .venv/bin/activate
```

#### Install necessary packages using PIP
The project was developed with Python 3.11 since open-instruct requires it. So it is necessary to use Python 3.11.*.

Create a virtual environment:
```shell
python3 -m venv .venv
source .venv/bin/activate
```

Check if the virtual environment is used:
```shell
which python
```

Install all necessary packages:
```shell
pip install -r requirements.txt
pip install --no-build-isolation traker[fast]==0.3.2
```

### Docker Setup (Alternative)
Docker images are automatically built and pushed to the Docker Hub repository [here](https://hub.docker.com/r/lukashinterleitner/master-thesis-data-science).
This docker image is then used in the SLURM scripts to run the analysis on a SLURM cluster and support batch processing over multiple nodes.

## Before Execution
Create a .env file in the root folder of the repository with the following content:
```shell
HF_TOKEN=your_token_here

# SET MODEL NAME FROM HUGGING FACE, e.g. amd/AMD-OLMo-1B-SFT
MT_MODEL_NAME=model_name_here 

# SET DEVICE TO RUN DEVICE ON CPU (FOR CPU USE cpu) OR GPU (FOR GPU USE cuda:0 OR cuda:1, WHERE 0 AND 1 ARE GPU IDS)
MT_DEVICE=cuda:0

# SAMPLE SIZE TO SELECT SIZE OF SUBSET OF DATA (DON'T SET IF YOU WANT TO USE WHOLE DATA)
MT_SAMPLE_SIZE=100

# ONLY NEEDED FOR PARAPHRASING (THIS HAS ALREADY BEEN DONE, SO NO NEED TO DECLARE HERE)
OPENAI_API_KEY=your_key_here
```

## Dataset Preparation

Before running the main analysis, you need to prepare the datasets. This doesn't need to be done for the model amd/AMD-OLMo-1B-SFT, as the repository already contains the necessary datasets.
The project supports two types of modified datasets:

### 1. Paraphrased Dataset
Creates paraphrased versions of the original LIMA dataset samples using OpenAI's API.

### 2. Model-Generated Dataset
Creates model-generated responses to paraphrased questions using the specified model.

**Important**: The model-generated dataset requires the paraphrased dataset to exist first.

### Creating Datasets

Use the `paraphrase.py` script to create the required datasets:

```shell
# Create only the paraphrased dataset
python paraphrase.py --dataset-type paraphrased

# Create only the model-generated dataset (requires paraphrased dataset to exist)
python paraphrase.py --dataset-type model-generated

# Create both datasets sequentially
python paraphrase.py --dataset-type both
```

#### Dataset Creation Options:
- `--dataset-type paraphrased`: Creates paraphrased versions of LIMA dataset samples
- `--dataset-type model-generated`: Creates model-generated responses to paraphrased questions (requires paraphrased dataset)
- `--dataset-type both`: Creates both datasets in the correct order

#### Dataset Dependencies:
- **Paraphrased dataset**: Independent, can be created first
- **Model-generated dataset**: Depends on paraphrased dataset existing first

The script will automatically:
- Validate dataset dependencies
- Check for existing datasets
- Provide clear error messages if prerequisites are missing
- Create necessary directory structures

## Usage

The project supports two main types of analysis:
1. Calculating gradient similarities between original and modified datasets
2. Computing dot products between layer outputs of original and modified datasets

### Running the Analysis

Execute the main script with the following arguments:

```shell
python main.py --setting [paraphrased|model-generated] --computation-type [dot-product|gradient-similarity] [--use-random-projection]
```

Arguments:
- `--setting`: Specify whether to use paraphrased or model-generated versions of the dataset
- `--computation-type`: Choose between dot-product or gradient-similarity computation
- `--use-random-projection`: (Optional) Use random projection for gradient similarity calculation to improve efficiency
- `--partition-start` and `--partition-end`: (Optional) Process only a specific partition of the dataset

Examples:
```shell
# Calculate gradient similarity between original and paraphrased samples
python main.py --setting paraphrased --computation-type gradient-similarity

# Calculate dot products between original and model-generated samples
python main.py --setting model-generated --computation-type dot-product

# Calculate gradient similarity with random projection 
python main.py --setting paraphrased --computation-type gradient-similarity --use-random-projection
```

## Distributed Processing

For large datasets, the computation can be distributed across multiple processes. See [README_SLURM.md](README_SLURM.md) for instructions on running the analysis on a SLURM cluster.

## Output Files

Results are stored in the `data/` directory with the following structure:

- `data/datasets/paraphrased/`: Paraphrased dataset files
- `data/datasets/model_generated/`: Model-generated dataset files
- `data/gradient_similarity/paraphrased/`: Gradient similarity results for paraphrased samples
- `data/gradient_similarity/model_generated/`: Gradient similarity results for model-generated samples  
- `data/gradient_similarity/random_projection/`: Results using random projection technique
- `data/dot_products/paraphrased/`: Dot product results for paraphrased samples
- `data/dot_products/model_generated/`: Dot product results for model-generated samples

Results follow a JSON structure containing similarity scores between original and modified samples across different model layers.

## Analysis

After running the computations, analysis notebooks are available in the `notebooks/` directory:
- `analysis.ipynb`: General analysis of similarity results
- `paraphrased_dataset.ipynb`: Analysis specific to paraphrased dataset
- `playground_bm25.ipynb`: BM25 retrieval experiments
- `playground_model_generation.ipynb`: Model generation experiments

### Results Overview

The analysis examines multiple dimensions of gradient similarities:

#### Model Coverage
The project analyzes models from multiple providers:
- AMD models (e.g., AMD-OLMo)
- AllenAI models
- OpenAI Community models (for testing purposes)

#### Analysis Types
Different analytical perspectives are explored:

1. **Accuracy Per Layer**: Measures how accurately gradient similarities can identify matching samples across different model layers.

2. **Layer Comparison**: Analyzes how gradients differ between layers and which layers contribute most to similarity detection.

3. **Parameter Distribution**: Examines the distribution and importance of parameters within each layer.

4. **Self-Similarity**: Measures how similar a layer's gradients are to itself across different samples and modifications.

## Development

### Code Organization

The project follows a modular architecture with clear separation of concerns:

- **src/**: Contains all core functionality organized by purpose
- **Configuration**: Environment variables and model settings in `.env`
- **Testing**: Unit tests in `tests.py` for core functionality (needs to be extended)
- **Documentation**: Comprehensive docstrings and type hints throughout

### Running Tests

Execute the test suite to verify functionality:

```shell
python tests.py
```

## License

This project is part of a Master's thesis research. Please contact the author for usage permissions and citation requirements.

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{hinterleitner2025select,
  title={Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations},
  author={Hinterleitner, Lukas},
  year={2025},
  school={[University of Vienna]},
  type={Master's thesis}
}
```

For SLURM cluster usage instructions, see [README_SLURM.md](README_SLURM.md).

## Create requirements.txt from UV config files
```shell
uv export --format requirements-txt --no-hashes -o requirements.txt
```
