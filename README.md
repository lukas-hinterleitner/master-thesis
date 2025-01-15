# Can gradient-based explanations find training data on similar data points?

## Setup

### Initialize Git Submodules
```bash
git submodule init
git submodule update
```

### Create Virtual Environment
```bash
python3 -m venv master_thesis.venv
source master_thesis.venv/bin/activate
```

Check if virtual environment is used:
```bash
which python
```

### Install necessary packages
```bash
pip install -r requirements.txt
```

## Information
The gradient-similarity results are stored in data/gradient_similarity_*.json.

In the folder data/gradient_similarity_bm25_selected, the results are stored as followed:
```json
[
    "id_para_para": {
        "id_orig_orig": 0.6585582494735718,
        "id_orig_orig": 0.0036986665800213814,
        "id_orig_orig": 0.07739365100860596,
        "id_orig_orig": 0.008699750527739525,
        "id_orig_orig": 0.02530057355761528
    },
    ...
]
```

In the folder data/gradient_similarity_bm25_selected_model_generated, the results are stored as followed:
```json
[
    "id_para_gen": {
        "id_orig_orig": 0.6585582494735718,
        "id_orig_orig": 0.0036986665800213814,
        "id_orig_orig": 0.07739365100860596,
        "id_orig_orig": 0.008699750527739525,
        "id_orig_orig": 0.02530057355761528
    },
    ...
]
```

