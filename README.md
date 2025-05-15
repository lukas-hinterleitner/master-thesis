# Can gradient-based explanations find training data on similar data points?

## Setup

### Initialize Git Submodules
```shell
git submodule init
git submodule update
```

### Create Virtual Environment
The project was developed with Python 3.11 since open-instruct requires it.

```shell
python3 -m venv master_thesis.venv
source master_thesis.venv/bin/activate
```

Check if the virtual environment is used:
```shell
which python
```

### Install necessary packages
```shell
pip install -r requirements.txt
pip install --no-build-isolation traker[fast]==0.3.2
```

## Before Execution
Create a .env file in the folder _application_ _(application/.env)_ with the following content:
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
    }
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
    }
]
```

TODO: the above examples do not represent the new structures with individual layers, etc. --> add to readme

