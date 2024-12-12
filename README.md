# Can gradient-based explanations find training data on similar data points?

## Setup

### Initialize Git Submodules
```bash
git submodule init
git submodule update
```

### Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Check if virtual environment is used:
```bash
which python
```

### Install necessary packages
```bash
pip install -r requirements.txt
```

### Environment Variables
Before executing Python scripts in this repository, make sure the PYTHONPATH environment 
variable is set correctly. If not, you'll get ModuleNotFoundErrors.

```bash
export PYTHONPATH="${PYTHONPATH}:application"
```

