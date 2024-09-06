# master-thesis
Explainability of Large Language Models based on input data.

Notes: 
Transfer lima input data: 
```bash
python prepare_lima_data.py -t allenai/eleuther-ai-gpt-neox-20b-pii-special ../../data/lima
```

Install open-instruct:
cd code
pip install -e ../submodules/open-instruct/