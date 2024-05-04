# COGS209SP24-Project1-GroupNeuromorphs

## Getting started
1. Follow instructions frombrain-diffusor to create the python environment\
Note: please make sure tokenizers==0.12.1 and transformers==4.19.2. For the diffusion environment, you may use `requirement.txt`

For mac and linux:
```
virtualenv pyenv --python=3.10.12
source pyenv/bin/activate
pip install -r requirements.txt
```
For Windows:
```
virtualenv pyenv --python=3.10.12
pyenv\Scripts\activate
pip install -r requirements.txt
```


2. Download preprocessed eeg data: https://osf.io/anp5v/, unzip "sub01", "sub02", etc under data/things-eeg2_preproc.