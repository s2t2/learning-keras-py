# Getting started with Machine Learning in Python

## Configuring Local Dev Environment

Install anaconda from source: https://www.anaconda.com/download/#macos then restart terminal and check to see if it is installed:

```sh
conda --version #> conda 4.5.11
which conda #> /anaconda3/bin/conda
```

Create a new virtual environment, named something like `ml-env-1`, with the necessary python packages:

```sh
conda create -n mjr-ml4 python=3.6 # installing packages here seems to not work, also version 3.7 not yet supported by tensorflow, so use python 3.6
conda env list #> you should see your new environment included
```

Enter the virtual environemnt:

```sh
conda activate mjr-ml4  # ... to deactivate: conda deactivate

which python #> /anaconda3/envs/mjr-ml4/bin/python
python --version #> Python 3.6.7 :: Anaconda, Inc.

which pip #> /anaconda3/envs/mjr-ml4/bin/pip
pip --version #> pip 18.1 from /anaconda3/envs/mjr-ml4/lib/python3.6/site-packages/pip (python 3.6)
```

Install package dependencies inside the virtual environment:

```sh
pip install keras tensorflow matplotlib

pip list #> should see both, with supporting packages
```



## Creating a Model

```sh
python my_model.py
```
