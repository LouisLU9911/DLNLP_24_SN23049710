# DLNLP_assignment_24

* Name: Zhaoyan LU
* Student ID: 23049710

## Contents

- [Overview](#Overview)
- [Repo Structure](#Repo-Structure)
- [Datasets](#Datasets)
- [Requirements](#Requirements)
- [Usage](#Usage)

## Overview

This repo is for the final assignment of ELEC0141 Deep Learning for Natural Language Processing Assignment (2024).
This project aims at solving the [Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) task on [Kaggle](https://www.kaggle.com/).

## Repo Structure

```bash
$ tree
.
├── A
│   ├── configs                     # configs for different models
│   │   ├── baseline.json
│   │   ├── bert_base_cased.json
│   │   ├── electra_base.json
│   │   └── roberta_base.json
│   ├── constants.py                # constants
│   ├── logger.py                   # logging
│   ├── models.py                   # models
│   ├── preprocessing.py            # pre-processing
│   └── results                     # experiment results
│       ...
├── Datasets                        # dataset from kaggle
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── draw.py                         # draw pictures
├── environment.yaml
├── learning_curve.png
├── main.py                         # entrypoint of this assignment
├── Makefile
├── README.md
├── requirements.txt
└── token_len.png
```

## Datasets

* [Datasets/train.csv](Datasets/train.csv) ([original link](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data))

You can redownload it by:

```bash
$ make dataset
```

## Requirements

* Miniconda / Anaconda
* Nvidia GPU with cuda 11.8

* `environment.yaml`

```bash
$ cat environment.yaml
name: dlnlp-final-uceezl8
channels:
  - defaults
dependencies:
  - python=3.9
  - black                               # formatter
  - scikit-learn                        # split dataset
  - pandas                              # load dataframe
  - ipython                             # debugging
  - matplotlib                          # draw figures
  - numpy
  - pip
  - pip:
    - -r requirements.txt
```

* `requirements.txt`

```bash
$ cat requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu118
torch
# download the dataset from kaggle
kaggle
# preprocess
datasets
# metrics
evaluate
# fine-tuning and build custom models
transformers[torch]==4.39.3
# download and upload pretrained models
huggingface_hub
ipykernel
```

Create conda env:

```bash
$ make create-env
# or
$ conda env create -f environment.yml
```

## Usage

All options:

```bash
# help
$ python main.py -h
usage: main.py [-h] [-d] [-v] [-c CONFIG]

DLNLP Final Assignment

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           Print lots of debugging statements
  -v, --verbose         Be verbose
  -c CONFIG, --config CONFIG
                        Config for your model
```

* baseline

```bash
$ python main.py
```

* BERT-based pre-trained models

```bash
# BERT
$ python main.py -c bert_base_cased.json
# ELECTRA
$ python main.py -c electra_base.json
# RoBERTa
$ python main.py -c roberta_base.json
```
