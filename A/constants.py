#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constants"""

# config for task
CONFIG_DIR = "configs"
CONFIG_FILENAME = "baseline.json"

# default values
DEFAULT_SEED = 2024

# dataset
DATASET_DIR = "Datasets"
DATASET_TRAIN_CSV = "train.csv"

# (test + val) / (test + val + train) = 0.2
TEST_SIZE = 0.2
# val / (test + val) = 0.5
VAL_SIZE = 0.5

# Tokenizer
DEFAULT_TOKENIZER = "google-bert/bert-base-cased"

# Model
DEFAULT_BACKBONE = "LSTM"
OUTPUT_DIM = 6
DEFAULT_BATCH_SIZE = 15
DEFAULT_EPOCH = 10
DEFAULT_LR = 1e-5

# LSTM
DEFAULT_LSTM_HIDDEN_DIM = 512
