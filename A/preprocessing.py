#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-processing"""

from pathlib import Path

from .logger import logger
from .constants import (
    DATASET_DIR,
    DATASET_TRAIN_CSV,
    TEST_SIZE,
    VAL_SIZE,
    DEFAULT_SEED,
    DEFAULT_TOKENIZER,
)


def get_max_length_from_texts(
    texts,
    tokenizer,
    type,
):
    import numpy as np

    # Tokenize each text and calculate length
    token_lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_lengths.append(len(tokens))

    # Calculate and print statistics
    max_length = max(token_lengths)
    average_length = np.mean(token_lengths)
    percentile_90 = np.percentile(token_lengths, 90)

    logger.debug(f"Max length: {max_length}")
    logger.debug(f"Average length: {average_length}")
    logger.debug(f"90th percentile length: {percentile_90}")
    if type == "max":
        max_len = max_length
    elif type == "avg":
        max_len = average_length
    else:
        max_len = percentile_90

    logger.info(f"Choose {type} for max_len: {max_len}")
    return max_len


def data_preprocessing(workspace: Path, cfg: dict):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    random_state = cfg.get("seed", DEFAULT_SEED)

    abs_train_csv_path = workspace / DATASET_DIR / DATASET_TRAIN_CSV
    df = pd.read_csv(abs_train_csv_path)
    logger.debug(f"Read DataFrame from {abs_train_csv_path} successfully!")
    X = df["full_text"].values
    y = df.iloc[:, 2:8].values

    tokenizer_cfg = cfg.get("tokenizer", {})
    pretrained_tokenizer = tokenizer_cfg.get("model", DEFAULT_TOKENIZER)
    logger.debug(f"Loading tokenizer from {pretrained_tokenizer}")
    enable_padding = tokenizer_cfg.get("padding", "TRUE").upper() == "TRUE"
    logger.debug(f"{enable_padding=}")
    enable_truncation = tokenizer_cfg.get("truncation", "TRUE").upper() == "TRUE"
    logger.debug(f"{enable_truncation=}")
    max_len = tokenizer_cfg.get("max_length", "max")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    if isinstance(max_len, str):
        max_len = get_max_length_from_texts(X.tolist(), tokenizer, type=max_len)

    tokens = tokenizer(
        X.tolist(),
        padding=enable_padding,
        truncation=enable_truncation,
        return_tensors="pt",
        max_length=max_len,
    )

    if pretrained_tokenizer == "google-bert/bert-base-cased":
        X_tokens = tokens["input_ids"]
    else:
        X_tokens = tokens

    # Split data into train and remaining
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X_tokens, y, test_size=TEST_SIZE, random_state=random_state
    )

    # Split remaining into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=VAL_SIZE, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
