#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-processing"""

from pathlib import Path

from .logger import logger
from .constants import DATASET_DIR, DATASET_TRAIN_CSV, TEST_SIZE, VAL_SIZE, DEFAULT_SEED


def data_preprocessing(workspace: Path, cfg: dict):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    random_state = cfg.get("seed", DEFAULT_SEED)

    abs_train_csv_path = workspace / DATASET_DIR / DATASET_TRAIN_CSV
    df = pd.read_csv(abs_train_csv_path)
    logger.debug(f"Read DataFrame from {abs_train_csv_path} successfully!")
    X = df["full_text"].values
    y = df.iloc[:, 2:8].values

    # TODO: add tokenizer and
    X_tokens = X

    # Split data into train and remaining
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X_tokens, y, test_size=TEST_SIZE, random_state=random_state
    )

    # Split remaining into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=VAL_SIZE, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
