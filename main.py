#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import json
import os
import sys
from pathlib import Path


from A.logger import logger, set_log_level
from A.constants import CONFIG_FILENAME


CWD = Path(os.getcwd())


def setup_parse():
    import argparse
    import logging

    description = "DLNLP Final Assignment"
    parser = argparse.ArgumentParser(description=description)
    parser.set_defaults(whether_output=True)
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()
    return args


def main():
    from A.preprocessing import data_preprocessing

    try:
        args = setup_parse()
        set_log_level(args.loglevel)

        A_CONFIG_PATH = CWD / "A" / CONFIG_FILENAME
        with open(A_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
            logger.info(f"Read task A's config from {A_CONFIG_PATH} successfully!")
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(CWD, cfg)
        logger.debug(f"{X_train.shape=}")
        logger.debug(f"{X_val.shape=}")
        logger.debug(f"{X_test.shape=}")
        logger.debug(f'max_length of tokenizer: {cfg["tokenizer"]["max_length"]}')
        # ======================================================================================================================
        # # only one task: Task A
        # model_A = A(args...)                 # Build model object.
        # acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
        # acc_A_test = model_A.test(args...)   # Test model based on the test set.
        # # Clean up memory/GPU etc...             # Some code to free memory if necessary.

        # # ======================================================================================================================
        # # Print out your results with following format:
        # logger.info("TA:{},{};".format(acc_A_train, acc_A_test))
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
