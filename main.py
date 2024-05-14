#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import json
import os
import sys
from pathlib import Path


from A.logger import logger, set_log_level
from A.constants import CONFIG_FILENAME, CONFIG_DIR, DEFAULT_BATCH_SIZE
from A.models import ModelA


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
    parser.add_argument(
        "-c",
        "--config",
        help="Config for your model",
        action="store",
        default=CONFIG_FILENAME,
    )

    args, _ = parser.parse_known_args()
    return args


def main():
    from A.preprocessing import data_preprocessing

    try:
        args = setup_parse()
        set_log_level(args.loglevel)

        A_CONFIG_PATH = CWD / "A" / CONFIG_DIR / args.config
        with open(A_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
            logger.info(f"Read task A's config from {A_CONFIG_PATH} successfully!")
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(CWD, cfg)
        logger.debug(f"{X_train.shape=}")
        logger.debug(f"{X_val.shape=}")
        logger.debug(f"{X_test.shape=}")
        logger.debug(f"{y_train.shape=}")
        logger.debug(f"{y_val.shape=}")
        logger.debug(f"{y_test.shape=}")
        logger.debug(f'max_length of tokenizer: {cfg["tokenizer"]["max_length"]}')

        # ======================================================================================================================
        # only one task: Task A
        # build model object.
        model_A = ModelA(cfg)
        batch_size = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        mcrmse_A_train = model_A.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=batch_size,
        )  # Train model based on the training set (you should fine-tune your model based on validation set.)
        mcrmse_A_test = model_A.test(
            X_test=X_test, y_test=y_test, batch_size=batch_size
        )  # Test model based on the test set.
        # Save model
        # model_A.save()
        # Clean up memory/GPU etc...
        model_A.clean()

        # ======================================================================================================================
        # Print out your results with following format:
        logger.info("TA:{},{};".format(mcrmse_A_train, mcrmse_A_test))
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
