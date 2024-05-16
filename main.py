#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import json
import os
import sys
from pathlib import Path


from A.logger import logger, set_log_level
from A.constants import (
    CONFIG_FILENAME,
    CONFIG_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCH,
    DEFAULT_LR,
)
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
            (
                X_train,
                train_masks,
                X_val,
                val_masks,
                X_test,
                test_masks,
                y_train,
                y_val,
                y_test,
            ) = data_preprocessing(CWD, cfg)

        # ======================================================================================================================
        # only one task: Task A
        # build model object.
        model_A = ModelA(CWD, cfg)
        batch_size = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        epoch = cfg.get("epoch", DEFAULT_EPOCH)
        lr = cfg.get("lr", DEFAULT_LR)
        logger.debug(f"{batch_size=}")
        logger.debug(f"{epoch=}")
        logger.debug(f"{lr=}")
        mcrmse_A_train = model_A.train(
            X_train=X_train,
            train_masks=train_masks,
            X_val=X_val,
            val_masks=val_masks,
            y_train=y_train,
            y_val=y_val,
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epoch,
        )  # Train model based on the training set (you should fine-tune your model based on validation set.)
        # Save model
        model_A.save()
        mcrmse_A_test = model_A.test(
            X_test=X_test, test_masks=test_masks, y_test=y_test, batch_size=batch_size
        )  # Test model based on the test set.
        # Clean up memory/GPU etc...
        model_A.clean()

        # ======================================================================================================================
        # Print out your results with following format:
        final_result = "TA:{},{};".format(mcrmse_A_train, mcrmse_A_test)
        with open(model_A.results_dir / "result.txt", "w") as f:
            f.write(final_result)
        logger.info(final_result)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
