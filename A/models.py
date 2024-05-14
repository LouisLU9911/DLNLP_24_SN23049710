#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Models"""

from .constants import (
    DEFAULT_TOKENIZER,
    DEFAULT_BACKBONE,
    DEFAULT_LSTM_HIDDEN_DIM,
    OUTPUT_DIM,
)
from .logger import logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel


def get_embeddings(cfg):
    tokenizer_cfg = cfg.get("tokenizer", {})
    pretrained_tokenizer = tokenizer_cfg.get("model", DEFAULT_TOKENIZER)
    logger.debug(f"Use embeddings from {pretrained_tokenizer}")
    tokenizer = AutoModel.from_pretrained(pretrained_tokenizer)
    embeddings = tokenizer.get_input_embeddings()
    return embeddings


class ModelA:
    def __init__(self, cfg) -> None:
        self.embedding = get_embeddings(cfg)
        model_cfg = cfg.get("model", {})
        backbone = model_cfg.get("backbone", DEFAULT_BACKBONE)
        logger.info(
            "=========================================================>"
            f" Build a model using {backbone}..."
        )
        if backbone == "LSTM":
            hidden_dim = model_cfg.get("hidden_dim", DEFAULT_LSTM_HIDDEN_DIM)
            self.model = LSTMModel(self.embedding, hidden_dim, OUTPUT_DIM)
        else:
            self.model = AutoModel.from_pretrained(backbone)

    def train(self, X_train, y_train, X_val, y_val):
        pass

    def test(self, X_test, y_test):
        pass

    def save(self):
        pass

    def clean(self):
        pass


class LSTMModel(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = embeddings
        self.lstm = nn.LSTM(embeddings.embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        logger.debug(self.embedding)
        logger.debug(self.lstm)
        logger.debug(self.linear)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


class PretrainedModel(nn.Module):
    def __init__(self, backbone, output_dim):
        super(PretrainedModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        pass

    def forward(self, x):
        pass
