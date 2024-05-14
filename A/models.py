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

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=9e-6,
        batch_size=2,
        epochs=3,
    ) -> float:
        """Train the model and return the MCRMSE in the val set."""
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.debug("Generating Datasets...")
        train_dataset = TensorDataset(
            X_train, torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info("Training begins...")
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}"
            )

        return self.evaluate(val_loader)

    def evaluate(self, data_loader) -> float:
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def test(self, X_test, y_test, batch_size=2) -> float:
        """Test the model and return the MSE."""
        test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        return self.evaluate(test_loader)

    def save(self, path: str):
        """Save model to path."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def clean(self):
        """Clean up memory/GPU etc..."""
        del self.model
        torch.cuda.empty_cache()
        logger.info("Cleaned up model and GPU memory")


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
        self.linear = nn.Linear(self.backbone.config.hidden_size, output_dim)
        logger.debug(self.backbone)
        logger.debug(self.linear)

    def forward(self, X):
        outputs = self.backbone(X)
        hidden_state = outputs.last_hidden_state
        return self.linear(
            hidden_state[:, 0, :]
        )  # Use the hidden state of the [CLS] token
