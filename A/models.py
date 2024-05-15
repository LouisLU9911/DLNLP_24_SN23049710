#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Models"""
import json
import os
import random
from pathlib import Path
from datetime import datetime

from .constants import (
    DEFAULT_TOKENIZER,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCH,
    DEFAULT_LR,
    DEFAULT_LSTM_HIDDEN_DIM,
    OUTPUT_DIM,
)
from .logger import logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_embeddings(cfg):
    tokenizer_cfg = cfg.get("tokenizer", {})
    pretrained_tokenizer = tokenizer_cfg.get("model", DEFAULT_TOKENIZER)
    logger.debug(f"Use embeddings from {pretrained_tokenizer}")
    tokenizer = AutoModel.from_pretrained(pretrained_tokenizer)
    embeddings = tokenizer.get_input_embeddings()
    return embeddings


class ModelA:
    def __init__(self, workspace: Path, cfg: dict, seed: int = DEFAULT_SEED) -> None:
        # Set random seed for reproducibility
        set_seed(seed)

        # Set CUDA device if specified in environment variables
        cuda_device = os.getenv("CUDA_DEVICE", "0")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.cfg = cfg
        self.embedding = get_embeddings(self.cfg).to(self.device)
        model_cfg = self.cfg.get("model", {})
        backbone = model_cfg.get("backbone", DEFAULT_BACKBONE)
        pooling = model_cfg.get("pooling", None)
        logger.info(
            "=========================================================>"
            f" Build a model using {backbone}..."
        )
        if backbone == "LSTM":
            hidden_dim = model_cfg.get("hidden_dim", DEFAULT_LSTM_HIDDEN_DIM)
            self.model = LSTMModel(self.embedding, hidden_dim, OUTPUT_DIM, pooling).to(
                self.device
            )
        else:
            self.model = PretrainedModel(backbone, OUTPUT_DIM, pooling).to(self.device)
        self.training_losses = []
        self.mcrmses = []
        self.task_name = self.cfg.get("task_name", "default_task")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir: Path = (
            workspace / "A" / "results" / f"{self.task_name}_{timestamp}"
        )
        os.makedirs(self.results_dir, exist_ok=True)

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=DEFAULT_LR,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCH,
    ) -> float:
        """Train the model and return the MCRMSE in the val set."""
        criterion = nn.MSELoss()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

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
            self.model.train()
            for X_batch, y_batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.training_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch + 1}/{epochs}, MSELoss: {avg_epoch_loss}")
            val_mcrmse = self.evaluate(val_loader)
            self.mcrmses.append(val_mcrmse)
            logger.info(f"Validation MCRMSE after epoch {epoch + 1}: {val_mcrmse}")

        return self.evaluate(val_loader)

    def evaluate(self, data_loader) -> float:
        self.model.eval()
        total_squared_errors = torch.zeros(OUTPUT_DIM).to(self.device)
        num_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(data_loader, desc="Evaluating"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                squared_errors = (outputs - y_batch) ** 2
                total_squared_errors += torch.sum(squared_errors, dim=0)
                num_samples += X_batch.size(0)
        mean_squared_errors = total_squared_errors / num_samples
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)
        mcrmse = torch.mean(root_mean_squared_errors).item()
        return mcrmse

    def test(self, X_test, y_test, batch_size=DEFAULT_BATCH_SIZE) -> float:
        """Test the model and return the MCRMSE."""
        test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return self.evaluate(test_loader)

    def save_training_losses(self):
        csv_path = self.results_dir / f"{self.task_name}_training_losses.csv"
        df = pd.DataFrame(
            {
                "Epoch": range(1, len(self.training_losses) + 1),
                "MSE Loss": self.training_losses,
            }
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Training losses saved to {csv_path}")

    def save_mcrmses(self):
        csv_path = self.results_dir / f"{self.task_name}_mcrmse.csv"
        df = pd.DataFrame(
            {
                "Epoch": range(1, len(self.mcrmses) + 1),
                "MCRMSE": self.mcrmses,
            }
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"MCRMSE saved to {csv_path}")

    def save_configs(self):
        dst = self.results_dir / f"{self.task_name}.json"
        with open(dst, "w") as f:
            json.dump(self.cfg, f)
        logger.info(f"task config saved to {dst}")

    def plot_training_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plot_path = self.results_dir / f"{self.task_name}_training_curve.png"
        plt.savefig(plot_path)
        logger.info(f"Training curve saved to {plot_path}")

    def save_model(self):
        """Save model to path."""
        model_path = self.results_dir / f"{self.task_name}_model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    def save(self):
        self.save_configs()
        self.save_training_losses()
        self.save_mcrmses()
        self.plot_training_curve()
        self.save_model()

    def clean(self):
        """Clean up memory/GPU etc..."""
        del self.model
        torch.cuda.empty_cache()
        logger.info("Cleaned up model and GPU memory")


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        weights = self.attention_weights(x)  # [batch_size, seq_len, 1]
        weights = torch.softmax(weights, dim=1)  # [batch_size, seq_len, 1]
        weighted_sum = torch.sum(weights * x, dim=1)  # [batch_size, input_dim]
        return weighted_sum


class LSTMModel(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim, pooling=None):
        super(LSTMModel, self).__init__()
        self.embedding = embeddings
        self.lstm = nn.LSTM(embeddings.embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling.lower() if pooling else None
        if self.pooling == "attention":
            self.attention_pooling = AttentionPooling(hidden_dim)
        logger.debug(self.embedding)
        logger.debug(self.lstm)
        logger.debug(self.linear)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (ht, ct) = self.lstm(x)

        if self.pooling == "mean":
            pooled = torch.mean(lstm_out, dim=1)
            return self.linear(pooled)
        elif self.pooling == "max":
            pooled, _ = torch.max(lstm_out, dim=1)
            return self.linear(pooled)
        elif self.pooling == "attention":
            pooled = self.attention_pooling(lstm_out)
            return self.linear(pooled)
        elif self.pooling is None:
            return self.linear(ht[-1])
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")


class PretrainedModel(nn.Module):
    def __init__(self, backbone, output_dim, pooling=None):
        super(PretrainedModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.linear = nn.Linear(self.backbone.config.hidden_size, output_dim)
        self.pooling = pooling.lower() if pooling else None
        if self.pooling == "attention":
            self.attention_pooling = AttentionPooling(self.backbone.config.hidden_size)
        logger.debug(self.backbone)
        logger.debug(self.linear)
        logger.debug(self.pooling)

    def forward(self, X):
        outputs = self.backbone(X)
        hidden_state = outputs.last_hidden_state

        if self.pooling == "mean":
            pooled = torch.mean(hidden_state, dim=1)
            return self.linear(pooled)
        elif self.pooling == "max":
            pooled, _ = torch.max(hidden_state, dim=1)
            return self.linear(pooled)
        elif self.pooling == "attention":
            pooled = self.attention_pooling(hidden_state)
            return self.linear(pooled)
        elif self.pooling is None:
            return self.linear(
                hidden_state[:, 0, :]
            )  # Use the hidden state of the [CLS] token
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")
