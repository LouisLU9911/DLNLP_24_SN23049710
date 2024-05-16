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
    DEFAULT_LSTM_NUM_LAYERS,
    DEFAULT_PATIENCE,
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


class EarlyStopping:
    def __init__(
        self,
        patience=DEFAULT_PATIENCE,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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
            num_layers = model_cfg.get("num_layers", DEFAULT_LSTM_NUM_LAYERS)
            self.model = LSTMModel(
                self.embedding, hidden_dim, OUTPUT_DIM, pooling, num_layers
            ).to(self.device)
        else:
            self.model = PretrainedModel(backbone, OUTPUT_DIM, pooling).to(self.device)
        self.training_losses = []
        self.mcrmses = []
        self.task_name = self.cfg.get("task_name", "default_task")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir: Path = (
            workspace / "A" / "results" / f"{self.task_name}_{timestamp}"
        )
        self.checkpoint_path = self.results_dir / "checkpoint.pth"
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
        patience=DEFAULT_PATIENCE,
    ) -> float:
        """Train the model and return the MCRMSE in the val set."""
        criterion = nn.MSELoss()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, path=self.checkpoint_path
        )

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

            # Early stopping check
            early_stopping(val_mcrmse, self.model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break

        # Load the best model
        self.model.load_state_dict(torch.load(self.checkpoint_path))

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
    def __init__(self, embeddings, hidden_dim, output_dim, pooling=None, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = embeddings
        self.lstm = nn.LSTM(
            embeddings.embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling.lower() if pooling else None
        self.num_layers = num_layers
        if self.pooling == "attention":
            self.attention_pooling = AttentionPooling(hidden_dim)
        logger.debug(self.embedding)
        logger.debug(self.lstm)
        logger.debug(self.linear)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, (ht, ct) = self.lstm(x, (h0, c0))

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
