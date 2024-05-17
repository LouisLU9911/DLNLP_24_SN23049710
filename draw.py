import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from transformers import BertTokenizer

cwd = os.getcwd()

dataset_path = Path(cwd) / "Datasets"

train_csv_filepath = dataset_path / "train.csv"

df = pd.read_csv(train_csv_filepath)

# Sample dataset
texts = df["full_text"].values

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset and compute token lengths
token_lengths = []
for text in texts:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_lengths.append(len(tokens))

# Print token lengths for debugging
print("Token lengths:", token_lengths)

# Plot the histogram
plt.figure(figsize=(6, 6))
plt.hist(
    token_lengths,
    bins=range(min(token_lengths), max(token_lengths) + 1, 20),
    edgecolor="black",
)
plt.title("Histogram of Token Lengths (BERT Tokenizer)")
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("token_len.png")

# File paths for the four models
model_files = [
    {
        "mcrmse": "A/results/baseline_20240516_171940/baseline_mcrmse.csv",
        "mse_loss": "A/results/baseline_20240516_171940/baseline_training_losses.csv",
    },
    {
        "mcrmse": "A/results/roberta_base_20240517_000445/roberta_base_mcrmse.csv",
        "mse_loss": "A/results/roberta_base_20240517_000445/roberta_base_training_losses.csv",
    },
    {
        "mcrmse": "A/results/electra_base_20240517_000446/electra_base_mcrmse.csv",
        "mse_loss": "A/results/electra_base_20240517_000446/electra_base_training_losses.csv",
    },
    {
        "mcrmse": "A/results/bert_base_cased_20240517_000447/bert_base_cased_mcrmse.csv",
        "mse_loss": "A/results/bert_base_cased_20240517_000447/bert_base_cased_training_losses.csv",
    },
]

# Colors and styles for different models
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
styles = ["-", "--", "-.", ":"]
model_names = ["baseline", "RoBERTa", "ELECTRA", "BERT"]

# Create the plot
fig, ax1 = plt.subplots(figsize=(8, 8))

# Iterate over each model and plot the data
for i, files in enumerate(model_files):
    # Read CSV files
    mcrmse_df = pd.read_csv(files["mcrmse"])
    mseloss_df = pd.read_csv(files["mse_loss"])

    # Extract data
    epochs = mcrmse_df["Epoch"]
    mcrmse = mcrmse_df["MCRMSE"]
    mse_loss = mseloss_df["MSE Loss"]

    # Plot MSE Loss on the primary y-axis
    ax1.plot(
        epochs,
        mse_loss,
        label=f"Model ({model_names[i]}) MSELoss",
        color=colors[i],
        linestyle=styles[i],
        marker="o",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.tick_params(axis="y")
    ax1.grid(True)

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Iterate over each model and plot the data
for i, files in enumerate(model_files):
    # Read CSV files
    mcrmse_df = pd.read_csv(files["mcrmse"])
    mseloss_df = pd.read_csv(files["mse_loss"])

    # Extract data
    epochs = mcrmse_df["Epoch"]
    mcrmse = mcrmse_df["MCRMSE"]

    # Plot MCRMSE on the secondary y-axis
    ax2.plot(
        epochs,
        mcrmse,
        label=f"Model ({model_names[i]}) MCRMSE",
        color=colors[i],
        linestyle=styles[i],
        marker="x",
    )
    ax2.set_ylabel("MCRMSE")
    ax2.tick_params(axis="y")

# Add title and legend
fig.suptitle("Training MSE Loss and Validation MCRMSE for Four Models")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.savefig("learning_curve.png")
