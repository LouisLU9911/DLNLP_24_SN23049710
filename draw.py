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
plt.figure(figsize=(10, 6))
plt.hist(
    token_lengths,
    bins=range(min(token_lengths), max(token_lengths) + 1, 1),
    edgecolor="black",
)
plt.title("Histogram of Token Lengths (BERT Tokenizer)")
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("token_len.png")
