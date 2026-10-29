#!/usr/bin/env python3
"""
Sample and inspect the Lichess position evaluations dataset from Hugging Face.
Run this first to verify the schema before downloading large amounts of data.

Usage:
    python scripts/sample_lichess_evals.py
"""
from datasets import load_dataset
from itertools import islice

print("Loading 10 sample rows from Lichess/chess-position-evaluations...")
print("(This will download a small amount of data on first run)\n")

ds = load_dataset(
    "Lichess/chess-position-evaluations",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

for i, row in enumerate(islice(ds, 10)):
    print(f"Row {i}: {row}")

print("\nDone. Schema confirmed.")
print("Run scripts/convert_lichess_evals_stream.py to save a larger sample.")
