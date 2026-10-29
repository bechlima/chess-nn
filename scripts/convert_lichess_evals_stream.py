#!/usr/bin/env python3
"""
Stream Lichess position evaluations from Hugging Face and save as JSONL.
This creates the value pretraining dataset.

Usage:
    python scripts/convert_lichess_evals_stream.py --n 100000
    python scripts/convert_lichess_evals_stream.py --n 1000000
"""
import argparse
import json
import math
from pathlib import Path
from itertools import islice
from tqdm import tqdm

def cp_to_value(cp: float, scale: float = 400.0) -> float:
    """Convert centipawn eval to value in [-1, 1]."""
    return math.tanh(cp / scale)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100_000, help='Number of rows to stream')
    p.add_argument('--out', type=str, default='data/processed/value_pretrain/lichess_evals.jsonl')
    p.add_argument('--max_cp', type=float, default=3000.0, help='Clip evals beyond this (centipawns)')
    return p.parse_args()

def main():
    args = parse_args()
    from datasets import load_dataset

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {args.n} rows from Lichess/chess-position-evaluations...")

    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    written = 0
    skipped = 0

    with out_path.open('w') as f:
        for row in tqdm(islice(ds, args.n), total=args.n):
            # Inspect schema on first row
            if written == 0 and skipped == 0:
                print(f"\nSchema: {list(row.keys())}")
                print(f"Sample row: {row}\n")

            # Extract FEN and eval — handle both possible schemas
            fen = row.get('fen') or row.get('FEN')
            cp = row.get('cp') or row.get('eval') or row.get('score')

            if fen is None or cp is None:
                skipped += 1
                continue

            # Skip mate scores (they come as very large numbers or strings)
            try:
                cp = float(cp)
            except (ValueError, TypeError):
                skipped += 1
                continue

            # Clip extreme evals
            cp = max(-args.max_cp, min(args.max_cp, cp))
            value = cp_to_value(cp)

            record = {
                'fen': fen,
                'cp': cp,
                'value': value,
            }
            f.write(json.dumps(record) + '\n')
            written += 1

    print(f"\nDone: {written} rows written to {out_path}")
    if skipped:
        print(f"Skipped: {skipped} rows (missing fields or mate scores)")

if __name__ == '__main__':
    main()
