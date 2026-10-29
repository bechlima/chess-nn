#!/usr/bin/env python3
"""
Download and inspect Lichess puzzles dataset.
Puzzles are used for tactical benchmarking (not main training).

Usage:
    python scripts/download_puzzles.py --sample 100
"""
import argparse
import csv
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=int, default=100, help='Print this many sample rows')
    p.add_argument('--csv', type=str, default=None, help='Path to downloaded puzzle CSV file')
    return p.parse_args()

def main():
    args = parse_args()

    if args.csv is None:
        print("Lichess puzzles download instructions:")
        print("  1. Go to: https://database.lichess.org/#puzzles")
        print("  2. Download the puzzle CSV file")
        print("  3. Place it in: data/raw/lichess_puzzles/")
        print("  4. Run: python scripts/download_puzzles.py --csv data/raw/lichess_puzzles/lichess_db_puzzle.csv")
        print()
        print("Or use kaggle:")
        print("  kaggle datasets download -d lichess/chess-puzzles")
        print("  unzip chess-puzzles.zip -d data/raw/lichess_puzzles/")
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print(f"Reading {csv_path}...")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < args.sample:
                print(f"Puzzle {i}: FEN={row.get('FEN','?')[:40]}... Moves={row.get('Moves','?')[:30]}... Rating={row.get('Rating','?')}")
            else:
                pass
        print(f"\nTotal rows scanned: {i+1}")

if __name__ == '__main__':
    main()
