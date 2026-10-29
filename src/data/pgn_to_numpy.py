"""
One-time preprocessing: PGN files → numpy/memmap shards.

Shard format (fast random access, no Python overhead at training time):
    data/processed/policy_pretrain/shard_NNNN/
        states.npy   [N, 18, 8, 8]  uint8  (board planes, 0 or 1)
        moves.npy    [N]             int16  (move index 0-4095)
        values.npy   [N]             int8   (-1, 0, or 1)

Usage:
    python src/data/pgn_to_numpy.py --pgn data/raw/lichess_elite/lichess_elite_2024-12.pgn
    python src/data/pgn_to_numpy.py --pgn data/raw/lichess_elite/*.pgn --shard-size 100000
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.pgn_loader import stream_pgn_file


def save_numpy_shard(states, moves, values, shard_dir: str):
    """Save a shard as numpy arrays."""
    os.makedirs(shard_dir, exist_ok=True)
    np.save(os.path.join(shard_dir, 'states.npy'), np.array(states, dtype=np.uint8))
    np.save(os.path.join(shard_dir, 'moves.npy'), np.array(moves, dtype=np.int16))
    np.save(os.path.join(shard_dir, 'values.npy'), np.array(values, dtype=np.int8))
    return len(states)


class NumpyChessDataset:
    """
    Fast PyTorch Dataset backed by numpy shards.
    Loads all shards into memory on init (fast random access).
    For very large datasets, use memmap variant instead.
    """

    def __init__(self, shard_dir: str):
        import torch
        from torch.utils.data import Dataset

        self.states_list = []
        self.moves_list = []
        self.values_list = []

        shard_dir = Path(shard_dir)
        shards = sorted(shard_dir.glob('shard_*/'))
        print(f"Loading {len(shards)} shards from {shard_dir}...")
        for shard in shards:
            self.states_list.append(np.load(shard / 'states.npy'))
            self.moves_list.append(np.load(shard / 'moves.npy'))
            self.values_list.append(np.load(shard / 'values.npy'))

        self.states = np.concatenate(self.states_list, axis=0)
        self.moves  = np.concatenate(self.moves_list, axis=0)
        self.values = np.concatenate(self.values_list, axis=0)
        print(f"Loaded {len(self.states)} positions")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        import torch
        import config as cfg
        state = torch.from_numpy(self.states[idx].astype(np.float32))
        pi = torch.zeros(cfg.MOVE_SPACE, dtype=torch.float32)
        move_idx = int(self.moves[idx])
        if 0 <= move_idx < cfg.MOVE_SPACE:
            pi[move_idx] = 1.0
        z = float(self.values[idx])
        return state, pi, z


def process_pgn_to_numpy(
    pgn_path: str,
    output_dir: str,
    shard_size: int = 100_000,
    max_games: int = None,
    min_rating: int = 1800,
    sample_every: int = 2,
):
    """Convert PGN to numpy shards."""
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    states_buf, moves_buf, values_buf = [], [], []
    total = 0

    print(f"Processing {pgn_path} → {output_dir}")

    for state_tensor, pi_tensor, z in tqdm(
        stream_pgn_file(pgn_path, max_games, min_rating, sample_every)
    ):
        # Convert to compact numpy
        state_np = (state_tensor.numpy() * 255).astype(np.uint8)  # float32 → uint8
        move_idx = int(pi_tensor.argmax()) if pi_tensor.sum() > 0 else -1
        value = int(round(z))  # -1, 0, or 1

        states_buf.append(state_np)
        moves_buf.append(move_idx)
        values_buf.append(value)

        if len(states_buf) >= shard_size:
            shard_dir = os.path.join(output_dir, f'shard_{shard_idx:06d}')
            n = save_numpy_shard(states_buf, moves_buf, values_buf, shard_dir)
            total += n
            print(f"  Shard {shard_idx}: {n} positions → {shard_dir}")
            shard_idx += 1
            states_buf, moves_buf, values_buf = [], [], []

    if states_buf:
        shard_dir = os.path.join(output_dir, f'shard_{shard_idx:06d}')
        n = save_numpy_shard(states_buf, moves_buf, values_buf, shard_dir)
        total += n
        print(f"  Shard {shard_idx}: {n} positions → {shard_dir}")

    print(f"Done: {total} positions in {shard_idx + 1} shards")
    return total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pgn', nargs='+', required=True)
    p.add_argument('--out', default='data/processed/policy_pretrain')
    p.add_argument('--shard-size', type=int, default=100_000)
    p.add_argument('--max-games', type=int, default=None)
    p.add_argument('--min-rating', type=int, default=1800)
    p.add_argument('--sample-every', type=int, default=2)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for pgn in args.pgn:
        process_pgn_to_numpy(pgn, args.out, args.shard_size, args.max_games, args.min_rating, args.sample_every)
