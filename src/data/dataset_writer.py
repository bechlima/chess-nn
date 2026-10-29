"""
Saves and loads processed training examples efficiently.

Format: list of (state_tensor, pi_tensor, z_float) saved as .pt files in shards.
"""
import os
import torch
from typing import List, Tuple, Iterator
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_shard(
    examples: List[Tuple[torch.Tensor, torch.Tensor, float]],
    path: str,
):
    """Save a list of (state, pi, z) examples to a .pt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Stack for efficient storage
    states = torch.stack([e[0] for e in examples])
    pis = torch.stack([e[1] for e in examples])
    zs = torch.tensor([e[2] for e in examples], dtype=torch.float32)
    torch.save({'states': states, 'pis': pis, 'zs': zs, 'count': len(examples)}, path)
    return len(examples)


def load_shard(path: str) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Load examples from a .pt shard file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    states = data['states']
    pis = data['pis']
    zs = data['zs']
    return [(states[i], pis[i], zs[i].item()) for i in range(len(zs))]


def stream_shards(shard_dir: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Yield examples from all .pt shards in a directory."""
    shard_dir = Path(shard_dir)
    shards = sorted(shard_dir.glob('*.pt'))
    for shard_path in shards:
        for example in load_shard(str(shard_path)):
            yield example


def process_pgn_to_shards(
    pgn_path: str,
    output_dir: str,
    shard_size: int = 50_000,
    max_games: int = None,
    min_rating: int = 0,
    sample_every: int = 2,
):
    """
    Parse a PGN file and save examples in shards.

    Args:
        pgn_path: input PGN file
        output_dir: directory to write .pt shards
        shard_size: examples per shard file
        max_games: max games to process
        min_rating: minimum player rating filter
        sample_every: use every Nth position
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from data.pgn_loader import stream_pgn_file

    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    buffer = []
    total = 0

    print(f"Processing {pgn_path} -> {output_dir}")
    for state, pi, z in tqdm(stream_pgn_file(pgn_path, max_games, min_rating, sample_every)):
        buffer.append((state, pi, z))
        if len(buffer) >= shard_size:
            path = os.path.join(output_dir, f'shard_{shard_idx:04d}.pt')
            save_shard(buffer, path)
            total += len(buffer)
            print(f"  Saved shard {shard_idx}: {len(buffer)} examples -> {path}")
            shard_idx += 1
            buffer = []

    if buffer:
        path = os.path.join(output_dir, f'shard_{shard_idx:04d}.pt')
        save_shard(buffer, path)
        total += len(buffer)
        print(f"  Saved shard {shard_idx}: {len(buffer)} examples -> {path}")

    print(f"Done: {total} total examples in {shard_idx + 1} shards")
    return total
