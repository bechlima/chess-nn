#!/usr/bin/env python3
"""
Supervised pretraining for ChessNet.

Two data sources:
  1. PGN files (Lichess Elite) — policy + value targets from human games
  2. Lichess eval JSONL — value-only targets from Stockfish evaluations

Usage:
    # Pretrain on PGN files
    python src/pretrain.py --pgn data/raw/lichess_elite/games.pgn --epochs 3

    # Pretrain on Lichess evals (value only)
    python src/pretrain.py --evals data/processed/value_pretrain/lichess_evals.jsonl

    # Both together
    python src/pretrain.py --pgn data/raw/lichess_elite/games.pgn --evals data/processed/value_pretrain/lichess_evals.jsonl

    # Resume from checkpoint
    python src/pretrain.py --pgn games.pgn --resume
"""
import os
import sys
import json
import math
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import ChessNet
from train import save_checkpoint, load_checkpoint, get_latest_checkpoint
from encoding import encode_board
import chess


# ── Dataset classes ────────────────────────────────────────────────────────────

class PGNDataset(Dataset):
    """In-memory dataset from PGN file(s). Use for smaller files."""

    def __init__(self, examples: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        self.states = torch.stack([e[0] for e in examples])
        self.pis    = torch.stack([e[1] for e in examples])
        self.zs     = torch.tensor([e[2] for e in examples], dtype=torch.float32)

    def __len__(self):
        return len(self.zs)

    def __getitem__(self, idx):
        return self.states[idx], self.pis[idx], self.zs[idx]


class EvalDataset(Dataset):
    """
    Dataset from Lichess eval JSONL.
    Each row: {"fen": "...", "cp": 23.0, "value": 0.057}
    Policy target = zeros (value-only training).
    """

    def __init__(self, jsonl_path: str, max_rows: int = None):
        self.examples = []
        print(f"Loading eval dataset from {jsonl_path}...")
        with open(jsonl_path) as f:
            for i, line in enumerate(tqdm(f)):
                if max_rows and i >= max_rows:
                    break
                row = json.loads(line.strip())
                fen = row.get('fen')
                value = row.get('value')
                if fen is None or value is None:
                    continue
                try:
                    board = chess.Board(fen)
                    state = encode_board(board)
                    pi = torch.zeros(config.MOVE_SPACE, dtype=torch.float32)
                    self.examples.append((state, pi, float(value)))
                except Exception:
                    continue

        print(f"Loaded {len(self.examples)} eval positions")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, z = self.examples[idx]
        return state, pi, torch.tensor(z, dtype=torch.float32)


# ── Loss function ───────────────────────────────────────────────────────────────

def pretrain_loss(pred_logits, pred_values, target_pis, target_zs, policy_weight=1.0, value_weight=1.0):
    """
    Combined policy + value loss for supervised pretraining.

    For positions with a played move (pi sums to 1): use cross-entropy.
    For eval-only positions (pi all zeros): skip policy loss.
    """
    # Value loss: always apply
    value_loss = F.mse_loss(pred_values.squeeze(1), target_zs)

    # Policy loss: only for positions with a real policy target
    has_policy = target_pis.sum(dim=1) > 0.5  # batch mask
    if has_policy.any():
        log_probs = F.log_softmax(pred_logits[has_policy], dim=1)
        policy_loss = -(target_pis[has_policy] * log_probs).sum(dim=1).mean()
    else:
        policy_loss = torch.tensor(0.0, device=pred_logits.device)

    total = policy_weight * policy_loss + value_weight * value_loss
    return total, policy_loss.item(), value_loss.item()


# ── Training loop ───────────────────────────────────────────────────────────────

def pretrain_epoch(model, dataloader, optimizer, scheduler=None, log_every=200):
    model.train()
    total_loss = total_policy = total_value = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        states, pis, zs = batch
        states = states.to(config.DEVICE)
        pis    = pis.to(config.DEVICE)
        zs     = zs.to(config.DEVICE)

        pred_logits, pred_values = model(states)
        loss, p_loss, v_loss = pretrain_loss(pred_logits, pred_values, pis, zs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
        optimizer.step()

        total_loss   += loss.item()
        total_policy += p_loss
        total_value  += v_loss
        steps += 1

        if steps % log_every == 0:
            avg = total_loss / steps
            print(f"  step {steps}: loss={avg:.4f} policy={total_policy/steps:.4f} value={total_value/steps:.4f}")

    if scheduler:
        scheduler.step(total_loss / max(steps, 1))

    return {
        'loss': total_loss / max(steps, 1),
        'policy_loss': total_policy / max(steps, 1),
        'value_loss': total_value / max(steps, 1),
        'steps': steps,
    }


def run_pretraining(
    pgn_paths: List[str] = None,
    eval_paths: List[str] = None,
    epochs: int = 3,
    max_pgn_games: int = 50_000,
    max_eval_rows: int = 500_000,
    resume: bool = False,
    checkpoint_name: str = 'pretrain',
):
    """Main pretraining entry point."""
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    # Model
    model = ChessNet(
        board_planes=config.BOARD_PLANES,
        channels=config.CHANNELS,
        res_blocks=config.RES_BLOCKS,
        move_space=config.MOVE_SPACE,
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    start_epoch = 0
    if resume:
        ckpt = get_latest_checkpoint()
        if ckpt:
            start_epoch, _ = load_checkpoint(model, ckpt, optimizer)
            print(f"Resumed from {ckpt} (epoch {start_epoch})")

    # Build datasets
    all_examples = []

    if pgn_paths:
        from data.pgn_loader import stream_pgn_file
        for pgn_path in pgn_paths:
            if not os.path.exists(pgn_path):
                print(f"PGN not found: {pgn_path} — skipping")
                continue
            print(f"Loading PGN: {pgn_path}")
            count = 0
            for ex in tqdm(stream_pgn_file(pgn_path, max_games=max_pgn_games, min_rating=1800, sample_every=2)):
                all_examples.append(ex)
                count += 1
            print(f"  Loaded {count} positions from {pgn_path}")

    eval_datasets = []
    if eval_paths:
        for ep in eval_paths:
            if not os.path.exists(ep):
                print(f"Eval file not found: {ep} — skipping")
                continue
            eval_datasets.append(EvalDataset(ep, max_rows=max_eval_rows))

    if not all_examples and not eval_datasets:
        print("No data found. Provide --pgn or --evals arguments.")
        return

    # Training
    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"\n=== Pretraining Epoch {epoch + 1}/{start_epoch + epochs} ===")
        t0 = time.time()
        epoch_metrics = {}

        # PGN data
        if all_examples:
            dataset = PGNDataset(all_examples)
            loader = DataLoader(
                dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                num_workers=0, drop_last=False
            )
            print(f"PGN dataset: {len(dataset)} positions")
            metrics = pretrain_epoch(model, loader, optimizer, scheduler)
            epoch_metrics['pgn'] = metrics
            print(f"PGN loss={metrics['loss']:.4f} policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}")

        # Eval data
        for i, eval_ds in enumerate(eval_datasets):
            loader = DataLoader(
                eval_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                num_workers=0, drop_last=False
            )
            print(f"Eval dataset {i}: {len(eval_ds)} positions")
            metrics = pretrain_epoch(model, loader, optimizer, scheduler)
            epoch_metrics[f'eval_{i}'] = metrics
            print(f"Eval loss={metrics['loss']:.4f} value={metrics['value_loss']:.4f}")

        elapsed = time.time() - t0
        ckpt_path = save_checkpoint(
            model, optimizer, epoch + 1,
            {**epoch_metrics, 'elapsed': elapsed}
        )
        print(f"Epoch done in {elapsed:.1f}s — checkpoint: {ckpt_path}")

    print("\nPretraining complete!")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Supervised pretraining for ChessNet")
    p.add_argument('--pgn', nargs='+', default=None, help='PGN file(s) for policy pretraining')
    p.add_argument('--evals', nargs='+', default=None, help='JSONL eval file(s) for value pretraining')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-games', type=int, default=50_000)
    p.add_argument('--max-evals', type=int, default=500_000)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pretraining(
        pgn_paths=args.pgn,
        eval_paths=args.evals,
        epochs=args.epochs,
        max_pgn_games=args.max_games,
        max_eval_rows=args.max_evals,
        resume=args.resume,
    )
