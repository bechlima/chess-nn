#!/usr/bin/env python3
"""
Mini AlphaZero Chess Engine — Main Training Loop
"""
import os
import sys
import copy
import time
import json
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import config
from model import ChessNet
from self_play import self_play_game, ReplayBuffer
from train import train_steps, save_checkpoint, load_checkpoint, get_latest_checkpoint
from arena import play_match, vs_random


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_or_create_model():
    model = ChessNet(
        board_planes=config.BOARD_PLANES,
        channels=config.CHANNELS,
        res_blocks=config.RES_BLOCKS,
        move_space=config.MOVE_SPACE,
    ).to(config.DEVICE)

    ckpt_path = get_latest_checkpoint()
    start_iter = 0
    if ckpt_path:
        log(f"Loading checkpoint: {ckpt_path}")
        start_iter, metrics = load_checkpoint(model, ckpt_path)
        log(f"Resumed from iteration {start_iter}")
    else:
        log("No checkpoint found. Starting fresh.")

    return model, start_iter


def run_training(num_iterations: int = 100, num_self_play_games: int = None):
    if num_self_play_games is None:
        num_self_play_games = config.SELF_PLAY_GAMES_PER_ITER

    os.makedirs(config.SELF_PLAY_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    model, start_iter = load_or_create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    replay_buffer = ReplayBuffer(max_size=config.REPLAY_BUFFER_SIZE)
    metrics_history = []

    log(f"Starting training for {num_iterations} iterations")
    log(f"Self-play games/iter: {num_self_play_games}, Train steps/iter: {config.TRAIN_STEPS_PER_ITER}")
    log(f"MCTS sims: {config.MCTS_SIMS}, Buffer size: {config.REPLAY_BUFFER_SIZE}")

    for iteration in range(start_iter, start_iter + num_iterations):
        iter_start = time.time()
        log(f"\n=== Iteration {iteration + 1} ===")

        # 1. Self-play (parallel if multiple cores available)
        import multiprocessing as _mp
        num_workers = max(1, (_mp.cpu_count() or 4) - 1)
        log(f"Generating {num_self_play_games} self-play games ({num_workers} workers)...")
        model.eval()
        if num_workers > 1:
            from self_play_parallel import parallel_self_play
            new_examples = parallel_self_play(
                model,
                num_workers=num_workers,
                total_games=num_self_play_games,
                num_simulations=config.MCTS_SIMS,
            )
        else:
            new_examples = []
            for game_idx in tqdm(range(num_self_play_games), desc="Self-play", leave=False):
                examples = self_play_game(model, num_simulations=config.MCTS_SIMS)
                new_examples.extend(examples)

        replay_buffer.add(new_examples)
        log(f"Buffer: {len(replay_buffer)} examples (+{len(new_examples)} new)")

        # Save self-play data
        sp_path = os.path.join(config.SELF_PLAY_DIR, f"iter_{iteration:04d}.pt")
        torch.save(new_examples, sp_path)

        # 2. Train
        if len(replay_buffer) < config.BATCH_SIZE:
            log(f"Buffer too small ({len(replay_buffer)} < {config.BATCH_SIZE}), skipping training.")
            continue

        log(f"Training {config.TRAIN_STEPS_PER_ITER} steps...")
        old_model = copy.deepcopy(model)
        train_metrics = train_steps(
            model,
            replay_buffer.get_all(),
            num_steps=config.TRAIN_STEPS_PER_ITER,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        log(f"loss={train_metrics['loss']:.4f} | policy={train_metrics['policy_loss']:.4f} | value={train_metrics['value_loss']:.4f}")

        # 3. Arena evaluation
        log(f"Arena: new model vs old model ({config.ARENA_GAMES} games)...")
        model.eval()
        old_model.eval()
        arena_stats = play_match(model, old_model, num_games=config.ARENA_GAMES, num_simulations=config.MCTS_SIMS)
        log(f"Arena: W={arena_stats['wins']} D={arena_stats['draws']} L={arena_stats['losses']} score={arena_stats['score']:.3f}")

        # 4. Accept or reject new model
        always_accept = iteration < config.ARENA_ALWAYS_ACCEPT_ITERS
        accepted = always_accept or arena_stats['score'] >= config.ARENA_THRESHOLD
        if accepted:
            log(f"New model ACCEPTED (always_accept={always_accept}, score={arena_stats['score']:.3f})")
        else:
            log(f"New model REJECTED (score={arena_stats['score']:.3f} < threshold {config.ARENA_THRESHOLD})")
            model.load_state_dict(old_model.state_dict())

        # 5. Checkpoint
        ckpt_path = save_checkpoint(model, optimizer, iteration + 1, train_metrics)
        log(f"Checkpoint saved: {ckpt_path}")

        # 6. Metrics
        elapsed = time.time() - iter_start
        iter_metrics = {
            'iteration': iteration + 1,
            'examples_generated': len(new_examples),
            'buffer_size': len(replay_buffer),
            'train_loss': train_metrics['loss'],
            'policy_loss': train_metrics['policy_loss'],
            'value_loss': train_metrics['value_loss'],
            'arena_score': arena_stats['score'],
            'arena_wins': arena_stats['wins'],
            'arena_draws': arena_stats['draws'],
            'arena_losses': arena_stats['losses'],
            'model_accepted': accepted,
            'elapsed_seconds': elapsed,
        }
        metrics_history.append(iter_metrics)

        metrics_path = os.path.join(config.DATA_DIR, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)

        log(f"Iteration done in {elapsed:.1f}s")

    log("\nTraining complete!")
    return metrics_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini AlphaZero Chess Engine")
    parser.add_argument('--iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=None, help='Self-play games per iteration')
    parser.add_argument('--sims', type=int, default=None, help='MCTS simulations per move')
    args = parser.parse_args()

    if args.sims:
        config.MCTS_SIMS = args.sims

    run_training(num_iterations=args.iterations, num_self_play_games=args.games)
