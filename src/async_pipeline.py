"""
Async self-play + training pipeline.

Trainer and workers run concurrently:
- Workers generate self-play games continuously on CPU
- Workers poll checkpoints/version.txt to reload updated weights
- Workers send examples to trainer via Queue
- Trainer trains continuously on MPS/GPU

Usage:
    from async_pipeline import run_async_training
    run_async_training(num_workers=4, total_iterations=100)

Or from command line:
    python src/async_pipeline.py --workers 4 --iterations 100
"""
import os
import sys
import time
import json
import copy
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import ChessNet
from self_play import self_play_game
from train import compute_loss, compute_wdl_loss, save_checkpoint, _reconstruct_pi


LATEST_CKPT = os.path.join(config.CHECKPOINTS_DIR, "latest.pt")
VERSION_FILE = os.path.join(config.CHECKPOINTS_DIR, "version.txt")
SAVE_EVERY = 50  # save checkpoint every N train steps


def _write_checkpoint(model: torch.nn.Module, step: int):
    """Write latest.pt and version.txt atomically."""
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    tmp = LATEST_CKPT + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, LATEST_CKPT)
    Path(VERSION_FILE).write_text(str(step))


def _worker_process(worker_id: int, queue: mp.Queue, num_simulations: int):
    """
    Worker process: continuously generates self-play games.
    Polls version.txt to reload model when trainer updates it.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

    model = ChessNet(
        board_planes=config.BOARD_PLANES,
        channels=config.CHANNELS,
        res_blocks=config.RES_BLOCKS,
        move_space=config.MOVE_SPACE,
    ).to(config.INFERENCE_DEVICE)
    model.eval()

    local_version = None
    games_played = 0

    while True:
        # Poll for model update
        try:
            version = Path(VERSION_FILE).read_text().strip() if Path(VERSION_FILE).exists() else None
            if version != local_version and Path(LATEST_CKPT).exists():
                state = torch.load(LATEST_CKPT, map_location=config.INFERENCE_DEVICE, weights_only=False)
                model.load_state_dict(state)
                local_version = version
        except Exception:
            pass  # Keep using old model if reload fails

        # Play one game
        try:
            examples = self_play_game(model, num_simulations=num_simulations, add_noise=True)
            if examples:
                queue.put(examples)
                games_played += 1
        except Exception as e:
            print(f"[worker {worker_id}] self_play_game error: {e}", flush=True)

        # Brief pause to avoid hammering the queue
        if queue.qsize() > 500:
            time.sleep(1.0)


def run_async_training(
    num_workers: int = None,
    total_steps: int = 10_000,
    num_simulations: int = None,
    min_buffer_size: int = None,
    train_every: int = 1,
):
    """
    Run async self-play + training.

    Args:
        num_workers: parallel self-play workers (default: cpu_count - 1)
        total_steps: total training gradient steps
        num_simulations: MCTS sims per move for workers
        min_buffer_size: start training when buffer has this many examples
        train_every: train step every N examples received
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)
    if num_simulations is None:
        num_simulations = config.MCTS_SIMS
    if min_buffer_size is None:
        min_buffer_size = config.BATCH_SIZE * 4

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    print(f"Async pipeline: {num_workers} workers, device={config.DEVICE}")
    print(f"Training for {total_steps} steps, MCTS sims={num_simulations}")

    # Build trainer model on GPU
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
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Write initial checkpoint for workers
    _write_checkpoint(model, step=0)

    # Start workers
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(maxsize=1000)

    workers = []
    for i in range(num_workers):
        p = ctx.Process(
            target=_worker_process,
            args=(i, queue, num_simulations),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"Started {num_workers} workers")

    # Replay buffer (trainer-owned)
    replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)

    step = 0
    total_examples = 0
    metrics_history = []
    t_start = time.time()

    print(f"Waiting for {min_buffer_size} examples before training...")

    try:
        while step < total_steps:
            # Drain queue
            drained = 0
            while not queue.empty() and drained < 100:
                try:
                    examples = queue.get_nowait()
                    replay_buffer.extend(examples)
                    total_examples += len(examples)
                    drained += 1
                except Exception:
                    break

            if len(replay_buffer) < min_buffer_size:
                time.sleep(0.1)
                continue

            # Sample batch and train
            import random
            batch_size = min(config.BATCH_SIZE, len(replay_buffer))
            indices = random.sample(range(len(replay_buffer)), batch_size)
            batch = [replay_buffer[i] for i in indices]

            states = torch.stack([e[0] for e in batch]).to(config.DEVICE)
            pis    = torch.stack([_reconstruct_pi(e[1], config.MOVE_SPACE) for e in batch]).to(config.DEVICE)
            zs     = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(config.DEVICE)

            model.train()
            pred_logits, pred_values = model(states)
            if getattr(model, 'use_wdl', False):
                loss, p_loss, v_loss = compute_wdl_loss(pred_logits, pred_values, pis, zs)
            else:
                loss, p_loss, v_loss = compute_loss(pred_logits, pred_values, pis, zs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            step += 1

            # Save checkpoint periodically
            if step % SAVE_EVERY == 0:
                _write_checkpoint(model, step)
                elapsed = time.time() - t_start
                scheduler.step(loss.item())
                print(f"Step {step}/{total_steps} | loss={loss.item():.4f} "
                      f"| buffer={len(replay_buffer)} | examples={total_examples} "
                      f"| {elapsed:.0f}s")

                metrics_history.append({
                    'step': step,
                    'loss': loss.item(),
                    'policy_loss': p_loss,
                    'value_loss': v_loss,
                    'buffer_size': len(replay_buffer),
                    'total_examples': total_examples,
                    'elapsed': elapsed,
                })

                metrics_path = os.path.join(config.DATA_DIR, 'async_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_history, f, indent=2)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Save final checkpoint
        _write_checkpoint(model, step)
        save_checkpoint(model, optimizer, step, {})
        print(f"\nDone. {step} steps, {total_examples} examples. Checkpoint saved.")

        for p in workers:
            p.terminate()
            p.join(timeout=5)

    return model


def parse_args():
    p = argparse.ArgumentParser(description="Async self-play + training pipeline")
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--steps', type=int, default=10_000)
    p.add_argument('--sims', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    if args.sims:
        config.MCTS_SIMS = args.sims
    run_async_training(num_workers=args.workers, total_steps=args.steps)
