"""
Parallel self-play using multiprocessing.

Workers run on CPU (INFERENCE_DEVICE) for MPS compatibility.
Training happens on MPS/GPU in the main process.

Usage:
    examples = parallel_self_play(model, num_workers=4, games_per_worker=5)
"""
import torch
import torch.multiprocessing as mp
import chess
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import ChessNet
from self_play import self_play_game, ReplayBuffer


def _worker_fn(
    worker_id: int,
    state_dict: dict,
    num_games: int,
    num_simulations: int,
    result_queue: mp.Queue,
):
    """Worker process: plays num_games self-play games, puts examples in queue."""
    try:
        # Each worker loads its own model on CPU
        model = ChessNet(
            board_planes=config.BOARD_PLANES,
            channels=config.CHANNELS,
            res_blocks=config.RES_BLOCKS,
            move_space=config.MOVE_SPACE,
        ).to(config.INFERENCE_DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        all_examples = []
        for game_idx in range(num_games):
            examples = self_play_game(
                model,
                num_simulations=num_simulations,
                add_noise=True,
            )
            all_examples.extend(examples)

        result_queue.put((worker_id, all_examples, None))
    except Exception as e:
        result_queue.put((worker_id, [], str(e)))


def parallel_self_play(
    model: torch.nn.Module,
    num_workers: int = None,
    games_per_worker: int = None,
    num_simulations: int = None,
    total_games: int = None,
) -> List[Tuple]:
    """
    Run self-play games in parallel across multiple CPU workers.

    Args:
        model: the current ChessNet (will be copied to each worker)
        num_workers: number of parallel processes (default: cpu_count - 1, min 1)
        games_per_worker: games each worker plays (default: auto from total_games)
        num_simulations: MCTS sims per move (default: config.MCTS_SIMS)
        total_games: total games to generate (overrides games_per_worker)

    Returns:
        List of (state, pi, z) training examples
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    if num_simulations is None:
        num_simulations = config.MCTS_SIMS

    if total_games is not None:
        games_per_worker = max(1, (total_games + num_workers - 1) // num_workers)
    elif games_per_worker is None:
        games_per_worker = max(1, config.SELF_PLAY_GAMES_PER_ITER // num_workers)

    print(f"Parallel self-play: {num_workers} workers × {games_per_worker} games = ~{num_workers * games_per_worker} games")
    print(f"MCTS sims: {num_simulations}, Inference device: {config.INFERENCE_DEVICE}")

    # Get model weights on CPU for sharing
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Use spawn to avoid MPS fork issues
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    processes = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=_worker_fn,
            args=(worker_id, state_dict, games_per_worker, num_simulations, result_queue),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # Collect results
    all_examples = []
    errors = []
    for _ in range(num_workers):
        worker_id, examples, error = result_queue.get(timeout=3600)
        if error:
            errors.append(f"Worker {worker_id}: {error}")
            print(f"  Worker {worker_id} failed: {error}")
        else:
            all_examples.extend(examples)
            print(f"  Worker {worker_id}: {len(examples)} examples")

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    if errors:
        print(f"Warning: {len(errors)} workers had errors")

    print(f"Total examples collected: {len(all_examples)}")
    return all_examples


def get_optimal_workers() -> int:
    """Suggest number of workers for this machine."""
    cores = os.cpu_count() or 4
    return max(1, cores - 1)
