import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def _reconstruct_pi(pi_sparse, move_space):
    """Reconstruct dense pi tensor from (indices, values) sparse format."""
    if isinstance(pi_sparse, torch.Tensor):
        return pi_sparse  # backwards compat with old dense format
    indices, values = pi_sparse
    pi = torch.zeros(move_space, dtype=torch.float32)
    pi[indices.long()] = values
    return pi


class ChessDataset(Dataset):
    def __init__(self, examples: list):
        """examples: list of (state_tensor, pi_sparse_or_tensor, z_float)"""
        self.states = torch.stack([e[0] for e in examples])
        self.pis = torch.stack([_reconstruct_pi(e[1], config.MOVE_SPACE) for e in examples])
        self.zs = torch.tensor([e[2] for e in examples], dtype=torch.float32)

    def __len__(self):
        return len(self.zs)

    def __getitem__(self, idx):
        return self.states[idx], self.pis[idx], self.zs[idx]


def compute_loss(pred_logits, pred_values, target_pis, target_zs):
    """
    pred_logits: (batch, 4096)
    pred_values: (batch, 1)
    target_pis:  (batch, 4096) — normalized, 0 for illegal moves
    target_zs:   (batch,) — float in [-1, 1]
    """
    # Policy loss: cross-entropy with MCTS distribution
    log_probs = F.log_softmax(pred_logits, dim=1)
    policy_loss = -(target_pis * log_probs).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(pred_values.squeeze(1), target_zs)

    return policy_loss + value_loss, policy_loss.item(), value_loss.item()


def compute_wdl_loss(pred_logits, pred_wdl_logits, target_pis, target_zs):
    """
    Loss for WDL value head.
    target_zs: float in [-1, 1] — convert to WDL class for loss.
    """
    # Policy loss (same as before)
    log_probs = F.log_softmax(pred_logits, dim=1)
    policy_loss = -(target_pis * log_probs).sum(dim=1).mean()

    # WDL loss: convert scalar z to class index
    # z=+1 → class 0 (win), z=0 → class 1 (draw), z=-1 → class 2 (loss)
    # Use soft labels: z=+1 → [1,0,0], z=0 → [0,1,0], z=-1 → [0,0,1]
    # Correct soft WDL: z=1→[1,0,0], z=0→[0,1,0], z=-1→[0,0,1], draw=1-|z|
    win  = target_zs.clamp(min=0)
    loss = (-target_zs).clamp(min=0)
    draw = 1.0 - win - loss
    wdl_targets = torch.stack([win, draw, loss], dim=1)

    log_wdl = F.log_softmax(pred_wdl_logits, dim=1)
    wdl_loss = -(wdl_targets * log_wdl).sum(dim=1).mean()

    return policy_loss + wdl_loss, policy_loss.item(), wdl_loss.item()


def train_steps(model, examples: list, num_steps: int = None, optimizer=None, scheduler=None):
    """
    Train model on examples for num_steps gradient updates.
    Returns dict of metrics.
    """
    if num_steps is None:
        num_steps = config.TRAIN_STEPS_PER_ITER

    if len(examples) < config.BATCH_SIZE:
        print(f"Warning: only {len(examples)} examples, less than batch size {config.BATCH_SIZE}")

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )

    dataset = ChessDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=min(config.BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # pin_memory only works with CUDA
        drop_last=False,
    )

    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    steps_done = 0

    loader_iter = iter(loader)
    for _ in range(num_steps):
        try:
            states, pis, zs = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            states, pis, zs = next(loader_iter)

        states = states.to(config.DEVICE)
        pis = pis.to(config.DEVICE)
        zs = zs.to(config.DEVICE)

        pred_logits, pred_values = model(states)
        if getattr(model, 'use_wdl', False):
            loss, p_loss, v_loss = compute_wdl_loss(pred_logits, pred_values, pis, zs)
        else:
            loss, p_loss, v_loss = compute_loss(pred_logits, pred_values, pis, zs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += p_loss
        total_value_loss += v_loss
        steps_done += 1

    if scheduler is not None and steps_done > 0:
        scheduler.step(total_loss / steps_done)

    metrics = {
        'loss': total_loss / max(steps_done, 1),
        'policy_loss': total_policy_loss / max(steps_done, 1),
        'value_loss': total_value_loss / max(steps_done, 1),
        'steps': steps_done,
    }
    return metrics


def save_checkpoint(model, optimizer, iteration: int, metrics: dict = None, path: str = None):
    """Save model + optimizer state."""
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(config.CHECKPOINTS_DIR, f"checkpoint_{iteration:04d}.pt")
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics or {},
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(model, path: str, optimizer=None):
    """Load model (and optionally optimizer) from checkpoint."""
    checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('iteration', 0), checkpoint.get('metrics', {})


def get_latest_checkpoint() -> str:
    """Return path to latest checkpoint, or None."""
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    checkpoints = sorted([
        f for f in os.listdir(config.CHECKPOINTS_DIR)
        if f.startswith('checkpoint_') and f.endswith('.pt')
    ])
    if not checkpoints:
        return None
    return os.path.join(config.CHECKPOINTS_DIR, checkpoints[-1])


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Try to apply torch.compile for faster training.
    MPS uses backend='aot_eager', CUDA/CPU uses default.
    Falls back gracefully if compilation fails.
    """
    device = next(model.parameters()).device
    try:
        if device.type == "mps":
            model = torch.compile(model, backend="aot_eager")
            print("torch.compile enabled (backend=aot_eager, MPS)")
        else:
            model = torch.compile(model)
            print(f"torch.compile enabled (default backend, {device})")
    except Exception as e:
        print(f"torch.compile disabled: {e}")
    return model
