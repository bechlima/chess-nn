#!/usr/bin/env python3
"""
Benchmark forward pass and training step speed.
Run before and after torch.compile to measure speedup.

Usage:
    python src/bench.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import config
from model import ChessNet
from encoding import encode_board
from train import compile_model
import chess

def bench_forward(model, device, n=100, batch_size=32):
    model.eval()
    x = torch.randn(batch_size, config.BOARD_PLANES, 8, 8).to(device)
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(x)
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            model(x)
    elapsed = time.perf_counter() - t0
    return elapsed / n * 1000  # ms per batch

def bench_train_step(model, device, n=50, batch_size=32):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    import torch.nn.functional as F
    # Warmup
    for _ in range(3):
        x = torch.randn(batch_size, config.BOARD_PLANES, 8, 8).to(device)
        pi = torch.zeros(batch_size, config.MOVE_SPACE).to(device)
        z = torch.zeros(batch_size).to(device)
        logits, val = model(x)
        loss = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean() + F.mse_loss(val.squeeze(1), z)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    t0 = time.perf_counter()
    for _ in range(n):
        x = torch.randn(batch_size, config.BOARD_PLANES, 8, 8).to(device)
        pi = torch.zeros(batch_size, config.MOVE_SPACE).to(device)
        z = torch.zeros(batch_size).to(device)
        logits, val = model(x)
        loss = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean() + F.mse_loss(val.squeeze(1), z)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    elapsed = time.perf_counter() - t0
    return elapsed / n * 1000

def main():
    device = config.DEVICE
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    model = ChessNet().to(device)

    print("--- Without torch.compile ---")
    fwd = bench_forward(model, device)
    trn = bench_train_step(model, device)
    print(f"Forward pass (batch=32): {fwd:.2f} ms")
    print(f"Train step  (batch=32): {trn:.2f} ms")
    print()

    compiled = compile_model(ChessNet().to(device))
    print("--- With torch.compile ---")
    fwd2 = bench_forward(compiled, device)
    trn2 = bench_train_step(compiled, device)
    print(f"Forward pass (batch=32): {fwd2:.2f} ms  ({fwd/fwd2:.2f}x)")
    print(f"Train step  (batch=32): {trn2:.2f} ms  ({trn/trn2:.2f}x)")

if __name__ == "__main__":
    main()
