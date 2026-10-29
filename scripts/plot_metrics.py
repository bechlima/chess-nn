#!/usr/bin/env python3
"""Training metrics visualization for mini AlphaZero chess engine."""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_metrics(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def print_summary(metrics: list[dict]) -> None:
    header = f"{'Iter':>4} | {'Loss':>8} | {'Policy':>8} | {'Value':>8} | {'Arena':>6} | Accepted"
    sep    = f"{'':->4}-+-{'':->8}-+-{'':->8}-+-{'':->8}-+-{'':->6}-+-{'':->8}"
    print(header)
    print(sep)
    for m in metrics:
        accepted = "✓" if m.get("model_accepted") else "✗"
        print(
            f"{m['iteration']:>4} | "
            f"{m['train_loss']:>8.4f} | "
            f"{m['policy_loss']:>8.4f} | "
            f"{m['value_loss']:>8.4f} | "
            f"{m['arena_score']:>6.3f} | "
            f"{accepted}"
        )


def plot_metrics(metrics: list[dict], output_path: str) -> None:
    plt.style.use("dark_background")

    iters        = [m["iteration"]       for m in metrics]
    train_loss   = [m["train_loss"]      for m in metrics]
    policy_loss  = [m["policy_loss"]     for m in metrics]
    value_loss   = [m["value_loss"]      for m in metrics]
    arena_score  = [m["arena_score"]     for m in metrics]
    buffer_size  = [m["buffer_size"]     for m in metrics]
    wins         = [m["arena_wins"]      for m in metrics]
    draws        = [m["arena_draws"]     for m in metrics]
    losses       = [m["arena_losses"]    for m in metrics]
    accepted     = [m.get("model_accepted", True) for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("AlphaZero Chess — Training Dashboard", fontsize=15, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")

    COLORS = {
        "total":   "#58a6ff",
        "policy":  "#f78166",
        "value":   "#3fb950",
        "arena":   "#d2a8ff",
        "buffer":  "#79c0ff",
        "wins":    "#3fb950",
        "draws":   "#e3b341",
        "losses":  "#f78166",
        "reject":  "#ff4444",
        "thresh":  "#ffaa00",
    }

    # ── Top-left: Loss curves ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(iters, train_loss,  color=COLORS["total"],  linewidth=2, marker="o", ms=4, label="Total loss")
    ax.plot(iters, policy_loss, color=COLORS["policy"], linewidth=2, marker="s", ms=4, label="Policy loss")
    ax.plot(iters, value_loss,  color=COLORS["value"],  linewidth=2, marker="^", ms=4, label="Value loss")

    # Mark rejected iterations with a vertical red line
    for i, (it, acc) in enumerate(zip(iters, accepted)):
        if not acc:
            ax.axvline(x=it, color=COLORS["reject"], linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_title("Loss Curves", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.2)

    # ── Top-right: Arena score ────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(iters, arena_score, color=COLORS["arena"], linewidth=2, marker="o", ms=4)
    ax.axhline(y=0.55, color=COLORS["thresh"], linewidth=1.5, linestyle="--", label="Acceptance threshold (0.55)")
    ax.set_ylim(0, 1)
    ax.set_title("Arena Score", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (win + 0.5·draw)")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.2)

    # ── Bottom-left: Buffer size ──────────────────────────────────────────────
    ax = axes[1, 0]
    ax.fill_between(iters, buffer_size, color=COLORS["buffer"], alpha=0.3)
    ax.plot(iters, buffer_size, color=COLORS["buffer"], linewidth=2, marker="o", ms=4)
    ax.set_title("Replay Buffer Size", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Examples")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.2)

    # ── Bottom-right: Arena W/D/L stacked bar ─────────────────────────────────
    ax = axes[1, 1]
    x = np.arange(len(iters))
    width = 0.6
    bar_w = ax.bar(x, wins,   width, label="Wins",   color=COLORS["wins"],   alpha=0.85)
    bar_d = ax.bar(x, draws,  width, label="Draws",  color=COLORS["draws"],  alpha=0.85, bottom=wins)
    bar_l = ax.bar(x, losses, width, label="Losses", color=COLORS["losses"], alpha=0.85,
                   bottom=[w + d for w, d in zip(wins, draws)])

    ax.set_title("Arena W / D / L", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Games")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in iters], fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nPlot saved to: {output_path}")


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Plot AlphaZero training metrics.")
    parser.add_argument(
        "--metrics",
        default=os.path.join(project_root, "data", "metrics.json"),
        help="Path to metrics.json (default: data/metrics.json)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "data", "metrics.png"),
        help="Output image path (default: data/metrics.png)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        print(f"Error: metrics file not found: {args.metrics}", file=sys.stderr)
        sys.exit(1)

    metrics = load_metrics(args.metrics)
    if not metrics:
        print("Error: metrics.json is empty.", file=sys.stderr)
        sys.exit(1)

    print_summary(metrics)
    plot_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
