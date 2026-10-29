# chess-nn

An AlphaZero-inspired chess engine in Python and PyTorch. Implements a residual policy/value network, Monte Carlo Tree Search, parallel self-play, replay-buffer training, checkpointing and head-to-head model evaluation. Includes a separate classical alpha-beta search engine with quiescence search, a transposition table and a UCI interface.

## Structure
- `src/model.py`: residual network with policy/value heads and optional win/draw/loss outputs.
- `src/mcts.py`: neural-guided Monte Carlo Tree Search.
- `src/self_play*.py`, `src/async_pipeline.py`: self-play generation and training pipeline.
- `src/train.py`, `src/arena.py`: training, checkpoints and model comparison.
- `src/search/`, `src/uci.py`: classical search and chess-GUI integration.
- `tests/`: existing tests, not rerun during portfolio preparation.

## Setup
Install `requirements.txt` in a Python environment. Inspect `src/config.py` before starting `python main.py`: training is compute-intensive. `python src/uci.py` starts the classical UCI entry point.

## Status
Research prototype. No measured Elo or successful end-to-end self-improvement result is asserted. Trained weights and generated training data are not included. This is a source snapshot of the local project; development history remains in the original local repository.
