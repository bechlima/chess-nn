# chess-nn

An AlphaZero-style chess engine written from scratch in Python and PyTorch. A residual policy/value network guides Monte Carlo Tree Search and improves by playing itself. A separate classical alpha-beta engine plugs into any chess GUI over UCI.

## Features

**Neural engine**
- Residual CNN with policy and value heads (18 input planes, 4672-move AlphaZero move encoding over 8×8×73)
- PUCT Monte Carlo Tree Search with Dirichlet root noise and temperature scheduling
- Playout-cap randomisation: 25% of moves get a deep search and the rest a cheap one, which yields more games per hour of compute
- Parallel self-play workers feeding a replay buffer, with an asynchronous trainer on MPS/CUDA
- Arena gating: a new checkpoint replaces the current best only if it scores ≥ 55% head-to-head
- Supervised pretraining from Lichess Elite PGNs (policy and value) and Stockfish evaluations from Lichess (value)

**Classical engine**
- Alpha-beta search with quiescence search, a transposition table, move ordering and time management
- Hand-crafted evaluation with material and piece-square tables
- UCI interface, so it runs in Arena, Cute Chess, BanksiaGUI and other GUIs

## Project layout

```
main.py                    synchronous self-play → train → arena loop
src/
  model.py                 ResNet policy/value network
  mcts.py                  neural-guided MCTS
  encoding.py, move_*.py   board planes and move indexing
  self_play*.py            self-play (single-process and parallel)
  async_pipeline.py        concurrent workers + trainer
  train.py, arena.py       training steps, checkpoints, model gating
  pretrain.py              supervised pretraining from PGN / eval data
  search/                  alpha-beta, TT, move ordering, time manager
  eval/handcrafted.py      classical evaluation
  uci.py                   UCI front-end (alpha-beta default, --mode mcts)
scripts/                   Lichess data download/conversion, metric plots
tests/                     phase-by-phase verification scripts
```

## Usage

```bash
pip install -r requirements.txt

python main.py --iterations 50 --games 20 --sims 50      # self-play training loop
python src/async_pipeline.py --workers 4                 # parallel pipeline
python src/pretrain.py --pgn games.pgn --epochs 3        # supervised warm start
python src/uci.py                                        # UCI engine (alpha-beta)
python src/uci.py --mode mcts                            # UCI engine (neural MCTS)
python tests/verify_all.py                               # run verification suite
```

Hyperparameters are in `src/config.py`, and the device (MPS, CUDA or CPU) is picked automatically. Checkpoints and training data go to `data/`, which is git-ignored. No trained weights are included.

## Status
This is an active personal project. Every stage of the pipeline is implemented (pretraining, self-play, training, gating and UCI play), but no long training run has been done yet, so no Elo figure is claimed.
