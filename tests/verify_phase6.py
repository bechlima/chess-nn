"""Phase 6: MCTS returns valid probability distribution."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import chess
from model import ChessNet
from mcts import run_mcts, select_move_from_pi
import config

board = chess.Board()
model = ChessNet().to(config.DEVICE)
model.eval()

pi = run_mcts(board, model, num_simulations=10, add_noise=True)
assert len(pi) > 0, "Empty policy"
total = sum(pi.values())
assert abs(total - 1.0) < 0.01, f"Probs don't sum to 1: {total}"

move = select_move_from_pi(pi, move_number=5)
assert move in board.legal_moves, f"{move} is illegal!"
print(f"Phase 6 PASS: MCTS returned {len(pi)} moves, sum={total:.4f}, selected={move}")
