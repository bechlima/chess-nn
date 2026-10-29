"""Phase 3: Move indexing — legal mask has 20 moves from start."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import chess
from move_indexing import legal_move_mask, move_to_index, index_to_move

board = chess.Board()
mask = legal_move_mask(board)
count = int(mask.sum().item())
assert count == 20, f"Expected 20 legal moves, got {count}"

move = chess.Move.from_uci('e2e4')
idx = move_to_index(move)
recovered = index_to_move(idx)
assert recovered.from_square == move.from_square and recovered.to_square == move.to_square
print(f"Phase 3 PASS: {count} legal moves, roundtrip OK")
