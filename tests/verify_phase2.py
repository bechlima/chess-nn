"""Phase 2: Board encoding — 18×8×8 tensor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import chess, torch
from encoding import encode_board

board = chess.Board()
x = encode_board(board)
assert x.shape == torch.Size([18, 8, 8]), f"Wrong shape: {x.shape}"
assert x.dtype == torch.float32
assert x[0].sum().item() == 8, "Expected 8 white pawns"
assert x[6].sum().item() == 8, "Expected 8 black pawns"
assert x[12].sum().item() == 64, "Side-to-move plane should be all 1s for White"
print("Phase 2 PASS: encode_board returns shape", x.shape)
