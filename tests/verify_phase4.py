"""Phase 4: Neural network — correct output shapes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import chess, torch
from encoding import encode_board
from model import ChessNet
import config

board = chess.Board()
model = ChessNet().to(config.DEVICE)
model.eval()

x = encode_board(board).unsqueeze(0).to(config.DEVICE)
with torch.no_grad():
    policy_logits, value = model(x)

assert policy_logits.shape == (1, 4096), f"Wrong policy shape: {policy_logits.shape}"
assert value.shape == (1, 1), f"Wrong value shape: {value.shape}"
assert -1.0 <= value.item() <= 1.0, f"Value out of range: {value.item()}"
print(f"Phase 4 PASS: policy={policy_logits.shape}, value={value.shape}, value={value.item():.4f}")
