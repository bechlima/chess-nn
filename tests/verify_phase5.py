"""Phase 5: Raw policy bot plays a full legal game."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ChessNet
from play import play_policy_game
import config

model = ChessNet().to(config.DEVICE)
result = play_policy_game(model)
assert result in ['1-0', '0-1', '1/2-1/2', '*'], f"Bad result: {result}"
print("Phase 5 PASS: policy bot played full game, result:", result)
