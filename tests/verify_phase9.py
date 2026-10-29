"""Phase 9: Arena runs and returns valid stats."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import config
from model import ChessNet
from arena import play_match, play_one_game, random_bot_move, material_bot_move

model_a = ChessNet().to(config.DEVICE)
model_b = ChessNet().to(config.DEVICE)

result = play_one_game(random_bot_move, material_bot_move)
assert result in ['1-0', '0-1', '1/2-1/2', '*']

stats = play_match(model_a, model_b, num_games=2, num_simulations=3)
assert stats['wins'] + stats['draws'] + stats['losses'] == 2
assert 0.0 <= stats['score'] <= 1.0
print(f"Phase 9 PASS: {stats['wins']}W {stats['draws']}D {stats['losses']}L score={stats['score']:.2f}")
