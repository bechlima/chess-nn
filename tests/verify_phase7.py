"""Phase 7: Self-play generates valid training examples."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import config
config.MCTS_SIMS = 3  # fast test
from model import ChessNet
from self_play import self_play_game, ReplayBuffer

model = ChessNet().to(config.DEVICE)
examples = self_play_game(model, num_simulations=3)
assert len(examples) > 0

state, pi, z = examples[0]
assert state.shape == (18, 8, 8)
assert pi.shape == (config.MOVE_SPACE,)
assert abs(pi.sum().item() - 1.0) < 0.01
assert z in [-1.0, 0.0, 1.0]

buf = ReplayBuffer(max_size=1000)
buf.add(examples)
assert len(buf) == len(examples)
print(f"Phase 7 PASS: {len(examples)} examples, z={z}, buffer size={len(buf)}")
