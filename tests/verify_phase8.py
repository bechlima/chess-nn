"""Phase 8: Training loop runs and checkpoints save/reload."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch, random
import config
config.BATCH_SIZE = 4
from model import ChessNet
from train import train_steps, save_checkpoint, load_checkpoint

model = ChessNet().to(config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

examples = []
for _ in range(20):
    state = torch.randn(18, 8, 8)
    pi = torch.zeros(config.MOVE_SPACE)
    idxs = random.sample(range(config.MOVE_SPACE), 20)
    probs = torch.softmax(torch.randn(20), dim=0)
    for i, idx in enumerate(idxs):
        pi[idx] = probs[i].item()
    z = random.choice([-1.0, 0.0, 1.0])
    examples.append((state, pi, z))

metrics = train_steps(model, examples, num_steps=5, optimizer=optimizer)
assert metrics['steps'] == 5

path = save_checkpoint(model, optimizer, 0)
model2 = ChessNet().to(config.DEVICE)
load_checkpoint(model2, path)
for p1, p2 in zip(model.parameters(), model2.parameters()):
    assert torch.allclose(p1, p2)

print(f"Phase 8 PASS: loss={metrics['loss']:.4f}, checkpoint saved/loaded")
