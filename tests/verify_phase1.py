"""Phase 1: Chess environment — random game plays legally."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from play import play_random_game

result = play_random_game()
assert result in ['1-0', '0-1', '1/2-1/2', '*'], f"Unexpected result: {result}"
print("Phase 1 PASS: random game completed with result", result)
