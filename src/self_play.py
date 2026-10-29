import chess
import torch
import numpy as np
from collections import deque
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from encoding import encode_board
from move_indexing import move_to_index
from mcts import run_mcts, select_move_from_pi


def _mirror_pi(pi_tensor, pi_dict, turn: chess.Color):
    """Build a mirrored pi tensor by reflecting each move horizontally (file a↔h)."""
    mirrored = torch.zeros_like(pi_tensor)
    for uci, prob in pi_dict.items():
        move = chess.Move.from_uci(uci)
        m_from = chess.square_mirror(move.from_square)
        m_to = chess.square_mirror(move.to_square)
        m_promo = move.promotion  # promotion piece type unchanged by horizontal mirror
        m_move = chess.Move(m_from, m_to, promotion=m_promo)
        idx = move_to_index(m_move, turn)
        mirrored[idx] = prob
    pi_sum = mirrored.sum().item()
    if pi_sum > 0:
        mirrored /= pi_sum
    return mirrored


def _to_sparse(pi_tensor):
    """Convert a dense pi tensor to (indices, values) sparse format."""
    nonzero_mask = pi_tensor > 0
    indices = nonzero_mask.nonzero(as_tuple=True)[0].to(torch.int16)
    values = pi_tensor[nonzero_mask]
    return (indices, values)


class ReplayBuffer:
    """Fixed-size replay buffer storing (state, pi, z) training examples."""

    def __init__(self, max_size: int = None):
        self.max_size = max_size or config.REPLAY_BUFFER_SIZE
        self.buffer = deque(maxlen=self.max_size)

    def add(self, examples: list):
        """Add a list of (state_tensor, pi_tensor, z_float) tuples."""
        self.buffer.extend(examples)

    def __len__(self):
        return len(self.buffer)

    def sample(self, n: int) -> list:
        """Sample n random examples."""
        indices = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def get_all(self) -> list:
        return list(self.buffer)


_MATERIAL_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _material_z(board: chess.Board, turn: chess.Color) -> float:
    """Approximate game value from material balance (used when game is truncated)."""
    score = sum(
        (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))) * v
        for pt, v in _MATERIAL_VALUES.items()
    )
    normalized = max(-1.0, min(1.0, score / 20.0))
    return normalized if turn == chess.WHITE else -normalized


def _result_to_z(result: str, turn: chess.Color, board: chess.Board = None) -> float:
    """Convert game result to z from a given player's perspective."""
    if result == '1-0':
        return 1.0 if turn == chess.WHITE else -1.0
    elif result == '0-1':
        return 1.0 if turn == chess.BLACK else -1.0
    elif result == '1/2-1/2':
        return 0.0
    else:  # '*' — game truncated by move limit, use material as proxy
        return _material_z(board, turn) if board is not None else 0.0


def self_play_game(model, num_simulations: int = None, add_noise: bool = True) -> list:
    """
    Play one self-play game using MCTS.
    Returns list of (state_tensor, pi_tensor, z_float) training examples.

    state_tensor: float32 (18, 8, 8)
    pi_tensor:    float32 (4096,) — normalized visit distribution over legal moves
    z_float:      float in {-1, 0, +1} from side-to-move perspective at that position
    """
    if num_simulations is None:
        num_simulations = config.MCTS_SIMS

    board = chess.Board()
    game_history = []   # list of (state_tensor, pi_tensor, turn)
    move_number = 0

    while not board.is_game_over(claim_draw=True) and move_number < config.MAX_GAME_MOVES:
        state = encode_board(board)  # (18, 8, 8)

        # Playout cap randomization: most moves use cheap search, some use deep search
        import random as _random
        use_full_search = _random.random() < config.FULL_SEARCH_PROB
        sims = (config.FULL_SIMS if use_full_search else config.FAST_SIMS) if num_simulations == config.MCTS_SIMS else num_simulations

        pi_dict = run_mcts(board, model, num_simulations=sims, add_noise=add_noise)

        if not pi_dict:
            break

        # Build dense pi tensor from MCTS visit counts (both full and fast search)
        pi_tensor = torch.zeros(config.MOVE_SPACE, dtype=torch.float32)
        current_turn = board.turn
        for uci, prob in pi_dict.items():
            move = chess.Move.from_uci(uci)
            pi_tensor[move_to_index(move, current_turn)] = prob
        pi_sum = pi_tensor.sum().item()
        if pi_sum > 0:
            pi_tensor /= pi_sum

        # Original position (sparse)
        game_history.append((state, _to_sparse(pi_tensor), board.turn))

        # Mirrored position (sparse) — horizontally symmetric, same z value
        mirrored_board = board.mirror()
        mirrored_state = encode_board(mirrored_board)
        mirrored_pi = _mirror_pi(pi_tensor, pi_dict, board.turn)
        game_history.append((mirrored_state, _to_sparse(mirrored_pi), board.turn))

        # Select and play move
        move = select_move_from_pi(pi_dict, move_number)
        board.push(move)
        move_number += 1

    # Determine result
    result = board.result(claim_draw=True)

    # Assign z to each position based on that position's side-to-move
    # For truncated games ('*'), use material balance as proxy value
    examples = []
    for state, pi_tensor, turn in game_history:
        z = _result_to_z(result, turn, board if result == '*' else None)
        examples.append((state, pi_tensor, z))

    return examples
