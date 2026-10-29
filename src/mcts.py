import math
import chess
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from encoding import encode_board
from move_indexing import move_to_index, index_to_move, legal_move_mask


class MCTSNode:
    __slots__ = ('board_fen', 'children', 'N', 'W', 'Q', 'P', 'is_terminal', 'terminal_value', '_board')

    def __init__(self, board: chess.Board):
        self._board = board.copy()
        self.board_fen = board.fen()
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = 0.0
        self.is_terminal = board.is_game_over(claim_draw=True)
        self.terminal_value = _terminal_value(board) if self.is_terminal else None

    def is_expanded(self):
        return len(self.children) > 0


def _terminal_value(board: chess.Board) -> float:
    result = board.result(claim_draw=True)
    if result == '1-0':
        return 1.0 if board.turn == chess.WHITE else -1.0
    elif result == '0-1':
        return 1.0 if board.turn == chess.BLACK else -1.0
    return 0.0


def _batch_evaluate(boards: list, model: torch.nn.Module, device: torch.device):
    """Evaluate a batch of boards in one forward pass. Returns list of (probs, value)."""
    model.eval()
    with torch.no_grad():
        tensors = torch.stack([encode_board(b) for b in boards]).to(device)
        logits_batch, values_batch = model(tensors)

    results = []
    for i, board in enumerate(boards):
        logits = logits_batch[i].to('cpu')
        value = values_batch[i].item()
        mask = legal_move_mask(board)
        logits[mask == 0] = -1e9
        probs = F.softmax(logits, dim=0)
        results.append((probs, value))
    return results


def _expand(node: MCTSNode, board: chess.Board, probs: torch.Tensor, value: float, add_noise: bool = False):
    """Expand node using pre-computed probs and value."""
    legal_moves_list = list(board.legal_moves)

    if add_noise and legal_moves_list:
        # Sample Dirichlet over legal moves only — not the full action space
        noise = np.random.dirichlet([config.DIRICHLET_ALPHA] * len(legal_moves_list))
        probs = probs.clone()
        for i, move in enumerate(legal_moves_list):
            idx = move_to_index(move, board.turn)
            probs[idx] = (1 - config.DIRICHLET_EPS) * probs[idx] + config.DIRICHLET_EPS * noise[i]

    for move in legal_moves_list:
        idx = move_to_index(move, board.turn)
        child_board = board.copy()
        child_board.push(move)
        child = MCTSNode(child_board)
        child.P = probs[idx].item()
        node.children[move.uci()] = child

    return value


def _select_child(node: MCTSNode) -> tuple:
    best_score = -float('inf')
    best_uci = None
    best_child = None
    sqrt_n = math.sqrt(max(node.N, 1))
    for uci, child in node.children.items():
        u = config.CPUCT * child.P * sqrt_n / (1 + child.N)
        score = child.Q + u
        if score > best_score:
            best_score = score
            best_uci = uci
            best_child = child
    return best_uci, best_child


def _backup(path: list, value: float):
    for node in reversed(path):
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value


def run_mcts(
    board: chess.Board,
    model: torch.nn.Module,
    num_simulations: int = None,
    add_noise: bool = True,
    batch_size: int = 8,
    device: torch.device = None,
) -> dict:
    """
    Run MCTS with batched leaf evaluation.

    batch_size: number of leaves to collect before one NN forward pass.
                Higher = more GPU efficiency, less search quality.
                8-16 is a good default. Use 1 for original per-leaf behavior.
    """
    if num_simulations is None:
        num_simulations = config.MCTS_SIMS
    if device is None:
        device = config.INFERENCE_DEVICE

    root = MCTSNode(board)

    # Expand root immediately with a single evaluation
    if not root.is_terminal:
        results = _batch_evaluate([board], model, device)
        probs, value = results[0]
        _expand(root, board, probs, value, add_noise=add_noise)

    sims_done = 0
    while sims_done < num_simulations:
        leaves = []

        for _ in range(min(batch_size, num_simulations - sims_done)):
            node = root
            current_board = board.copy()
            path = [node]

            while node.is_expanded() and not node.is_terminal:
                uci, node = _select_child(node)
                current_board.push(chess.Move.from_uci(uci))
                path.append(node)

            # Apply virtual loss so later paths in this batch avoid the same nodes
            for n in path:
                n.N += 1
                n.W -= 1.0
                n.Q = n.W / n.N

            if node.is_terminal:
                # Undo virtual loss, then do real backup
                for n in path:
                    n.N -= 1
                    n.W += 1.0
                    n.Q = n.W / n.N if n.N > 0 else 0.0
                _backup(path, node.terminal_value)
                sims_done += 1
            else:
                leaves.append((node, current_board.copy(), path))

        if not leaves:
            continue

        # Batch-evaluate all collected leaves in one forward pass
        boards_to_eval = [item[1] for item in leaves]
        results = _batch_evaluate(boards_to_eval, model, device)

        for (node, leaf_board, path), (probs, value) in zip(leaves, results):
            # Undo virtual loss before real backup
            for n in path:
                n.N -= 1
                n.W += 1.0
                n.Q = n.W / n.N if n.N > 0 else 0.0
            _expand(node, leaf_board, probs, value, add_noise=False)
            _backup(path, value)
            sims_done += 1

    # Build visit-count distribution
    if not root.children:
        return {}

    total = sum(c.N for c in root.children.values())
    if total == 0:
        n = len(root.children)
        return {uci: 1.0 / n for uci in root.children}

    return {uci: child.N / total for uci, child in root.children.items()}


def select_move_from_pi(pi: dict, move_number: int, temperature: float = None) -> chess.Move:
    """
    Select a move from the MCTS probability distribution.
    Uses temperature scheduling: TEMP_EARLY for first TEMP_MOVES, TEMP_LATE after.
    """
    if not pi:
        raise ValueError("Empty policy distribution")

    if temperature is None:
        temperature = config.TEMP_EARLY if move_number < config.TEMP_MOVES else config.TEMP_LATE

    moves = list(pi.keys())
    probs = np.array([pi[m] for m in moves], dtype=np.float64)

    if temperature < 0.01:
        return chess.Move.from_uci(moves[int(np.argmax(probs))])
    else:
        probs = probs ** (1.0 / temperature)
        probs /= probs.sum()
        return chess.Move.from_uci(moves[np.random.choice(len(moves), p=probs)])
