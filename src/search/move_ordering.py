"""
Move ordering for alpha-beta search.
Order: TT move > winning captures (MVV-LVA) > killer moves > history > quiet > losing captures
"""
import chess
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.handcrafted import MATERIAL

# MVV-LVA: score = victim_value * 10 - attacker_value
def mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if victim is None or attacker is None:
        return 0
    return MATERIAL.get(victim.piece_type, 0) * 10 - MATERIAL.get(attacker.piece_type, 0)

class MoveOrderer:
    def __init__(self, max_depth: int = 64):
        self.killers: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(max_depth)]
        self.history: dict = {}  # (from_sq, to_sq) -> score

    def store_killer(self, move: chess.Move, ply: int):
        if ply < len(self.killers):
            if move != self.killers[ply][0]:
                self.killers[ply][1] = self.killers[ply][0]
                self.killers[ply][0] = move

    def update_history(self, move: chess.Move, depth: int):
        key = (move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth

    def score_move(self, board: chess.Board, move: chess.Move, ply: int,
                   tt_move: Optional[chess.Move] = None) -> int:
        if tt_move and move == tt_move:
            return 2_000_000

        victim = board.piece_at(move.to_square)
        if victim is not None:
            score = mvv_lva(board, move)
            return 1_000_000 + score  # captures before quiet moves

        # Promotions
        if move.promotion:
            return 900_000

        # Killers
        if ply < len(self.killers):
            if move == self.killers[ply][0]:
                return 800_000
            if move == self.killers[ply][1]:
                return 700_000

        # History
        key = (move.from_square, move.to_square)
        return self.history.get(key, 0)

    def order_moves(self, board: chess.Board, moves: list, ply: int,
                    tt_move: Optional[chess.Move] = None) -> list:
        return sorted(moves, key=lambda m: self.score_move(board, m, ply, tt_move), reverse=True)

    def reset(self):
        self.killers = [[None, None] for _ in range(len(self.killers))]
        self.history.clear()
