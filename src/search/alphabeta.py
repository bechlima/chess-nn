"""
Alpha-beta search engine with quiescence search and iterative deepening.
"""
import chess
import math
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.handcrafted import evaluate_side_to_move, MATERIAL
from search.transposition import TranspositionTable, TTFlag
from search.move_ordering import MoveOrderer
from search.time_manager import TimeManager

INF = 100_000
CHECKMATE_SCORE = 30_000
DRAW_SCORE = 0

class SearchResult:
    def __init__(self):
        self.best_move: Optional[chess.Move] = None
        self.score: int = 0
        self.depth: int = 0
        self.nodes: int = 0

class AlphaBetaEngine:
    def __init__(self, tt_size_mb: int = 64):
        self.tt = TranspositionTable(tt_size_mb)
        self.orderer = MoveOrderer()
        self.timer = TimeManager()
        self.nodes = 0
        self._stop = False

    def _quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self._stop or self.timer.should_stop():
            self._stop = True
            return 0

        stand_pat = evaluate_side_to_move(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Only search captures (and promotions)
        for move in board.generate_pseudo_legal_moves(to_mask=board.occupied_co[not board.turn]):
            if not board.is_legal(move):
                continue
            # Delta pruning: skip if even capturing the queen can't help
            victim = board.piece_at(move.to_square)
            if victim and stand_pat + MATERIAL.get(victim.piece_type, 0) + 200 < alpha:
                continue

            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()

            if self._stop:
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1

        if self._stop or self.timer.should_stop():
            self._stop = True
            return 0

        # Check TT
        tt_entry = self.tt.get(board)
        tt_move = None
        if tt_entry is not None and tt_entry.depth >= depth:
            tt_move = tt_entry.best_move
            if tt_entry.flag == TTFlag.EXACT:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.LOWER_BOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TTFlag.UPPER_BOUND:
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score
        elif tt_entry is not None:
            tt_move = tt_entry.best_move

        # Terminal / leaf
        if board.is_checkmate():
            return -(CHECKMATE_SCORE - ply)  # sooner mate = better
        if board.is_game_over(claim_draw=True):
            return DRAW_SCORE
        if depth == 0:
            return self._quiescence(board, alpha, beta)

        moves = list(board.legal_moves)
        moves = self.orderer.order_moves(board, moves, ply, tt_move)

        best_score = -INF
        best_move = None
        original_alpha = alpha

        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if self._stop:
                return 0

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
                if not board.piece_at(move.to_square):  # quiet move
                    self.orderer.update_history(move, depth)
            if alpha >= beta:
                if not board.piece_at(move.to_square):
                    self.orderer.store_killer(move, ply)
                self.tt.put(board, depth, best_score, TTFlag.LOWER_BOUND, best_move)
                return best_score

        # Store in TT
        if best_score <= original_alpha:
            flag = TTFlag.UPPER_BOUND
        elif best_score >= beta:
            flag = TTFlag.LOWER_BOUND
        else:
            flag = TTFlag.EXACT
        self.tt.put(board, depth, best_score, flag, best_move)

        return best_score

    def search(self, board: chess.Board, movetime_ms: float = 1000.0, max_depth: int = 20) -> SearchResult:
        """Iterative deepening search. Returns best move and score."""
        self.timer.start(movetime_ms)
        self._stop = False
        self.nodes = 0
        self.orderer.reset()

        result = SearchResult()

        # Fallback: first legal move
        legal = list(board.legal_moves)
        if not legal:
            return result
        result.best_move = legal[0]

        for depth in range(1, max_depth + 1):
            if self.timer.should_stop():
                break

            score = self._negamax(board, depth, -INF, INF, 0)

            if self._stop:
                break  # incomplete search — keep previous depth result

            result.score = score
            result.depth = depth
            result.nodes = self.nodes

            # Get best move from TT
            tt_entry = self.tt.get(board)
            if tt_entry and tt_entry.best_move and tt_entry.best_move in board.legal_moves:
                result.best_move = tt_entry.best_move

        return result
