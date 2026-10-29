"""
AlphaZero-style 8×8×73 move encoding.
4672 total actions vs the old 4096.

Usage:
    from move_encoding_8x8x73 import move_to_index_8x8x73, legal_move_mask_8x8x73, MOVE_SPACE_8X8X73
"""
import chess
import torch
import numpy as np
from typing import Optional

MOVE_SPACE_8X8X73 = 8 * 8 * 73  # = 4672

# Queen-like directions: (delta_rank, delta_file)
# rank increases going "up" (toward rank 8)
QUEEN_DIRS = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

KNIGHT_DELTAS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

# Underpromotion pieces (queen is NOT here — handled as queen-like move)
UNDER_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _sq_to_row_col(square: int):
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    row = 7 - rank
    col = file
    return row, col


def move_to_index_8x8x73(move: chess.Move, turn: chess.Color = chess.WHITE) -> Optional[int]:
    """
    Convert a chess.Move to 8×8×73 index.
    Returns None if the move cannot be encoded (should not happen for legal moves).

    turn: the side making the move (needed to orient pawn promotions)
    """
    from_sq = move.from_square
    to_sq = move.to_square

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    from_row = 7 - from_rank
    from_col = from_file

    delta_rank = to_rank - from_rank
    delta_file = to_file - from_file

    # Underpromotion (non-queen promotion)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_piece = move.promotion
        if promo_piece not in UNDER_PROMO_PIECES:
            return None
        piece_idx = UNDER_PROMO_PIECES.index(promo_piece)
        # Direction: left-capture, push, right-capture
        # From White's perspective: file decreases = left, no change = push, increases = right
        # From Black's perspective: mirrored
        if turn == chess.WHITE:
            if delta_file == -1:
                dir_idx = 0
            elif delta_file == 0:
                dir_idx = 1
            else:
                dir_idx = 2
        else:
            if delta_file == 1:
                dir_idx = 0
            elif delta_file == 0:
                dir_idx = 1
            else:
                dir_idx = 2
        plane = 64 + piece_idx * 3 + dir_idx
        return (from_row * 8 + from_col) * 73 + plane

    # Knight moves
    delta = (delta_rank, delta_file)
    if delta in KNIGHT_DELTAS:
        knight_idx = KNIGHT_DELTAS.index(delta)
        plane = 56 + knight_idx
        return (from_row * 8 + from_col) * 73 + plane

    # Queen-like moves (including queen promotions, which are normal forward moves)
    if delta_rank == 0 and delta_file == 0:
        return None  # null move

    # Determine direction
    abs_rank = abs(delta_rank)
    abs_file = abs(delta_file)

    if abs_rank == 0:
        dir_delta = (0, 1 if delta_file > 0 else -1)
        distance = abs_file
    elif abs_file == 0:
        dir_delta = (1 if delta_rank > 0 else -1, 0)
        distance = abs_rank
    elif abs_rank == abs_file:
        dir_delta = (1 if delta_rank > 0 else -1, 1 if delta_file > 0 else -1)
        distance = abs_rank
    else:
        return None  # Not a valid queen-like or knight move

    if dir_delta not in QUEEN_DIRS:
        return None

    dir_idx = QUEEN_DIRS.index(dir_delta)
    plane = dir_idx * 7 + (distance - 1)
    return (from_row * 8 + from_col) * 73 + plane


def index_to_move_8x8x73(index: int, board: chess.Board) -> chess.Move:
    """
    Convert 8×8×73 index back to chess.Move given the board context.
    Board is needed to determine promotion context and turn.
    """
    from_flat = index // 73
    plane = index % 73
    from_row = from_flat // 8
    from_col = from_flat % 8
    from_rank = 7 - from_row
    from_file = from_col
    from_sq = chess.square(from_file, from_rank)

    promotion = None

    if plane >= 64:
        # Underpromotion
        promo_idx = (plane - 64) // 3
        dir_idx = (plane - 64) % 3
        promotion = UNDER_PROMO_PIECES[promo_idx]
        if board.turn == chess.WHITE:
            delta_file = dir_idx - 1  # -1, 0, +1
            to_rank = from_rank + 1
        else:
            delta_file = -(dir_idx - 1)
            to_rank = from_rank - 1
        to_file = from_file + delta_file
        to_sq = chess.square(to_file, to_rank)
    elif plane >= 56:
        # Knight
        knight_idx = plane - 56
        dr, df = KNIGHT_DELTAS[knight_idx]
        to_rank = from_rank + dr
        to_file = from_file + df
        to_sq = chess.square(to_file, to_rank)
    else:
        # Queen-like
        dir_idx = plane // 7
        distance = plane % 7 + 1
        dr, df = QUEEN_DIRS[dir_idx]
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        to_sq = chess.square(to_file, to_rank)

        # Check if this is a queen promotion
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if (board.turn == chess.WHITE and to_rank == 7) or \
               (board.turn == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_move_mask_8x8x73(board: chess.Board) -> torch.Tensor:
    """Returns float32 tensor of shape (4672,), 1.0 for legal moves."""
    mask = torch.zeros(MOVE_SPACE_8X8X73, dtype=torch.float32)
    for move in board.legal_moves:
        idx = move_to_index_8x8x73(move, board.turn)
        if idx is not None and 0 <= idx < MOVE_SPACE_8X8X73:
            mask[idx] = 1.0
    return mask
