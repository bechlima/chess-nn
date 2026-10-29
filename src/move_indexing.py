import chess
import torch
from move_encoding_8x8x73 import (
    move_to_index_8x8x73,
    index_to_move_8x8x73,
    legal_move_mask_8x8x73,
    MOVE_SPACE_8X8X73,
)

MOVE_SPACE = MOVE_SPACE_8X8X73  # 4672


def move_to_index(move: chess.Move, turn: chess.Color = chess.WHITE) -> int:
    """Convert move to 8x8x73 index. turn is required for correct underpromotion encoding."""
    idx = move_to_index_8x8x73(move, turn)
    if idx is None:
        raise ValueError(f"Cannot encode move {move.uci()} for turn={'WHITE' if turn else 'BLACK'}")
    return idx


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert 8x8x73 index back to chess.Move. board is required for queen promotion detection."""
    return index_to_move_8x8x73(index, board)


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Returns float32 tensor of shape (4672,), 1.0 for legal moves."""
    return legal_move_mask_8x8x73(board)
