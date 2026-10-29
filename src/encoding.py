import chess
import torch
import numpy as np


PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def encode_board(board: chess.Board) -> torch.Tensor:
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes
    for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for piece_idx, piece_type in enumerate(PIECE_ORDER):
            plane_idx = color_idx * 6 + piece_idx
            for square in board.pieces(piece_type, color):
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                planes[plane_idx, row, col] = 1.0

    # Side to move
    planes[12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    planes[13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Constant plane
    planes[17] = 1.0

    return torch.from_numpy(planes)
