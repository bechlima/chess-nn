import chess
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Material values in centipawns
MATERIAL = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# Piece-square tables (from White's perspective, a1=index 0, h8=index 63)
# Index: square a1=0 ... h8=63, but PST is typically given rank 8 to rank 1 top-to-bottom
# We store them rank8..rank1 left-to-right, then flip for actual square lookup.

_PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

_KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

_BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

_ROOK_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

_QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

_KING_MG_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

# Map piece type to PST (indexed rank8-to-rank1, file a-to-h)
_PST = {
    chess.PAWN:   _PAWN_PST,
    chess.KNIGHT: _KNIGHT_PST,
    chess.BISHOP: _BISHOP_PST,
    chess.ROOK:   _ROOK_PST,
    chess.QUEEN:  _QUEEN_PST,
    chess.KING:   _KING_MG_PST,
}


def _pst_score(piece_type: int, square: int, color: chess.Color) -> int:
    """PST value for a piece at a square, from that color's perspective."""
    table = _PST[piece_type]
    if color == chess.WHITE:
        # White: rank 1 = bottom. PST stored rank8-top = index 0.
        # square index: a1=0, h1=7, a8=56, h8=63
        rank = chess.square_rank(square)   # 0=rank1, 7=rank8
        file = chess.square_file(square)   # 0=a, 7=h
        pst_idx = (7 - rank) * 8 + file
    else:
        # Black: mirror vertically
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        pst_idx = rank * 8 + file
    return table[pst_idx]


def evaluate(board: chess.Board) -> int:
    """
    Static evaluation from White's perspective.
    Returns centipawns: positive = good for White, negative = good for Black.
    Does NOT check for terminal states (caller's responsibility).
    """
    if board.is_checkmate():
        # Side to move is in checkmate — they lose
        return -30000 if board.turn == chess.WHITE else 30000

    score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        mat = MATERIAL[piece.piece_type]
        pst = _pst_score(piece.piece_type, square, piece.color)
        if piece.color == chess.WHITE:
            score += mat + pst
        else:
            score -= mat + pst

    # Mobility bonus (rough)
    # Count legal moves for each side (expensive but simple)
    if board.turn == chess.WHITE:
        white_mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        black_mobility = board.legal_moves.count() if not board.is_check() else 0
        board.pop()
    else:
        black_mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        white_mobility = board.legal_moves.count() if not board.is_check() else 0
        board.pop()

    score += (white_mobility - black_mobility) * 5  # 5 cp per extra legal move

    # Bishop pair bonus
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30

    return score


def evaluate_side_to_move(board: chess.Board) -> int:
    """Evaluation from side-to-move perspective (positive = good for player to move)."""
    raw = evaluate(board)
    return raw if board.turn == chess.WHITE else -raw
