"""
PGN loader: parse chess PGN files into (state_tensor, move_index, z) training examples.

Usage:
    from src.data.pgn_loader import load_pgn_file, stream_pgn_file

    # Load entire file into memory
    examples = load_pgn_file("data/raw/lichess_elite/games.pgn", max_games=10000)

    # Stream from file (memory-efficient)
    for state, pi, z in stream_pgn_file("data/raw/lichess_elite/games.pgn"):
        ...
"""
import chess
import chess.pgn
import io
import torch
import numpy as np
from typing import Iterator, List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoding import encode_board
from move_indexing import move_to_index, legal_move_mask
import config


def _result_to_z(result: str, turn: chess.Color) -> float:
    """Convert PGN result string to z from side-to-move perspective."""
    if result == '1-0':
        return 1.0 if turn == chess.WHITE else -1.0
    elif result == '0-1':
        return 1.0 if turn == chess.BLACK else -1.0
    elif result == '1/2-1/2':
        return 0.0
    else:
        return 0.0  # unknown


def _get_min_rating(game: chess.pgn.Game) -> int:
    """Get minimum Elo rating of both players from PGN headers."""
    try:
        white_elo = int(game.headers.get('WhiteElo', 0))
        black_elo = int(game.headers.get('BlackElo', 0))
        return min(white_elo, black_elo)
    except (ValueError, TypeError):
        return 0


def _is_valid_game(game: chess.pgn.Game, min_rating: int = 1800) -> bool:
    """Filter out low-quality games."""
    if game is None:
        return False
    result = game.headers.get('Result', '*')
    if result not in ('1-0', '0-1', '1/2-1/2'):
        return False
    if min_rating > 0 and _get_min_rating(game) < min_rating:
        return False
    # Skip very short games
    move_count = sum(1 for _ in game.mainline_moves())
    if move_count < 10:
        return False
    return True


def stream_pgn_file(
    pgn_path: str,
    max_games: Optional[int] = None,
    min_rating: int = 0,
    sample_every: int = 1,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Stream (state, pi, z) examples from a PGN file.

    pi is a one-hot sparse tensor: probability 1.0 on the played move index.
    z is the game result from the position's side-to-move perspective.

    Args:
        pgn_path: path to .pgn file
        max_games: stop after this many valid games
        min_rating: skip games below this minimum rating
        sample_every: only use every Nth position (reduces correlation)
    """
    games_processed = 0

    with open(pgn_path, 'r', errors='replace') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            if not _is_valid_game(game, min_rating):
                continue

            result = game.headers.get('Result', '*')
            board = game.board()
            move_num = 0

            for move in game.mainline_moves():
                if move_num % sample_every == 0:
                    # Build state tensor
                    state = encode_board(board)

                    # Build one-hot policy tensor
                    pi = torch.zeros(config.MOVE_SPACE, dtype=torch.float32)
                    # Handle promotion: if move has promotion, it stays as-is in move_to_index
                    idx = move_to_index(move, board.turn)
                    if 0 <= idx < config.MOVE_SPACE:
                        pi[idx] = 1.0

                    # Value target from side-to-move perspective
                    z = _result_to_z(result, board.turn)

                    yield state, pi, z

                board.push(move)
                move_num += 1

            games_processed += 1
            if max_games and games_processed >= max_games:
                break


def load_pgn_file(
    pgn_path: str,
    max_games: Optional[int] = None,
    min_rating: int = 0,
    sample_every: int = 2,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Load PGN file into a list of examples. Use stream_pgn_file for large files."""
    examples = []
    for state, pi, z in stream_pgn_file(pgn_path, max_games, min_rating, sample_every):
        examples.append((state, pi, z))
    return examples


def stream_pgn_string(pgn_text: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Stream examples from a PGN string (useful for testing)."""
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return

    result = game.headers.get('Result', '*')
    board = game.board()

    for move in game.mainline_moves():
        state = encode_board(board)
        pi = torch.zeros(config.MOVE_SPACE, dtype=torch.float32)
        idx = move_to_index(move, board.turn)
        if 0 <= idx < config.MOVE_SPACE:
            pi[idx] = 1.0
        z = _result_to_z(result, board.turn)
        yield state, pi, z
        board.push(move)
