import chess
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from mcts import run_mcts, select_move_from_pi
from move_indexing import legal_move_mask
from encoding import encode_board


# Material values for bot
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def random_bot_move(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))


def material_bot_move(board: chess.Board) -> chess.Move:
    """Capture the highest-value piece; if no captures, move randomly."""
    best_value = -1
    best_moves = []

    for move in board.legal_moves:
        captured = board.piece_at(move.to_square)
        val = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
        if val > best_value:
            best_value = val
            best_moves = [move]
        elif val == best_value:
            best_moves.append(move)

    return random.choice(best_moves)


def score_result(result: str, color: chess.Color) -> tuple:
    """
    Returns (wins, draws, losses) from color's perspective.
    result is a string: '1-0', '0-1', '1/2-1/2', or '*'.
    color is chess.WHITE or chess.BLACK.
    """
    if result == '1/2-1/2':
        return (0, 1, 0)
    if result == '*':
        # Game hit move cap without conclusion — score as draw for arena purposes
        return (0, 1, 0)
    if color == chess.WHITE:
        if result == '1-0':
            return (1, 0, 0)
        else:
            return (0, 0, 1)
    else:
        if result == '0-1':
            return (1, 0, 0)
        else:
            return (0, 0, 1)


def choose_move_mcts(board: chess.Board, model, num_simulations: int = None, move_number: int = 0) -> chess.Move:
    """Use MCTS to choose a move (no noise during arena evaluation)."""
    pi = run_mcts(board, model, num_simulations=num_simulations, add_noise=False)
    if not pi:
        return random_bot_move(board)
    return select_move_from_pi(pi, move_number, temperature=config.TEMP_LATE)


def play_one_game(white_fn, black_fn, max_moves: int = None) -> str:
    """
    Play one game between white_fn and black_fn (callables: board -> move).
    Returns result string: '1-0', '0-1', '1/2-1/2', or '*'.
    """
    if max_moves is None:
        max_moves = config.MAX_GAME_MOVES

    board = chess.Board()
    move_number = 0

    while not board.is_game_over(claim_draw=True) and move_number < max_moves:
        if board.turn == chess.WHITE:
            move = white_fn(board)
        else:
            move = black_fn(board)

        # Safety: ensure legal move
        if move not in board.legal_moves:
            move = random_bot_move(board)

        board.push(move)
        move_number += 1

    return board.result(claim_draw=True)


def play_match(model_a, model_b, num_games: int = None, num_simulations: int = None) -> dict:
    """
    Play num_games between model_a and model_b, alternating colors.
    model_a plays White for first half, Black for second half.

    Returns dict: {
        'wins': int,    # model_a wins
        'draws': int,
        'losses': int,  # model_a losses
        'score': float, # (wins + 0.5*draws) / num_games
    }
    """
    if num_games is None:
        num_games = config.ARENA_GAMES
    if num_simulations is None:
        num_simulations = config.MCTS_SIMS

    wins = draws = losses = 0
    half = num_games // 2

    def move_a(board, move_number=0):
        return choose_move_mcts(board, model_a, num_simulations, move_number)

    def move_b(board, move_number=0):
        return choose_move_mcts(board, model_b, num_simulations, move_number)

    # model_a as White
    for _ in tqdm(range(half), desc="Arena (A=White)", leave=False):
        move_num = [0]
        def white_fn(board, _mn=move_num):
            m = move_a(board, _mn[0]); _mn[0] += 1; return m
        def black_fn(board, _mn=[0]):
            m = move_b(board, _mn[0]); _mn[0] += 1; return m
        result = play_one_game(white_fn, black_fn)
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1

    # model_a as Black
    for _ in tqdm(range(num_games - half), desc="Arena (A=Black)", leave=False):
        move_num = [0]
        def white_fn2(board, _mn=move_num):
            m = move_b(board, _mn[0]); _mn[0] += 1; return m
        def black_fn2(board, _mn=[0]):
            m = move_a(board, _mn[0]); _mn[0] += 1; return m
        result = play_one_game(white_fn2, black_fn2)
        if result == '0-1':
            wins += 1
        elif result == '1-0':
            losses += 1
        else:
            draws += 1

    score = (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'score': score,
        'games': num_games,
    }


def vs_random(model, num_games: int = 10, num_simulations: int = None) -> dict:
    """Play model vs random bot. Returns same stats dict."""
    if num_simulations is None:
        num_simulations = config.MCTS_SIMS

    wins = draws = losses = 0
    half = num_games // 2
    move_num = [0]

    def model_fn(board):
        m = choose_move_mcts(board, model, num_simulations, move_num[0])
        move_num[0] += 1
        return m

    # model as White
    for _ in range(half):
        move_num[0] = 0
        result = play_one_game(model_fn, random_bot_move)
        if result == '1-0': wins += 1
        elif result == '0-1': losses += 1
        else: draws += 1

    # model as Black
    for _ in range(num_games - half):
        move_num[0] = 0
        result = play_one_game(random_bot_move, model_fn)
        if result == '0-1': wins += 1
        elif result == '1-0': losses += 1
        else: draws += 1

    score = (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0
    return {'wins': wins, 'draws': draws, 'losses': losses, 'score': score, 'games': num_games}
