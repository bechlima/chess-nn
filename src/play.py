import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import random
import torch
import torch.nn.functional as F
from encoding import encode_board
from move_indexing import legal_move_mask, move_to_index, index_to_move
import config


def play_random_game(max_moves=300):
    board = chess.Board()

    while not board.is_game_over(claim_draw=True) and board.fullmove_number < max_moves:
        move = random.choice(list(board.legal_moves))
        board.push(move)

    print(board)
    print("Result:", board.result(claim_draw=True))
    return board.result(claim_draw=True)


def choose_move_with_policy(board, model):
    """Returns a legal chess.Move chosen by the untrained policy network."""
    model.eval()
    with torch.no_grad():
        x = encode_board(board).unsqueeze(0).to(config.DEVICE)
        logits, _ = model(x)
        logits = logits.squeeze(0)  # (4096,)

    mask = legal_move_mask(board).to(config.DEVICE)
    logits[mask == 0] = -1e9
    probs = F.softmax(logits, dim=0)

    # Sample from distribution
    idx = torch.multinomial(probs, 1).item()
    move = index_to_move(idx, board)

    if move not in board.legal_moves:
        # Fallback: pick highest-prob legal move
        for candidate_idx in probs.argsort(descending=True).tolist():
            candidate = index_to_move(candidate_idx, board)
            if candidate in board.legal_moves:
                move = candidate
                break

    return move


def play_policy_game(model, max_moves=300):
    """Play a full game using the neural network policy. Returns result string."""
    board = chess.Board()
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        move = choose_move_with_policy(board, model)
        board.push(move)
        move_count += 1

    result = board.result(claim_draw=True)
    print(f"Policy game finished after {move_count} moves: {result}")
    return result


if __name__ == "__main__":
    from model import ChessNet

    print("=== Random game ===")
    play_random_game()

    print("\n=== Policy game (untrained) ===")
    model = ChessNet().to(config.DEVICE)
    play_policy_game(model)
