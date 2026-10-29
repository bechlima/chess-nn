#!/usr/bin/env python3
"""
UCI (Universal Chess Interface) engine wrapper.

Usage:
    python src/uci.py                    # alpha-beta engine (default)
    python src/uci.py --mode mcts        # MCTS engine (experimental)
    python src/uci.py --movetime 2000    # default movetime in ms

Test manually:
    echo -e "uci\nisready\nposition startpos\ngo movetime 1000\nquit" | python src/uci.py
"""
import sys
import os
import threading
import argparse
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search.alphabeta import AlphaBetaEngine
import config

ENGINE_NAME = "ChessNN"
ENGINE_AUTHOR = "chess-nn project"


def parse_position(tokens: list) -> chess.Board:
    """Parse 'position startpos [moves ...]' or 'position fen <fen> [moves ...]'"""
    board = chess.Board()
    idx = 0

    if not tokens:
        return board

    if tokens[0] == 'startpos':
        idx = 1
    elif tokens[0] == 'fen':
        # Collect FEN parts until 'moves' or end
        fen_parts = []
        idx = 1
        while idx < len(tokens) and tokens[idx] != 'moves':
            fen_parts.append(tokens[idx])
            idx += 1
        fen = ' '.join(fen_parts)
        try:
            board = chess.Board(fen)
        except Exception:
            board = chess.Board()

    # Apply moves
    if idx < len(tokens) and tokens[idx] == 'moves':
        idx += 1
        while idx < len(tokens):
            try:
                move = chess.Move.from_uci(tokens[idx])
                if move in board.legal_moves:
                    board.push(move)
                else:
                    # Try with queen promotion
                    move_q = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                    if move_q in board.legal_moves:
                        board.push(move_q)
            except Exception:
                pass
            idx += 1

    return board


class UCIEngine:
    def __init__(self, default_movetime_ms: float = 1000.0):
        self.engine = AlphaBetaEngine(tt_size_mb=64)
        self.board = chess.Board()
        self.default_movetime_ms = default_movetime_ms
        self._search_thread: threading.Thread = None

    def handle(self, line: str):
        line = line.strip()
        if not line:
            return

        tokens = line.split()
        cmd = tokens[0]

        if cmd == 'uci':
            print(f"id name {ENGINE_NAME}")
            print(f"id author {ENGINE_AUTHOR}")
            print("option name Hash type spin default 64 min 1 max 512")
            print("option name MoveTime type spin default 1000 min 100 max 60000")
            print("uciok")
            sys.stdout.flush()

        elif cmd == 'isready':
            print("readyok")
            sys.stdout.flush()

        elif cmd == 'ucinewgame':
            self.engine.tt.clear()
            self.board = chess.Board()

        elif cmd == 'setoption':
            # setoption name Hash value 128
            if 'Hash' in tokens and 'value' in tokens:
                try:
                    idx = tokens.index('value')
                    mb = int(tokens[idx + 1])
                    self.engine.tt = type(self.engine.tt)(mb)
                except Exception:
                    pass

        elif cmd == 'position':
            self.board = parse_position(tokens[1:])

        elif cmd == 'go':
            self._handle_go(tokens[1:])

        elif cmd == 'stop':
            self.engine._stop = True

        elif cmd == 'quit' or cmd == 'exit':
            sys.exit(0)

        elif cmd == 'print' or cmd == 'd':
            print(self.board)
            sys.stdout.flush()

    def _handle_go(self, tokens: list):
        movetime_ms = self.default_movetime_ms
        max_depth = 20

        i = 0
        while i < len(tokens):
            if tokens[i] == 'movetime' and i + 1 < len(tokens):
                try:
                    movetime_ms = float(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == 'depth' and i + 1 < len(tokens):
                try:
                    max_depth = int(tokens[i + 1])
                    movetime_ms = 60_000  # large time when depth-limited
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == 'infinite':
                movetime_ms = 3_600_000  # 1 hour
                i += 1
            elif tokens[i] in ('wtime', 'btime', 'winc', 'binc', 'movestogo'):
                # Basic time control: use 1/20 of remaining time
                try:
                    color_key = 'wtime' if self.board.turn == chess.WHITE else 'btime'
                    if tokens[i] == color_key and i + 1 < len(tokens):
                        remaining = float(tokens[i + 1])
                        movetime_ms = max(100, remaining / 20)
                except (ValueError, IndexError):
                    pass
                i += 2
            else:
                i += 1

        board_copy = self.board.copy()

        def search_and_respond():
            result = self.engine.search(board_copy, movetime_ms=movetime_ms, max_depth=max_depth)
            if result.best_move:
                print(f"info depth {result.depth} score cp {result.score} nodes {result.nodes}")
                print(f"bestmove {result.best_move.uci()}")
            else:
                legal = list(board_copy.legal_moves)
                if legal:
                    print(f"bestmove {legal[0].uci()}")
                else:
                    print("bestmove 0000")
            sys.stdout.flush()

        # Run search in background thread so stop works
        self._search_thread = threading.Thread(target=search_and_respond, daemon=True)
        self._search_thread.start()
        self._search_thread.join()  # For simplicity, block until done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--movetime', type=float, default=1000.0)
    args = parser.parse_args()

    uci = UCIEngine(default_movetime_ms=args.movetime)

    for line in sys.stdin:
        uci.handle(line)


if __name__ == '__main__':
    main()
