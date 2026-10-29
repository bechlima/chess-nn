import chess
import chess.polyglot
from enum import IntEnum
from typing import Optional, NamedTuple

class TTFlag(IntEnum):
    EXACT = 0        # exact score
    LOWER_BOUND = 1  # alpha cutoff (fail-high)
    UPPER_BOUND = 2  # beta cutoff (fail-low)

class TTEntry(NamedTuple):
    key: int
    depth: int
    score: int
    flag: TTFlag
    best_move: Optional[chess.Move]

class TranspositionTable:
    """Fixed-size transposition table using dict (simple Python version)."""

    def __init__(self, size_mb: int = 64):
        # Approximate: each entry ~200 bytes; 1MB = ~5000 entries
        self.max_entries = (size_mb * 1024 * 1024) // 200
        self._table: dict[int, TTEntry] = {}

    def get(self, board: chess.Board) -> Optional[TTEntry]:
        key = chess.polyglot.zobrist_hash(board)
        entry = self._table.get(key)
        if entry is not None and entry.key == key:
            return entry
        return None

    def put(self, board: chess.Board, depth: int, score: int,
            flag: TTFlag, best_move: Optional[chess.Move]):
        key = chess.polyglot.zobrist_hash(board)
        # Simple replacement: always replace if deeper or table getting full
        existing = self._table.get(key)
        if existing is None or existing.depth <= depth or len(self._table) >= self.max_entries:
            if len(self._table) >= self.max_entries:
                # Evict ~10% of entries (simple)
                keys_to_del = list(self._table.keys())[:self.max_entries // 10]
                for k in keys_to_del:
                    del self._table[k]
            self._table[key] = TTEntry(key, depth, score, flag, best_move)

    def clear(self):
        self._table.clear()

    def __len__(self):
        return len(self._table)
