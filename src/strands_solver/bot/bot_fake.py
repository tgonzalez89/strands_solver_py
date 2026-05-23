"""In-memory bot implementation for fixtures and local tests."""

from pathlib import Path
from typing import TYPE_CHECKING

from strands_solver.bot.bot import Bot
from strands_solver.util.util import (
    BLOCKED_CELL,
    SPANGRAM_BLOCKED_CELL,
    load_board,
    load_moves,
    validate_board,
    validate_move_paths,
)

if TYPE_CHECKING:
    from strands_solver.util.util import BoardCoord


class BotFake(Bot):
    """In-memory game adapter used for local testing and fixtures."""

    def __init__(
        self,
        board: list[str] | Path,
        valid_moves: list[list[BoardCoord]] | Path,
        *,
        spangram_indexes: set[int] | None = None,
    ) -> None:
        """Initialize the fake bot from in-memory data or fixture paths.

        Args:
            board: Board rows or path to a board fixture.
            valid_moves: Valid move paths or path to a moves fixture.
            spangram_indexes: Move indexes classified as spangram.

        """
        loaded_board = load_board(board) if isinstance(board, Path) else board.copy()
        loaded_valid_moves = (
            load_moves(valid_moves) if isinstance(valid_moves, Path) else [move.copy() for move in valid_moves]
        )

        validate_board(loaded_board)
        validate_move_paths(loaded_valid_moves, loaded_board)

        self._board = self._normalize_board(loaded_board)
        self._valid_moves = {tuple(move) for move in loaded_valid_moves}
        selected_spangram_indexes = spangram_indexes or set()
        self._spangram_moves = {
            tuple(move) for move_idx, move in enumerate(loaded_valid_moves) if move_idx in selected_spangram_indexes
        }

    def get_board(self) -> list[str]:
        """Return the current board as immutable string rows."""
        return ["".join(row) for row in self._board]

    def apply_move(self, move: list[BoardCoord]) -> bool:
        """Apply a move by masking matched cells."""
        move_key = tuple(move)
        if move_key not in self._valid_moves:
            return False

        blocked_char = SPANGRAM_BLOCKED_CELL if move_key in self._spangram_moves else BLOCKED_CELL
        for row_idx, col_idx in move:
            self._board[row_idx][col_idx] = blocked_char

        self._valid_moves.remove(move_key)
        self._spangram_moves.discard(move_key)
        return True

    @staticmethod
    def _normalize_board(board: list[str]) -> list[list[str]]:
        """Convert string rows into mutable character rows."""
        return [list(row) for row in board]
