"""Game interfaces and test implementation for Strands solving."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from strands_solver.io_utils import coords_to_word, load_board, load_moves, validate_board, validate_move_paths

if TYPE_CHECKING:
    from strands_solver.solver import Coord, Trie


class StrandsGameBot(ABC):
    """Abstract interface for a Strands game adapter."""

    @abstractmethod
    def get_board(self) -> list[str]:
        """Return the current board state.

        Returns:
            Current board rows.
        """

    @abstractmethod
    def apply_move(self, move: list[Coord]) -> bool:
        """Apply a move in the backing game.

        Args:
            move: Coordinate path representing a candidate word.

        Returns:
            True when the move is accepted as a valid match, otherwise False.
        """

    def run(self, trie: Trie) -> list[tuple[str, list[Coord]]]:
        """Solve the current board by repeatedly trying trie paths.

        Args:
            trie: Trie containing allowed words.

        Returns:
            Matched moves as `(word, coords)` tuples in execution order.
        """
        successful_moves: list[tuple[str, list[Coord]]] = []
        board = self.get_board()
        match_found = True

        while match_found:
            match_found = False
            candidate_paths = trie.find_all_word_paths(board)

            for path in candidate_paths:
                match_found = self.apply_move(path)
                if match_found:
                    successful_moves.append((coords_to_word(board, path), path))
                    board = self.get_board()
                    break

        return successful_moves


class StrandsGameBotTest(StrandsGameBot):
    """In-memory game adapter used for local testing and fixtures."""

    def __init__(self, board: list[str] | Path, valid_moves: list[list[Coord]] | Path) -> None:
        """Build a test bot from direct data or file paths.

        Args:
            board: Board rows or path to a board file.
            valid_moves: Allowed move paths or path to a moves file.

        Raises:
            ValueError: If board or moves violate validation rules.
            TypeError: If loaded moves use invalid literal or coord types.
        """
        loaded_board = load_board(board) if isinstance(board, Path) else board.copy()
        loaded_valid_moves = (
            load_moves(valid_moves) if isinstance(valid_moves, Path) else [move.copy() for move in valid_moves]
        )

        validate_board(loaded_board)
        validate_move_paths(loaded_valid_moves, loaded_board)

        self._board = self._normalize_board(loaded_board)
        self._valid_moves = {tuple(move) for move in loaded_valid_moves}

    def get_board(self) -> list[str]:
        """Return the current board as immutable string rows.

        Returns:
            Board rows.
        """
        return ["".join(row) for row in self._board]

    def apply_move(self, move: list[Coord]) -> bool:
        """Apply a move by masking matched cells.

        Args:
            move: Coordinate path to apply.

        Returns:
            True if move is one of the configured valid moves; otherwise False.
        """
        move_key = tuple(move)
        if move_key not in self._valid_moves:
            return False

        for row_idx, col_idx in move:
            self._board[row_idx][col_idx] = "#"

        self._valid_moves.remove(move_key)
        return True

    @staticmethod
    def _normalize_board(board: list[str]) -> list[list[str]]:
        """Convert string rows into mutable character rows.

        Args:
            board: Board rows as strings.

        Returns:
            Mutable 2D board representation.
        """
        return [list(row) for row in board]
