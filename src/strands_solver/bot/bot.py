"""Base bot interface for Strands solving."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from strands_solver.util.util import board_to_text, coords_to_word

if TYPE_CHECKING:
    from strands_solver.solver.solver import Trie
    from strands_solver.util.util import BoardCoord


class Bot(ABC):
    """Abstract interface for a Strands game adapter."""

    @abstractmethod
    def get_board(self) -> list[str]:
        """Return the current board state.

        Returns:
            Current board rows.

        """

    @abstractmethod
    def apply_move(self, move: list[BoardCoord]) -> bool:
        """Apply a move in the backing game.

        Args:
            move: Coordinate path representing a candidate word.

        Returns:
            True when the move is accepted as a valid match, otherwise False.

        """

    def run(self, trie: Trie, *, verbose: bool = False) -> list[tuple[str, list[BoardCoord]]]:  # noqa: C901
        """Solve the current board by repeatedly trying trie paths.

        Args:
            trie: Trie containing allowed words.
            verbose: Whether to print verbose logging information.

        Returns:
            Matched moves as `(word, coords)` tuples in execution order.

        """
        successful_moves: list[tuple[str, list[BoardCoord]]] = []
        board = self.get_board()
        match_found = True
        failed_words: set[str] = set()

        while match_found:
            match_found = False
            if verbose:
                board_for_printing = board_to_text(board, " ")
                print(f"[VERBOSE] Finding paths for board:\n{board_for_printing}")
            candidate_paths = trie.find_all_word_paths(board)

            skipped_any = False
            for path in candidate_paths:
                word = coords_to_word(board, path)
                if word in failed_words:
                    if verbose:
                        print(f"[VERBOSE] Skipping cached failed word '{word}'.")
                    skipped_any = True
                    continue
                if verbose:
                    print(f"[VERBOSE] Trying candidate word '{word}' with path {path}")
                match_found = self.apply_move(path)
                if match_found:
                    if verbose:
                        print("[VERBOSE] Move accepted.")
                    successful_moves.append((word, path))
                    board = self.get_board()
                    break
                failed_words.add(word)

            # If no match was found but some candidates were skipped due to the
            # cache, clear the cache and retry — a previously-failed word may now
            # be valid on the updated board.
            if not match_found and skipped_any:
                if verbose:
                    print("[VERBOSE] No match found with cache active; clearing failed-word cache and retrying.")
                failed_words.clear()
                match_found = True

        return successful_moves
