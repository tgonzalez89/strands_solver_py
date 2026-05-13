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

    @staticmethod
    def _rank_candidate_paths(
        board: list[str],
        candidate_paths: list[list[BoardCoord]],
    ) -> list[tuple[str, list[BoardCoord]]]:
        """Rank candidate paths for faster solving.

        Heuristics:
        - Keep only one path per word (deduplicate repeated words).
        - Prefer longer words first.
        - For equal length, prefer words with more unique letters.
        - Use lexical tie-breaks for deterministic ordering.

        """
        best_path_by_word: dict[str, list[BoardCoord]] = {}

        for path in candidate_paths:
            word = coords_to_word(board, path)
            existing_path = best_path_by_word.get(word)
            if existing_path is None or tuple(path) < tuple(existing_path):
                best_path_by_word[word] = path

        ranked_candidates = list(best_path_by_word.items())
        ranked_candidates.sort(
            key=lambda item: (-len(item[0]), -len(set(item[0])), item[0], tuple(item[1])),
        )
        return ranked_candidates

    def run(self, trie: Trie, *, verbose: bool = False) -> list[tuple[str, list[BoardCoord]]]:
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

        while match_found:
            match_found = False
            if verbose:
                board_for_printing = board_to_text(board, " ")
                print(f"[VERBOSE] Finding paths for board:\n{board_for_printing}")
            candidate_paths = trie.find_all_word_paths(board)
            ranked_candidates = self._rank_candidate_paths(board, candidate_paths)

            for word, path in ranked_candidates:
                if verbose:
                    print(f"[VERBOSE] Trying candidate word '{word}' with path {path}")
                match_found = self.apply_move(path)
                if match_found:
                    if verbose:
                        print("[VERBOSE] Move accepted.")
                    successful_moves.append((word, path))
                    board = self.get_board()
                    break

        return successful_moves
