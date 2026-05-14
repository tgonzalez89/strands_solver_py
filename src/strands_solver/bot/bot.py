"""Base bot interface for Strands solving."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from strands_solver.solver.solver import get_neighbor_coords, path_would_self_cross
from strands_solver.util.util import BLOCKED_CELL, MIN_WORD_LEN, board_to_text, coords_to_word

if TYPE_CHECKING:
    from strands_solver.solver.solver import Trie
    from strands_solver.util.util import BoardCoord


class Bot(ABC):
    """Abstract interface for a Strands game adapter."""

    _FALLBACK_MAX_OPEN_CELLS = 10

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
    def _has_open_cells(board: list[str]) -> bool:
        """Return whether the board still has non-blocked cells."""
        return any(char != BLOCKED_CELL for row in board for char in row)

    @staticmethod
    def _open_cell_count(board: list[str]) -> int:
        """Return the number of non-blocked cells on the board."""
        return sum(1 for row in board for char in row if char != BLOCKED_CELL)

    @staticmethod
    def _diagonal_wall_segments(
        successful_moves: list[tuple[str, list[BoardCoord]]],
    ) -> list[tuple[BoardCoord, BoardCoord]]:
        """Return diagonal segments from already accepted moves."""
        return [
            (path[idx], path[idx + 1])
            for _, path in successful_moves
            for idx in range(len(path) - 1)
            if abs(path[idx + 1][0] - path[idx][0]) == 1 and abs(path[idx + 1][1] - path[idx][1]) == 1
        ]

    @staticmethod
    def _all_open_paths(  # noqa: C901
        board: list[str],
        wall_segments: list[tuple[BoardCoord, BoardCoord]] | None = None,
    ) -> list[list[BoardCoord]]:
        """Enumerate all non-crossing open-cell paths of length >= `MIN_WORD_LEN`."""
        found_paths: list[list[BoardCoord]] = []

        def dfs(current_path: list[BoardCoord]) -> None:
            if len(current_path) >= MIN_WORD_LEN:
                found_paths.append(current_path.copy())

            current_coord = current_path[-1]
            row, col = current_coord
            for next_row, next_col in get_neighbor_coords(board, row, col):
                next_coord = (next_row, next_col)
                if next_coord in current_path:
                    continue
                if board[next_row][next_col] == BLOCKED_CELL:
                    continue
                if path_would_self_cross(current_path, next_coord):
                    continue
                if wall_segments and any(
                    _segments_intersect(current_coord, next_coord, seg_start, seg_end)
                    for seg_start, seg_end in wall_segments
                ):
                    continue

                current_path.append(next_coord)
                dfs(current_path)
                current_path.pop()

        for row_idx, row in enumerate(board):
            for col_idx, char in enumerate(row):
                if char == BLOCKED_CELL:
                    continue
                dfs([(row_idx, col_idx)])

        found_paths.sort(key=lambda path: (-len(path), tuple(path)))
        return found_paths

    def run(self, trie: Trie, *, verbose: bool = False) -> list[tuple[str, list[BoardCoord]]]:  # noqa: C901, PLR0912, PLR0915
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
            wall_segments = self._diagonal_wall_segments(successful_moves)
            candidate_paths = trie.find_all_word_paths(board, wall_segments or None)

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
                continue

            open_cell_count = self._open_cell_count(board)
            if not match_found and open_cell_count >= MIN_WORD_LEN and open_cell_count <= self._FALLBACK_MAX_OPEN_CELLS:
                if verbose:
                    print("[VERBOSE] No dictionary moves accepted; trying exhaustive open-cell fallback paths.")

                fallback_paths = self._all_open_paths(board, wall_segments or None)
                for path in fallback_paths:
                    if verbose:
                        print(f"[VERBOSE] Trying fallback path {path}")
                    match_found = self.apply_move(path)
                    if match_found:
                        if verbose:
                            print("[VERBOSE] Fallback path accepted.")
                        successful_moves.append((coords_to_word(board, path), path))
                        board = self.get_board()
                        break
            elif not match_found and self._has_open_cells(board) and verbose:
                print(
                    "[VERBOSE] Skipping exhaustive fallback: "
                    f"open_cells={open_cell_count} exceeds max {self._FALLBACK_MAX_OPEN_CELLS}.",
                )

        return successful_moves


def _orientation(a: BoardCoord, b: BoardCoord, c: BoardCoord) -> int:
    """Return orientation sign for ordered triplet `(a, b, c)`."""
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    cx, cy = c[1], c[0]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect(a1: BoardCoord, a2: BoardCoord, b1: BoardCoord, b2: BoardCoord) -> bool:
    """Return whether two closed line segments intersect."""
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    return (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4)
