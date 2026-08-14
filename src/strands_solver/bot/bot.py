"""Base bot interface for Strands solving."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from strands_solver.solver.dict_solver import DictionarySolverOptions
from strands_solver.solver.open_path_solver import OpenPathSolverOptions, find_all_open_paths
from strands_solver.solver.solver import (
    board_has_spangram,
    diagonal_wall_segments,
    has_open_cells,
    open_cell_count,
)
from strands_solver.solver.spangram_solver import SpangramSolverOptions, find_all_spangram_paths
from strands_solver.util.util import MIN_WORD_LEN, board_to_text, coords_to_word

if TYPE_CHECKING:
    from strands_solver.solver.dict_solver import Trie
    from strands_solver.util.util import BoardCoord


class Bot(ABC):
    """Abstract interface for a Strands game adapter."""

    _FALLBACK_MAX_OPEN_CELLS = 19

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

    def _current_wall_segments(
        self,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        *,
        use_wall_segments: bool,
    ) -> list[tuple[BoardCoord, BoardCoord]] | None:
        """Return currently active diagonal wall segments for a solver phase."""
        if not use_wall_segments:
            return None
        return diagonal_wall_segments(successful_moves) or None

    def _try_candidate_paths(  # noqa: PLR0913
        self,
        board: list[str],
        candidate_paths: list[list[BoardCoord]],
        successful_moves: list[tuple[str, list[BoardCoord]]],
        *,
        verbose: bool,
        label: str,
        failed_paths: set[tuple[BoardCoord, ...]],
    ) -> bool:
        """Try candidate paths until one is accepted.

        Args:
            board: Current board state.
            candidate_paths: Paths to try in order.
            successful_moves: List to append accepted moves to.
            verbose: Whether to print logging.
            label: Label for verbose output.
            failed_paths: Set of paths that have failed; updated with new failures.

        Returns:
            True if a move was accepted, False if all candidates failed or were cached.

        """
        for path in candidate_paths:
            path_tuple = tuple(path)
            if path_tuple in failed_paths:
                word = coords_to_word(board, path)
                if verbose:
                    print(f"[VERBOSE] {label}: skipping cached failed path for '{word}'.")
                continue
            word = coords_to_word(board, path)
            if verbose:
                print(f"[VERBOSE] {label}: trying '{word}' with path {path}")
            if self.apply_move(path):
                if verbose:
                    print(f"[VERBOSE] {label}: move accepted.")
                successful_moves.append((word, path))
                return True
            failed_paths.add(path_tuple)

        return False

    def _run_dictionary_phase(  # noqa: PLR0913
        self,
        trie: Trie,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
        label: str,
        options: DictionarySolverOptions,
    ) -> None:
        """Run one dictionary-based solving phase until no more moves are accepted."""
        while True:
            board = self.get_board()
            if not has_open_cells(board):
                return
            if verbose:
                print(f"[VERBOSE] {label}: board:\n{board_to_text(board, ' ')}")

            wall_segments = self._current_wall_segments(
                successful_moves,
                use_wall_segments=options.use_wall_segments,
            )
            candidate_paths = trie.find_all_word_paths(board, wall_segments, options=options)
            if not candidate_paths:
                return

            match_found = self._try_candidate_paths(
                board,
                candidate_paths,
                successful_moves,
                verbose=verbose,
                label=label,
                failed_paths=failed_paths,
            )
            if match_found:
                continue
            return

    def _run_dictionary_phase2(  # noqa: PLR0913
        self,
        trie: Trie,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
        label: str,
        options: DictionarySolverOptions,
    ) -> None:
        """Run one dictionary-based solving phase until no more moves are accepted."""
        while True:
            board = self.get_board()
            if not has_open_cells(board):
                return
            if verbose:
                print(f"[VERBOSE] {label}: board:\n{board_to_text(board, ' ')}")

            wall_segments = self._current_wall_segments(
                successful_moves,
                use_wall_segments=options.use_wall_segments,
            )
            candidate_paths_limited = trie.find_all_word_paths(board, wall_segments, options=DictionarySolverOptions())
            candidate_paths_all = trie.find_all_word_paths(board, wall_segments, options=options)
            # Only consider candidates that are new under the relaxed options, to avoid retrying the same paths in multiple fallback phases.
            candidate_paths = [path for path in candidate_paths_all if path not in candidate_paths_limited]
            if not candidate_paths:
                return

            match_found = self._try_candidate_paths(
                board,
                candidate_paths,
                successful_moves,
                verbose=verbose,
                label=label,
                failed_paths=failed_paths,
            )
            if match_found:
                continue
            return

    def _run_spangram_phase(  # noqa: PLR0913
        self,
        trie: Trie,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
        label: str,
        options: SpangramSolverOptions,
    ) -> None:
        """Run the spangram-specific solving phase."""
        while True:
            board = self.get_board()
            if not has_open_cells(board) or board_has_spangram(board):
                return
            if verbose:
                print(f"[VERBOSE] {label}: board:\n{board_to_text(board, ' ')}")

            wall_segments = self._current_wall_segments(
                successful_moves,
                use_wall_segments=options.use_wall_segments,
            )
            candidate_paths = find_all_spangram_paths(trie, board, wall_segments, options=options)
            if not candidate_paths:
                return

            matched = self._try_candidate_paths(
                board,
                candidate_paths,
                successful_moves,
                verbose=verbose,
                label=label,
                failed_paths=failed_paths,
            )
            if not matched:
                return

    def _run_open_path_phase(
        self,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
        label: str,
        options: OpenPathSolverOptions,
    ) -> None:
        """Run one exhaustive open-path phase until no more moves are accepted."""
        while True:
            board = self.get_board()
            if not has_open_cells(board):
                return

            remaining_open_cells = open_cell_count(board)
            if remaining_open_cells < MIN_WORD_LEN:
                return
            if remaining_open_cells > self._FALLBACK_MAX_OPEN_CELLS:
                if verbose:
                    print(
                        f"[VERBOSE] {label}: skipping exhaustive search; "
                        f"open_cells={remaining_open_cells} exceeds max {self._FALLBACK_MAX_OPEN_CELLS}.",
                    )
                return

            if verbose:
                print(f"[VERBOSE] {label}: board:\n{board_to_text(board, ' ')}")
            wall_segments = self._current_wall_segments(
                successful_moves,
                use_wall_segments=options.use_wall_segments,
            )
            candidate_paths = find_all_open_paths(board, wall_segments, options=options)
            if not candidate_paths:
                return

            matched = self._try_candidate_paths(
                board,
                candidate_paths,
                successful_moves,
                verbose=verbose,
                label=label,
                failed_paths=failed_paths,
            )
            if not matched:
                return

    def solve_with_default_dictionary(
        self,
        trie: Trie,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
    ) -> None:
        """Try standard dictionary solving with all optimizations enabled."""
        # Phase 1: use the normal word dictionary with all path protections enabled.
        self._run_dictionary_phase(
            trie,
            successful_moves,
            failed_paths,
            verbose=verbose,
            label="default-solver",
            options=DictionarySolverOptions(dedupe_words=True, prevent_self_crossing=False),
        )

    def solve_with_spangram(
        self,
        trie: Trie,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
    ) -> None:
        """Try finding a missing spangram using segmented dictionary strings."""
        # Phase 2: search only for board-spanning paths that can be segmented
        # into dictionary words, including short words excluded from normal play.
        self._run_spangram_phase(
            trie,
            successful_moves,
            failed_paths,
            verbose=verbose,
            label="spangram-solver",
            options=SpangramSolverOptions(),
        )

    def solve_with_fallback_mode_1(
        self,
        trie: Trie,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
    ) -> None:
        """Try dictionary solving again with all protections disabled."""
        # Phase 3: keep dictionary pruning, but allow repeated words, crossings,
        # ignored wall segments, and small-island outcomes.
        self._run_dictionary_phase2(
            trie,
            successful_moves,
            failed_paths,
            verbose=verbose,
            label="fallback-mode-1",
            options=DictionarySolverOptions(
                dedupe_words=False,
                prevent_self_crossing=False,
                use_wall_segments=False,
                reject_small_islands=False,
            ),
        )

    def solve_with_fallback_mode_2(
        self,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
    ) -> None:
        """Try exhaustive open-path solving with normal geometric protections."""
        # Phase 4: enumerate all open-cell paths while still respecting walls
        # and self-crossing checks.
        self._run_open_path_phase(
            successful_moves,
            failed_paths,
            verbose=verbose,
            label="fallback-mode-2",
            options=OpenPathSolverOptions(),
        )

    def solve_with_fallback_mode_3(
        self,
        successful_moves: list[tuple[str, list[BoardCoord]]],
        failed_paths: set[tuple[BoardCoord, ...]],
        *,
        verbose: bool,
    ) -> None:
        """Try exhaustive open-path solving with relaxed geometry protections."""
        # Phase 5: exhaustive path search again, but ignore self-crossing and
        # historical wall segments.
        self._run_open_path_phase(
            successful_moves,
            failed_paths,
            verbose=verbose,
            label="fallback-mode-3",
            options=OpenPathSolverOptions(
                prevent_self_crossing=False,
                use_wall_segments=False,
            ),
        )

    def run(
        self,
        trie: Trie,
        spangram_trie: Trie | None = None,
        *,
        verbose: bool = False,
    ) -> list[tuple[str, list[BoardCoord]]]:
        """Solve the current board by running the configured solver phases.

        Args:
            trie: Trie containing standard allowed words.
            spangram_trie: Optional trie used for spangram solving.
            verbose: Whether to print verbose logging information.

        Returns:
            Matched moves as `(word, coords)` tuples in execution order.

        """
        successful_moves: list[tuple[str, list[BoardCoord]]] = []
        failed_paths: set[tuple[BoardCoord, ...]] = set()

        self.solve_with_default_dictionary(trie, successful_moves, failed_paths, verbose=verbose)

        board = self.get_board()
        if has_open_cells(board) and not board_has_spangram(board):
            self.solve_with_spangram(spangram_trie or trie, successful_moves, failed_paths, verbose=verbose)

        if has_open_cells(self.get_board()):
            self.solve_with_fallback_mode_1(trie, successful_moves, failed_paths, verbose=verbose)

        if has_open_cells(self.get_board()):
            self.solve_with_fallback_mode_2(successful_moves, failed_paths, verbose=verbose)

        if has_open_cells(self.get_board()):
            self.solve_with_fallback_mode_3(successful_moves, failed_paths, verbose=verbose)

        return successful_moves
