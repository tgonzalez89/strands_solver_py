"""Exhaustive open-cell path enumeration for fallback solvers."""

from dataclasses import dataclass

from strands_solver.solver.common import PathSearchOptions, can_extend_path, get_neighbor_coords
from strands_solver.util.util import MIN_WORD_LEN, BoardCoord


@dataclass(frozen=True, slots=True)
class OpenPathSolverOptions(PathSearchOptions):
    """Options for exhaustive open-path enumeration."""

    min_path_length: int = MIN_WORD_LEN


def find_all_open_paths(
    board: list[str],
    wall_segments: list[tuple[BoardCoord, BoardCoord]] | None = None,
    *,
    options: OpenPathSolverOptions | None = None,
) -> list[list[BoardCoord]]:
    """Enumerate open-cell paths that satisfy the requested constraints."""
    found_paths: list[list[BoardCoord]] = []
    search_options = options or OpenPathSolverOptions()

    def dfs(current_path: list[BoardCoord]) -> None:
        if len(current_path) >= search_options.min_path_length:
            found_paths.append(current_path.copy())

        current_coord = current_path[-1]
        row, col = current_coord
        for next_row, next_col in get_neighbor_coords(board, row, col):
            next_coord = (next_row, next_col)
            if not can_extend_path(
                board,
                current_path,
                next_coord,
                wall_segments=wall_segments,
                prevent_self_crossing=search_options.prevent_self_crossing,
                use_wall_segments=search_options.use_wall_segments,
            ):
                continue

            current_path.append(next_coord)
            dfs(current_path)
            current_path.pop()

    for row_idx, row in enumerate(board):
        for col_idx, char in enumerate(row):
            if char in {"#", "$"}:
                continue
            dfs([(row_idx, col_idx)])

    found_paths.sort(key=lambda path: (-len(path), tuple(path)))
    return found_paths
