"""Shared board-search helpers used by solver strategies."""

import itertools
from dataclasses import dataclass

from strands_solver.util.util import BLOCKED_CELLS, MIN_WORD_LEN, SPANGRAM_BLOCKED_CELL, BoardCoord


@dataclass(frozen=True, slots=True)
class PathSearchOptions:
    """Common path-validation options for solver searches."""

    prevent_self_crossing: bool = True
    use_wall_segments: bool = True
    reject_small_islands: bool = True


def is_blocked_cell(char: str) -> bool:
    """Return whether a board character represents a solved cell."""
    return char in BLOCKED_CELLS


def has_open_cells(board: list[str]) -> bool:
    """Return whether the board still has non-blocked cells."""
    return any(not is_blocked_cell(char) for row in board for char in row)


def open_cell_count(board: list[str]) -> int:
    """Return the number of non-blocked cells on the board."""
    return sum(1 for row in board for char in row if not is_blocked_cell(char))


def board_has_spangram(board: list[str]) -> bool:
    """Return whether the board contains at least one solved spangram cell."""
    return any(char == SPANGRAM_BLOCKED_CELL for row in board for char in row)


def diagonal_wall_segments(
    successful_moves: list[tuple[str, list[BoardCoord]]],
) -> list[tuple[BoardCoord, BoardCoord]]:
    """Return diagonal segments from already accepted moves."""
    return [
        (path[idx], path[idx + 1])
        for _, path in successful_moves
        for idx in range(len(path) - 1)
        if abs(path[idx + 1][0] - path[idx][0]) == 1 and abs(path[idx + 1][1] - path[idx][1]) == 1
    ]


def get_neighbor_coords(board: list[str], row: int, col: int) -> list[BoardCoord]:
    """Return all valid neighboring coordinates around a cell."""
    neighbors: list[BoardCoord] = []

    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            if row_delta == 0 and col_delta == 0:
                continue

            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            if neighbor_row < 0 or neighbor_row >= len(board):
                continue
            if neighbor_col < 0 or neighbor_col >= len(board[neighbor_row]):
                continue

            neighbors.append((neighbor_row, neighbor_col))

    return neighbors


def leaves_small_island(
    board: list[str],
    removed_path: list[BoardCoord],
    wall_segments: list[tuple[BoardCoord, BoardCoord]] | None = None,
) -> bool:
    """Check whether removing a path creates too-small connected islands.

    Args:
        board: Current board state.
        removed_path: The candidate path being evaluated.
        wall_segments: Additional diagonal wall segments from previously accepted
            moves that restrict connectivity between cells.

    """
    removed_coords = set(removed_path)
    effective_walls = list(itertools.pairwise(removed_path))
    if wall_segments:
        effective_walls.extend(wall_segments)
    remaining_coords = {
        (row_idx, col_idx)
        for row_idx, row in enumerate(board)
        for col_idx, char in enumerate(row)
        if not is_blocked_cell(char) and (row_idx, col_idx) not in removed_coords
    }

    unseen_coords = remaining_coords.copy()
    while unseen_coords:
        start_coord = unseen_coords.pop()
        island_size = 0
        stack = [start_coord]

        while stack:
            coord = stack.pop()
            island_size += 1

            row, col = coord
            for neighbor in get_neighbor_coords(board, row, col):
                if neighbor in unseen_coords:
                    if any(
                        segments_intersect(coord, neighbor, seg_start, seg_end)
                        for seg_start, seg_end in effective_walls
                    ):
                        continue
                    unseen_coords.remove(neighbor)
                    stack.append(neighbor)

        if island_size < MIN_WORD_LEN:
            return True

    return False


def orientation(a: BoardCoord, b: BoardCoord, c: BoardCoord) -> int:
    """Return orientation sign for ordered triplet `(a, b, c)`."""
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    cx, cy = c[1], c[0]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_intersect(a1: BoardCoord, a2: BoardCoord, b1: BoardCoord, b2: BoardCoord) -> bool:
    """Return whether two closed line segments intersect."""
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)

    return (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4)


def path_would_self_cross(path: list[BoardCoord], next_coord: BoardCoord) -> bool:
    """Check whether extending `path` to `next_coord` would self-intersect."""
    if len(path) < 3:  # noqa: PLR2004
        return False

    new_start = path[-1]
    new_end = next_coord

    for idx in range(len(path) - 2):
        seg_start = path[idx]
        seg_end = path[idx + 1]
        if segments_intersect(seg_start, seg_end, new_start, new_end):
            return True

    return False


def can_extend_path(  # noqa: PLR0913
    board: list[str],
    current_path: list[BoardCoord],
    next_coord: BoardCoord,
    *,
    wall_segments: list[tuple[BoardCoord, BoardCoord]] | None = None,
    prevent_self_crossing: bool = True,
    use_wall_segments: bool = True,
) -> bool:
    """Return whether `current_path` can be extended to `next_coord`."""
    if next_coord in current_path:
        return False

    next_row, next_col = next_coord
    if is_blocked_cell(board[next_row][next_col]):
        return False

    current_coord = current_path[-1]
    if prevent_self_crossing and path_would_self_cross(current_path, next_coord):
        return False

    return not (
        use_wall_segments
        and wall_segments
        and any(
            segments_intersect(current_coord, next_coord, seg_start, seg_end) for seg_start, seg_end in wall_segments
        )
    )


def is_spangram_path(board: list[str], path: list[BoardCoord]) -> bool:
    """Return whether `path` touches opposite board edges."""
    if not path:
        return False

    max_row = len(board) - 1
    max_col = len(board[0]) - 1
    rows = {row for row, _ in path}
    cols = {col for _, col in path}
    return (0 in rows and max_row in rows) or (0 in cols and max_col in cols)
