"""Trie-based solver utilities for finding valid word paths on a board."""

import itertools
from dataclasses import dataclass, field
from typing import Self

from strands_solver.util.util import BLOCKED_CELL, MIN_WORD_LEN, BoardCoord, coords_to_word


def get_neighbor_coords(board: list[str], row: int, col: int) -> list[BoardCoord]:
    """Return all valid neighboring coordinates around a cell.

    Includes 8-directional adjacency and excludes the center coordinate itself.

    Args:
        board: Board rows.
        row: Source row index.
        col: Source column index.

    Returns:
        Neighbor coordinates inside board bounds.

    """
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
        board: Current board rows.
        removed_path: Coordinates treated as removed for this validation check.
        wall_segments: Pre-computed diagonal walls (blocked-cell pairs plus the
            current path's own edges) that must not be crossed during the BFS.
            Callers should build this once and pass it in rather than letting
            this function recompute it on every recursive call.

    Returns:
        True if any remaining connected component has size less than
        `MIN_WORD_LEN`; otherwise False.

    """
    removed_coords = set(removed_path)
    effective_walls = list(itertools.pairwise(removed_path))
    if wall_segments:
        effective_walls.extend(wall_segments)
    remaining_coords = {
        (row_idx, col_idx)
        for row_idx, row in enumerate(board)
        for col_idx, char in enumerate(row)
        if char != BLOCKED_CELL and (row_idx, col_idx) not in removed_coords
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
                        _segments_intersect(coord, neighbor, seg_start, seg_end)
                        for seg_start, seg_end in effective_walls
                    ):
                        continue
                    unseen_coords.remove(neighbor)
                    stack.append(neighbor)

        if island_size < MIN_WORD_LEN:
            return True

    return False


def _orientation(a: BoardCoord, b: BoardCoord, c: BoardCoord) -> int:
    """Return orientation sign for ordered triplet `(a, b, c)`.

    Positive/negative values indicate clockwise/counter-clockwise, and zero
    indicates collinearity.

    """
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


def path_would_self_cross(path: list[BoardCoord], next_coord: BoardCoord) -> bool:
    """Check whether extending `path` to `next_coord` would self-intersect."""
    if len(path) < 3:  # noqa: PLR2004
        return False

    new_start = path[-1]
    new_end = next_coord

    for idx in range(len(path) - 2):
        seg_start = path[idx]
        seg_end = path[idx + 1]
        if _segments_intersect(seg_start, seg_end, new_start, new_end):
            return True

    return False


@dataclass
class Node:
    """Trie node containing children and terminal-word metadata."""

    is_word: bool = False
    children: dict[str, Self] = field(default_factory=dict)

    def find_word_paths(
        self,
        board: list[str],
        current_path: list[BoardCoord],
        found_paths: list[list[BoardCoord]],
        wall_segments: list[tuple[BoardCoord, BoardCoord]],
    ) -> None:
        """Recursively collect valid paths starting from this trie node.

        Args:
            board: Current board rows.
            current_path: Path built so far, ending at this node.
            found_paths: Mutable output list for discovered valid paths.
            wall_segments: Pre-computed diagonal walls derived from blocked
                cells on the board.  Steps that cross any of these are
                rejected, and the list is forwarded to `leaves_small_island`.

        """
        if self.is_word and not leaves_small_island(board, current_path, wall_segments):
            found_paths.append(current_path.copy())

        current_coord = current_path[-1]
        row, col = current_coord
        for next_row, next_col in get_neighbor_coords(board, row, col):
            next_coord = (next_row, next_col)
            if next_coord in current_path:
                continue
            if path_would_self_cross(current_path, next_coord):
                continue
            if any(
                _segments_intersect(current_coord, next_coord, seg_start, seg_end)
                for seg_start, seg_end in wall_segments
            ):
                continue

            next_char = board[next_row][next_col]
            next_node = self.children.get(next_char)
            if next_node is None:
                continue

            current_path.append(next_coord)
            next_node.find_word_paths(board, current_path, found_paths, wall_segments)
            current_path.pop()


class Trie:
    """Trie of allowed words with board path-search helpers."""

    def __init__(self) -> None:
        """Initialize an empty trie."""
        self.root = Node()

    @classmethod
    def build_from_words(cls, words: list[str]) -> Self:
        """Build a trie from a word list.

        Args:
            words: Words to insert.

        Returns:
            Populated trie.

        """
        trie = cls()
        for word in words:
            trie.insert(word)
        return trie

    def insert(self, word: str) -> None:
        """Insert one word into the trie.

        Args:
            word: Word to insert.

        """
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]

        node.is_word = True

    @staticmethod
    def _rank_candidate_paths(board: list[str], candidate_paths: list[list[BoardCoord]]) -> list[list[BoardCoord]]:
        """Rank and deduplicate candidate paths.

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
        return [path for _, path in ranked_candidates]

    @staticmethod
    def _get_blocked_cell_wall_segments(board: list[str]) -> list[tuple[BoardCoord, BoardCoord]]:
        """Return wall segments implied by diagonally adjacent blocked cells.

        When a word is played its cells become blocked (`#`).  Any two
        consecutive cells in the word's path that are diagonal neighbours
        leave an invisible wall that remaining cells must not cross.  Since
        the board no longer stores the original order, we conservatively
        treat *every* pair of diagonally adjacent `#` cells as a wall
        segment — they must have been a traversed edge of some prior word.

        Args:
            board: Current board rows (blocked cells marked with `#`).

        Returns:
            List of undirected wall segments as ``(coord_a, coord_b)`` pairs.

        """
        blocked: set[BoardCoord] = {
            (row_idx, col_idx)
            for row_idx, row in enumerate(board)
            for col_idx, char in enumerate(row)
            if char == BLOCKED_CELL
        }
        segments: list[tuple[BoardCoord, BoardCoord]] = []
        for row, col in blocked:
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                neighbor: BoardCoord = (row + dr, col + dc)
                # Emit each pair once (smaller coord first) to avoid duplicates.
                if neighbor in blocked and neighbor > (row, col):
                    segments.append(((row, col), neighbor))
        return segments

    def find_all_word_paths(self, board: list[str]) -> list[list[BoardCoord]]:
        """Find all valid trie word paths in the given board.

        Args:
            board: Board rows.

        Returns:
            List of coordinate paths for all valid matched words.

        """
        wall_segments = self._get_blocked_cell_wall_segments(board)
        found_paths: list[list[BoardCoord]] = []

        for row_idx, row in enumerate(board):
            for col_idx, char in enumerate(row):
                start_node = self.root.children.get(char)
                if start_node is None:
                    continue

                start_coord = (row_idx, col_idx)
                start_node.find_word_paths(board, [start_coord], found_paths, wall_segments)

        return self._rank_candidate_paths(board, found_paths)
