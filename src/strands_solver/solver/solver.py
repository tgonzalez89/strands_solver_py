"""Trie-based solver utilities for finding valid word paths on a board."""

from dataclasses import dataclass, field
from typing import Self

from strands_solver.util.util import BLOCKED_CELL, MIN_WORD_LEN, BoardCoord


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


def leaves_small_island(board: list[str], removed_path: list[BoardCoord]) -> bool:
    """Check whether removing a path creates too-small connected islands.

    Args:
        board: Current board rows.
        removed_path: Coordinates treated as removed for this validation check.

    Returns:
        True if any remaining connected component has size less than
        `MIN_WORD_LEN`; otherwise False.
    """
    removed_coords = set(removed_path)
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
            row, col = stack.pop()
            island_size += 1

            for neighbor in get_neighbor_coords(board, row, col):
                if neighbor in unseen_coords:
                    unseen_coords.remove(neighbor)
                    stack.append(neighbor)

        if island_size < MIN_WORD_LEN:
            return True

    return False


@dataclass
class Node:
    """Trie node containing children and terminal-word metadata."""

    is_word: bool = False
    children: dict[str, Self] = field(default_factory=dict)

    def find_word_paths(
        self, board: list[str], current_path: list[BoardCoord], found_paths: list[list[BoardCoord]]
    ) -> None:
        """Recursively collect valid paths starting from this trie node.

        Args:
            board: Current board rows.
            current_path: Path built so far, ending at this node.
            found_paths: Mutable output list for discovered valid paths.
        """
        if self.is_word and not leaves_small_island(board, current_path):
            found_paths.append(current_path.copy())

        row, col = current_path[-1]
        for next_row, next_col in get_neighbor_coords(board, row, col):
            next_coord = (next_row, next_col)
            if next_coord in current_path:
                continue

            next_char = board[next_row][next_col]
            next_node = self.children.get(next_char)
            if next_node is None:
                continue

            current_path.append(next_coord)
            next_node.find_word_paths(board, current_path, found_paths)
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

    def find_all_word_paths(self, board: list[str]) -> list[list[BoardCoord]]:
        """Find all valid trie word paths in the given board.

        Args:
            board: Board rows.

        Returns:
            List of coordinate paths for all valid matched words.
        """
        found_paths: list[list[BoardCoord]] = []

        for row_idx, row in enumerate(board):
            for col_idx, char in enumerate(row):
                start_node = self.root.children.get(char)
                if start_node is None:
                    continue

                start_coord = (row_idx, col_idx)
                start_node.find_word_paths(board, [start_coord], found_paths)

        return found_paths
