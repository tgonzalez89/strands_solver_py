"""Dictionary-backed path finding for Strands boards."""

from dataclasses import dataclass, field
from typing import Self

from strands_solver.solver.common import PathSearchOptions, can_extend_path, get_neighbor_coords, leaves_small_island
from strands_solver.util.util import BoardCoord, coords_to_word


@dataclass(frozen=True, slots=True)
class DictionarySolverOptions(PathSearchOptions):
    """Options for dictionary-based board search."""

    dedupe_words: bool = True


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
        *,
        wall_segments: list[tuple[BoardCoord, BoardCoord]] | None = None,
        options: DictionarySolverOptions,
    ) -> None:
        """Recursively collect valid paths starting from this trie node."""
        if self.is_word and (not options.reject_small_islands or not leaves_small_island(board, current_path)):
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
                prevent_self_crossing=options.prevent_self_crossing,
                use_wall_segments=options.use_wall_segments,
            ):
                continue

            next_char = board[next_row][next_col]
            next_node = self.children.get(next_char)
            if next_node is None:
                continue

            current_path.append(next_coord)
            next_node.find_word_paths(
                board,
                current_path,
                found_paths,
                wall_segments=wall_segments,
                options=options,
            )
            current_path.pop()


class Trie:
    """Trie of allowed words with board path-search helpers."""

    def __init__(self) -> None:
        """Initialize an empty trie."""
        self.root = Node()

    @classmethod
    def build_from_words(cls, words: list[str]) -> Self:
        """Build a trie from a word list."""
        trie = cls()
        for word in words:
            trie.insert(word)
        return trie

    def insert(self, word: str) -> None:
        """Insert one word into the trie."""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]

        node.is_word = True

    @staticmethod
    def _rank_candidate_paths(
        board: list[str],
        candidate_paths: list[list[BoardCoord]],
        *,
        dedupe_words: bool,
    ) -> list[list[BoardCoord]]:
        """Rank candidate paths using deterministic heuristics."""
        if dedupe_words:
            best_path_by_word: dict[str, list[BoardCoord]] = {}

            for path in candidate_paths:
                word = coords_to_word(board, path)
                existing_path = best_path_by_word.get(word)
                if existing_path is None or tuple(path) < tuple(existing_path):
                    best_path_by_word[word] = path

            ranked_candidates = list(best_path_by_word.items())
        else:
            ranked_candidates = [(coords_to_word(board, path), path) for path in candidate_paths]

        ranked_candidates.sort(
            key=lambda item: (-len(item[0]), -len(set(item[0])), item[0], tuple(item[1])),
        )
        return [path for _, path in ranked_candidates]

    def find_all_word_paths(
        self,
        board: list[str],
        wall_segments: list[tuple[BoardCoord, BoardCoord]] | None = None,
        *,
        options: DictionarySolverOptions | None = None,
    ) -> list[list[BoardCoord]]:
        """Find all valid trie word paths in the given board."""
        found_paths: list[list[BoardCoord]] = []
        search_options = options or DictionarySolverOptions()

        for row_idx, row in enumerate(board):
            for col_idx, char in enumerate(row):
                start_node = self.root.children.get(char)
                if start_node is None:
                    continue

                start_coord = (row_idx, col_idx)
                start_node.find_word_paths(
                    board,
                    [start_coord],
                    found_paths,
                    wall_segments=wall_segments if search_options.use_wall_segments else None,
                    options=search_options,
                )

        return self._rank_candidate_paths(board, found_paths, dedupe_words=search_options.dedupe_words)
