"""Spangram-specific search utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strands_solver.solver.common import (
    PathSearchOptions,
    can_extend_path,
    get_neighbor_coords,
    is_spangram_path,
    leaves_small_island,
)
from strands_solver.util.util import MIN_WORD_LEN, BoardCoord, coords_to_word

if TYPE_CHECKING:
    from strands_solver.solver.dict_solver import Node, Trie


@dataclass(frozen=True, slots=True)
class SpangramSolverOptions(PathSearchOptions):
    """Options for searching spangram candidate paths."""

    dedupe_words: bool = True


def _dedupe_nodes(nodes: list[Node]) -> tuple[Node, ...]:
    """Return nodes deduplicated by object identity while preserving order."""
    result: list[Node] = []
    seen_ids: set[int] = set()
    for node in nodes:
        node_id = id(node)
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        result.append(node)
    return tuple(result)


def _advance_segment_nodes(trie: Trie, active_nodes: tuple[Node, ...], char: str) -> tuple[Node, ...]:
    """Advance a segmented-word automaton by one character."""
    transition_sources = list(active_nodes)
    if any(node.is_word for node in active_nodes):
        transition_sources.append(trie.root)

    next_nodes = [child for node in transition_sources if (child := node.children.get(char)) is not None]
    return _dedupe_nodes(next_nodes)


def _rank_spangram_paths(
    board: list[str],
    candidate_paths: list[list[BoardCoord]],
    *,
    dedupe_words: bool,
) -> list[list[BoardCoord]]:
    """Rank spangram candidates deterministically."""
    if dedupe_words:
        best_path_by_word: dict[str, list[BoardCoord]] = {}
        for path in candidate_paths:
            word = coords_to_word(board, path)
            existing_path = best_path_by_word.get(word)
            if existing_path is None or tuple(path) < tuple(existing_path):
                best_path_by_word[word] = path
        ranked = list(best_path_by_word.items())
    else:
        ranked = [(coords_to_word(board, path), path) for path in candidate_paths]

    ranked.sort(key=lambda item: (-len(item[0]), -len(set(item[0])), item[0], tuple(item[1])))
    return [path for _, path in ranked]


def find_all_spangram_paths(  # noqa: C901
    trie: Trie,
    board: list[str],
    wall_segments: list[tuple[BoardCoord, BoardCoord]] | None = None,
    *,
    options: SpangramSolverOptions | None = None,
) -> list[list[BoardCoord]]:
    """Find paths that span the board and can be segmented into dictionary words."""
    found_paths: list[list[BoardCoord]] = []
    search_options = options or SpangramSolverOptions()

    def dfs(current_path: list[BoardCoord], active_nodes: tuple[Node, ...]) -> None:
        if (
            len(current_path) >= MIN_WORD_LEN
            and any(node.is_word for node in active_nodes)
            and is_spangram_path(board, current_path)
            and (not search_options.reject_small_islands or not leaves_small_island(board, current_path))
        ):
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

            next_nodes = _advance_segment_nodes(trie, active_nodes, board[next_row][next_col])
            if not next_nodes:
                continue

            current_path.append(next_coord)
            dfs(current_path, next_nodes)
            current_path.pop()

    if not board or not board[0]:
        return []

    max_row = len(board) - 1
    max_col = len(board[0]) - 1
    for row_idx, row in enumerate(board):
        for col_idx, char in enumerate(row):
            if row_idx not in {0, max_row} and col_idx not in {0, max_col}:
                continue

            start_nodes = _advance_segment_nodes(trie, (trie.root,), char)
            if not start_nodes:
                continue

            dfs([(row_idx, col_idx)], start_nodes)

    return _rank_spangram_paths(board, found_paths, dedupe_words=search_options.dedupe_words)
