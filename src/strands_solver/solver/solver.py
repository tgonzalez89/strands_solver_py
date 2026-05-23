"""Compatibility exports for solver functionality."""

from strands_solver.solver.common import (
    PathSearchOptions,
    board_has_spangram,
    can_extend_path,
    diagonal_wall_segments,
    get_neighbor_coords,
    has_open_cells,
    is_spangram_path,
    leaves_small_island,
    open_cell_count,
    path_would_self_cross,
    segments_intersect,
)
from strands_solver.solver.dict_solver import DictionarySolverOptions, Node, Trie
from strands_solver.solver.open_path_solver import OpenPathSolverOptions, find_all_open_paths
from strands_solver.solver.spangram_solver import SpangramSolverOptions, find_all_spangram_paths

__all__ = [
    "DictionarySolverOptions",
    "Node",
    "OpenPathSolverOptions",
    "PathSearchOptions",
    "SpangramSolverOptions",
    "Trie",
    "board_has_spangram",
    "can_extend_path",
    "diagonal_wall_segments",
    "find_all_open_paths",
    "find_all_spangram_paths",
    "get_neighbor_coords",
    "has_open_cells",
    "is_spangram_path",
    "leaves_small_island",
    "open_cell_count",
    "path_would_self_cross",
    "segments_intersect",
]
