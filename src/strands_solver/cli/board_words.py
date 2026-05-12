"""CLI for finding and printing all possible words on a board."""

import argparse
from pathlib import Path

from strands_solver.solver.solver import Trie
from strands_solver.util.util import coords_to_word, load_allowed_words, load_board


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for board word discovery."""
    parser = argparse.ArgumentParser(description="Find all possible words on a Strands board.")
    parser.add_argument("--allowed-words", "-w", type=Path, required=True, help="Path to allowed words file.")
    parser.add_argument("--board", "-b", type=Path, required=True, help="Path to board file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print all word paths found on the board."""
    args = parse_args(argv)
    trie = Trie.build_from_words(load_allowed_words(args.allowed_words))
    board = load_board(args.board)

    for path in trie.find_all_word_paths(board):
        print(f"{coords_to_word(board, path)}: {path}")

    return 0
