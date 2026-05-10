"""Command-line entrypoint for the Strands solver tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from strands_solver.bot import StrandsGameBotTest
from strands_solver.io_utils import coords_to_word, load_allowed_words, load_board, load_moves
from strands_solver.solver import Trie


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for running the solver.

    Args:
        argv: Optional explicit argv list. When omitted, reads from `sys.argv`.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Run a Strands solver bot against board and move fixtures.")
    parser.add_argument("--allowed-words", "-w", type=Path, required=True, help="Path to allowed words file.")
    parser.add_argument("--board", "-b", type=Path, required=True, help="Path to board file.")
    parser.add_argument("--valid-moves", "-m", type=Path, help="Path to valid moves file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI workflow and print matched moves.

    Args:
        argv: Optional explicit argv list. When omitted, reads from `sys.argv`.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    words = load_allowed_words(args.allowed_words)
    trie = Trie.build_from_words(words)

    if args.valid_moves is None:
        board = load_board(args.board)
        found_paths = trie.find_all_word_paths(board)
        for path in found_paths:
            print(f"{coords_to_word(board, path)}: {path}")
        return 0

    valid_moves = load_moves(args.valid_moves)
    bot = StrandsGameBotTest(args.board, args.valid_moves)
    successful_moves = bot.run(trie)

    for word, path in successful_moves:
        print(f"{word}: {path}")

    print(f"matched={len(successful_moves)}/{len(valid_moves)}")
    if len(successful_moves) < len(valid_moves):
        print("final_board:")
        for row in bot.get_board():
            print(" ".join(row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
