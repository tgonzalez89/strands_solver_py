"""CLI for running the file-based solver bot."""

import argparse
import sys
from pathlib import Path

from strands_solver.bot.bot_fake import BotFake
from strands_solver.solver.solver import Trie
from strands_solver.util.util import BoardCoord, load_allowed_words, load_moves


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the file-based bot."""
    parser = argparse.ArgumentParser(description="Run a Strands solver bot using board and move files.")
    parser.add_argument("--allowed-words", "-w", type=Path, required=True, help="Path to allowed words file.")
    parser.add_argument("--board", "-b", type=Path, required=True, help="Path to board file.")
    parser.add_argument("--valid-moves", "-m", type=Path, required=True, help="Path to valid moves file.")
    parser.add_argument(
        "--spangram-index",
        type=int,
        action="append",
        default=[],
        help="0-based move index to classify as spangram (repeatable).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def _print_successful_moves(successful_moves: list[tuple[str, list[BoardCoord]]]) -> None:
    for word, path in successful_moves:
        print(f"{word}: {path}")


def main(argv: list[str] | None = None) -> int:
    """Run the file-mode CLI workflow and print matched moves."""
    args = parse_args(argv)

    words = load_allowed_words(args.allowed_words)
    spangram_words = load_allowed_words(args.allowed_words, min_word_len=1)
    trie = Trie.build_from_words(words)
    spangram_trie = Trie.build_from_words(spangram_words)

    valid_moves = load_moves(args.valid_moves)
    bot = BotFake(
        args.board,
        valid_moves,
        spangram_indexes={index for index in args.spangram_index if index >= 0},
    )
    successful_moves = bot.run(trie, spangram_trie=spangram_trie, verbose=args.verbose)
    _print_successful_moves(successful_moves)

    print(f"matched={len(successful_moves)}/{len(valid_moves)}")
    if len(successful_moves) < len(valid_moves):
        print("final_board:")
        for row in bot.get_board():
            print(" ".join(row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
