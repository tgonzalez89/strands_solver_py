"""CLI for running the fake-device solver bot."""

import argparse
import sys
from pathlib import Path

from strands_solver.bot.bot_device_fake import BotDeviceFake, InitialOcrMismatchError
from strands_solver.image_renderer.board_image_renderer import RenderConfig
from strands_solver.solver.solver import Trie
from strands_solver.util.util import BoardCoord, load_allowed_words


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the fake-device bot."""
    parser = argparse.ArgumentParser(description="Run a Strands solver bot against fake rendered screenshots.")
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
    parser.add_argument(
        "--fake-mode",
        choices=("light", "dark"),
        default="light",
        help="Rendered board mode for fake screenshots.",
    )
    parser.add_argument(
        "--tessdata-dir",
        default="",
        help="Tesseract tessdata directory. Defaults to TESSDATA_PREFIX env var or built-in path.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def _print_successful_moves(successful_moves: list[tuple[str, list[BoardCoord]]]) -> None:
    for word, path in successful_moves:
        print(f"{word}: {path}")


def main(argv: list[str] | None = None) -> int:
    """Run the fake-device CLI workflow and print matched moves."""
    args = parse_args(argv)

    words = load_allowed_words(args.allowed_words)
    trie = Trie.build_from_words(words)

    try:
        bot = BotDeviceFake(
            board=args.board,
            valid_moves=args.valid_moves,
            spangram_indexes={index for index in args.spangram_index if index >= 0},
            mode=args.fake_mode,
            render_config=RenderConfig(),
            tessdata_dir=args.tessdata_dir or None,
        )
    except InitialOcrMismatchError as error:
        print(f"fake_mode_ocr_mismatch: {error}", file=sys.stderr)
        return 2

    successful_moves = bot.run(trie, verbose=args.verbose)
    _print_successful_moves(successful_moves)

    print(f"matched={len(successful_moves)}/{bot.expected_move_count}")
    if len(successful_moves) < bot.expected_move_count:
        print("final_board:")
        for row in bot.get_board():
            print(" ".join(row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
