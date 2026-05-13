"""CLI for running solver bots against fixtures or devices."""

import argparse
import sys
from pathlib import Path

from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCV
from strands_solver.bot.bot_device import BotDevice
from strands_solver.bot.bot_device_fake import BotDeviceFake, InitialOcrMismatchError
from strands_solver.bot.bot_fake import BotFake
from strands_solver.device.device_driver_appium import DeviceDriverAppium
from strands_solver.image_renderer.board_image_renderer import RenderConfig
from strands_solver.solver.solver import Trie
from strands_solver.util.util import BoardCoord, load_allowed_words, load_moves


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for running a solver bot."""
    parser = argparse.ArgumentParser(description="Run a Strands solver bot against board and move fixtures.")
    parser.add_argument("--allowed-words", "-w", type=Path, required=True, help="Path to allowed words file.")
    parser.add_argument(
        "--driver",
        choices=("file", "appium", "fake"),
        default="file",
        help="Execution backend: file fixtures, real appium, or generated fake screenshots.",
    )
    parser.add_argument("--board", "-b", type=Path, help="Path to board file.")
    parser.add_argument("--valid-moves", "-m", type=Path, help="Path to valid moves file.")
    parser.add_argument(
        "--spangram-index",
        type=int,
        action="append",
        default=[],
        help="0-based move index to classify as spangram in fake-driver mode.",
    )
    parser.add_argument(
        "--fake-mode",
        choices=("light", "dark"),
        default="light",
        help="Rendered board mode for fake-driver screenshots.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def _print_successful_moves(successful_moves: list[tuple[str, list[BoardCoord]]]) -> None:
    for word, path in successful_moves:
        print(f"{word}: {path}")


def _run_appium_mode(args: argparse.Namespace, trie: Trie) -> int:
    if args.board is not None or args.valid_moves is not None:
        print("--board/--valid-moves are not used with --driver appium", file=sys.stderr)
        return 2

    bot = BotDevice(driver=DeviceDriverAppium(), reader=BoardReaderTesseractOpenCV())
    try:
        successful_moves = bot.run(trie, verbose=args.verbose)
    except NotImplementedError as error:
        print(f"device_mode_not_ready: {error}", file=sys.stderr)
        return 2

    _print_successful_moves(successful_moves)
    print(f"matched={len(successful_moves)}")
    return 0


def _run_fake_mode(args: argparse.Namespace, trie: Trie) -> int:
    if args.board is None or args.valid_moves is None:
        print("--driver fake requires --board and --valid-moves", file=sys.stderr)
        return 2

    try:
        bot = BotDeviceFake(
            board=args.board,
            valid_moves=args.valid_moves,
            spangram_indexes={index for index in args.spangram_index if index >= 0},
            mode=args.fake_mode,
            render_config=RenderConfig(),
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


def _run_file_mode(args: argparse.Namespace, trie: Trie) -> int:
    if args.board is None or args.valid_moves is None:
        print("--driver file requires --board and --valid-moves", file=sys.stderr)
        return 2

    valid_moves = load_moves(args.valid_moves)
    bot = BotFake(args.board, valid_moves)
    successful_moves = bot.run(trie, verbose=args.verbose)
    _print_successful_moves(successful_moves)

    print(f"matched={len(successful_moves)}/{len(valid_moves)}")
    if len(successful_moves) < len(valid_moves):
        print("final_board:")
        for row in bot.get_board():
            print(" ".join(row))

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI workflow and print matched moves."""
    args = parse_args(argv)

    words = load_allowed_words(args.allowed_words)
    trie = Trie.build_from_words(words)

    if args.driver == "appium":
        return _run_appium_mode(args, trie)

    if args.driver == "fake":
        return _run_fake_mode(args, trie)

    return _run_file_mode(args, trie)


if __name__ == "__main__":
    sys.exit(main())
