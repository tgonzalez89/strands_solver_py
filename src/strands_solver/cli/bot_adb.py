"""CLI for running the ADB-backed solver bot."""

import argparse
import sys
from pathlib import Path

from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCV
from strands_solver.bot.bot_device import BotDevice
from strands_solver.device.device_driver_adb import DeviceDriverADB
from strands_solver.solver.solver import Trie
from strands_solver.util.util import BoardCoord, load_allowed_words


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the ADB-backed bot."""
    parser = argparse.ArgumentParser(description="Run a Strands solver bot on a real Android device via ADB.")
    parser.add_argument("--allowed-words", "-w", type=Path, required=True, help="Path to allowed words file.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--adb-path", default="adb", help="ADB executable path or command name.")
    parser.add_argument("--adb-host", default="", help="ADB server host (passed as `adb -H`).")
    parser.add_argument("--adb-port", type=int, default=0, help="ADB server port (passed as `adb -P`).")
    parser.add_argument("--device-serial", default="", help="ADB device serial (from `adb devices`).")
    parser.add_argument("--swipe-duration-ms", type=int, default=120, help="Duration in ms for each swipe segment.")
    parser.add_argument("--adb-timeout-s", type=float, default=15.0, help="Timeout in seconds per adb command.")
    return parser.parse_args(argv)


def _print_successful_moves(successful_moves: list[tuple[str, list[BoardCoord]]]) -> None:
    for word, path in successful_moves:
        print(f"{word}: {path}")


def main(argv: list[str] | None = None) -> int:
    """Run the ADB CLI workflow and print matched moves."""
    args = parse_args(argv)

    words = load_allowed_words(args.allowed_words)
    trie = Trie.build_from_words(words)

    driver = DeviceDriverADB(
        adb_path=args.adb_path,
        adb_server_host=args.adb_host or None,
        adb_server_port=args.adb_port if args.adb_port > 0 else None,
        device_serial=args.device_serial or None,
        swipe_duration_ms=args.swipe_duration_ms,
        command_timeout_s=args.adb_timeout_s,
    )
    bot = BotDevice(driver=driver, reader=BoardReaderTesseractOpenCV())
    try:
        successful_moves = bot.run(trie, verbose=args.verbose)
    except NotImplementedError as error:
        print(f"device_mode_not_ready: {error}", file=sys.stderr)
        return 2

    _print_successful_moves(successful_moves)
    print(f"matched={len(successful_moves)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
