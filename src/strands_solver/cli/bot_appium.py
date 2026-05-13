"""CLI for running the Appium-backed solver bot."""

import argparse
import sys
from pathlib import Path
from typing import cast

from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCV
from strands_solver.bot.bot_device import BotDevice
from strands_solver.device.device_driver_appium import AppiumSession, DeviceDriverAppium
from strands_solver.solver.solver import Trie
from strands_solver.util.util import BoardCoord, load_allowed_words


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Appium-backed bot."""
    parser = argparse.ArgumentParser(description="Run a Strands solver bot on a real Android device via Appium.")
    parser.add_argument("--allowed-words", "-w", type=Path, required=True, help="Path to allowed words file.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--appium-url",
        default="http://localhost:4723",
        help="Appium server URL (default: http://localhost:4723).",
    )
    parser.add_argument("--device-name", default="", help="ADB device serial (from `adb devices`).")
    parser.add_argument(
        "--app-package",
        default="com.nytimes.android",
        help="Android app package name (default: com.nytimes.android).",
    )
    parser.add_argument(
        "--app-activity",
        default="",
        help="Android activity that shows the Strands board (e.g. .games.strands.StrandsActivity).",
    )
    return parser.parse_args(argv)


def _print_successful_moves(successful_moves: list[tuple[str, list[BoardCoord]]]) -> None:
    for word, path in successful_moves:
        print(f"{word}: {path}")


def _create_appium_session(args: argparse.Namespace) -> AppiumSession:
    """Create an Appium session from CLI arguments.

    Args:
        args: Parsed CLI arguments containing Appium connection parameters.

    Returns:
        An initialized Appium WebDriver session.

    Raises:
        ImportError: If `appium-python-client` is not installed.

    """
    try:
        from appium import webdriver  # type: ignore[import-untyped]  # noqa: PLC0415
        from appium.options.android import UiAutomator2Options  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        msg = "appium-python-client is required for Appium mode. Install it with: uv sync --extra device"
        raise ImportError(msg) from None

    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.no_reset = True
    if args.device_name:
        options.device_name = args.device_name
    if args.app_package:
        options.app_package = args.app_package
    if args.app_activity:
        options.app_activity = args.app_activity

    return cast("AppiumSession", webdriver.Remote(args.appium_url, options=options))  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    """Run the Appium CLI workflow and print matched moves."""
    args = parse_args(argv)

    words = load_allowed_words(args.allowed_words)
    trie = Trie.build_from_words(words)

    driver = DeviceDriverAppium()
    try:
        session = _create_appium_session(args)
    except ImportError as error:
        print(f"device_mode_not_ready: {error}", file=sys.stderr)
        return 2

    driver.attach_session(session)
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
