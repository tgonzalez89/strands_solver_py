"""ADB-backed device driver implementation."""

import subprocess
from itertools import pairwise
from typing import TYPE_CHECKING

from strands_solver.device.device_driver import DeviceDriver

if TYPE_CHECKING:
    from strands_solver.util.util import PixelCoord


class DeviceDriverADB(DeviceDriver):
    """ADB driver for screenshot capture and path-gesture execution."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        adb_path: str = "adb",
        adb_server_host: str | None = None,
        adb_server_port: int | None = None,
        device_serial: str | None = None,
        swipe_duration_ms: int = 120,
        command_timeout_s: float = 15.0,
    ) -> None:
        """Initialize the ADB-backed driver.

        Args:
            adb_path: Executable path or command name for `adb`.
            adb_server_host: Optional adb server host passed via `adb -H`.
            adb_server_port: Optional adb server port passed via `adb -P`.
            device_serial: Optional device serial to pass via `adb -s`.
            swipe_duration_ms: Duration to use for each swipe call.
            command_timeout_s: Timeout in seconds per adb command.

        """
        self._adb_path = adb_path
        self._adb_server_host = adb_server_host
        self._adb_server_port = adb_server_port
        self._device_serial = device_serial
        self._swipe_duration_ms = swipe_duration_ms
        self._command_timeout_s = command_timeout_s

    def _build_adb_command(self, args: list[str]) -> list[str]:
        command = [self._adb_path]
        if self._adb_server_host:
            command.extend(["-H", self._adb_server_host])
        if self._adb_server_port is not None:
            command.extend(["-P", str(self._adb_server_port)])
        if self._device_serial:
            command.extend(["-s", self._device_serial])
        command.extend(args)
        return command

    def _run_adb_command(self, args: list[str]) -> bytes:
        command = self._build_adb_command(args)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=self._command_timeout_s,
            )
        except FileNotFoundError:
            msg = "adb executable not found. Install Android platform-tools and ensure `adb` is on PATH"
            raise NotImplementedError(msg) from None
        except subprocess.TimeoutExpired:
            msg = f"adb command timed out after {self._command_timeout_s:.1f}s"
            raise NotImplementedError(msg) from None

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            msg = f"adb command failed: {stderr or 'unknown error'}"
            raise NotImplementedError(msg)

        return result.stdout

    def capture_screen(self) -> bytes:
        """Capture one PNG screenshot from the connected device.

        Returns:
            PNG-encoded screenshot bytes.

        Raises:
            NotImplementedError: If adb is unavailable, times out, or returns no image.

        """
        screenshot_bytes = self._run_adb_command(["exec-out", "screencap", "-p"])
        if not screenshot_bytes:
            msg = "ADB returned an empty screenshot"
            raise NotImplementedError(msg)

        return screenshot_bytes.replace(b"\r\n", b"\n")

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        """Execute a board path as one or more ADB swipe gestures.

        Args:
            pixel_path: Ordered pixel coordinates to swipe through.

        Raises:
            ValueError: If `pixel_path` is empty.
            NotImplementedError: If adb command execution fails.

        """
        if not pixel_path:
            msg = "pixel_path must contain at least one coordinate"
            raise ValueError(msg)

        if len(pixel_path) == 1:
            start_x, start_y = pixel_path[0]
            self._run_adb_command(
                [
                    "shell",
                    "input",
                    "swipe",
                    str(start_x),
                    str(start_y),
                    str(start_x),
                    str(start_y),
                    str(self._swipe_duration_ms),
                ],
            )
            return

        for (start_x, start_y), (end_x, end_y) in pairwise(pixel_path):
            self._run_adb_command(
                [
                    "shell",
                    "input",
                    "swipe",
                    str(start_x),
                    str(start_y),
                    str(end_x),
                    str(end_y),
                    str(self._swipe_duration_ms),
                ],
            )
