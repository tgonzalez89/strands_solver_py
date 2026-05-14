"""ADB-backed device driver implementation."""

import subprocess
import time
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
        tap_delay_ms: int = 100,
        swipe_duration_ms: int | None = None,
        command_timeout_s: float = 10.0,
    ) -> None:
        """Initialize the ADB-backed driver.

        Args:
            adb_path: Executable path or command name for `adb`.
            adb_server_host: Optional adb server host passed via `adb -H`.
            adb_server_port: Optional adb server port passed via `adb -P`.
            device_serial: Optional device serial to pass via `adb -s`.
            tap_delay_ms: Delay in milliseconds between taps.
            swipe_duration_ms: Deprecated alias for `tap_delay_ms`.
            command_timeout_s: Timeout in seconds per adb command.

        """
        self._adb_path = adb_path
        self._adb_server_host = adb_server_host
        self._adb_server_port = adb_server_port
        self._device_serial = device_serial
        self._tap_delay_ms = swipe_duration_ms if swipe_duration_ms is not None else tap_delay_ms
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

        return screenshot_bytes

    def tap(self, coord: PixelCoord) -> None:
        """Tap a single pixel coordinate on the device.

        Args:
            coord: Pixel coordinate to tap.

        Raises:
            NotImplementedError: If adb command execution fails.

        """
        x, y = coord
        self._run_adb_command(["shell", "input", "tap", str(x), str(y)])

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        """Execute a board path as a sequence of taps followed by a confirmation tap.

        Each coordinate in the path is tapped once to select it. After all cells
        are tapped, the last cell is tapped a second time to confirm the word.
        A short delay of ``tap_delay_ms`` milliseconds is inserted between
        each tap so the app can register them individually.

        Args:
            pixel_path: Ordered pixel coordinates to swipe through.

        Raises:
            ValueError: If `pixel_path` is empty.
            NotImplementedError: If adb command execution fails.

        """
        if not pixel_path:
            msg = "pixel_path must contain at least one coordinate"
            raise ValueError(msg)

        delay_s = self._tap_delay_ms / 1000.0

        for coord in pixel_path:
            self.tap(coord)
            time.sleep(delay_s)

        # Confirm by tapping the last cell a second time.
        self.tap(pixel_path[-1])
