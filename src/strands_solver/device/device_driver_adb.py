"""ADB-backed device driver implementation."""

import subprocess
import time
from typing import TYPE_CHECKING

from strands_solver.device.device_driver import DeviceDriver

if TYPE_CHECKING:
    from strands_solver.board_reader.board_reader import BoardReader, BoardState
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
        tap_delay_ms: int = 0,
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

    def execute_path(
        self,
        pixel_path: list[PixelCoord],
        board_reader: BoardReader | None = None,
    ) -> None:
        """Execute a board path as a sequence of taps followed by a confirmation tap.

        Each coordinate in the path is tapped once to select it. After all cells
        are tapped, the last cell is tapped a second time to confirm the word.
        A short delay of ``tap_delay_ms`` milliseconds is inserted between
        each tap so the app can register them individually.

        Args:
            pixel_path: Ordered pixel coordinates to swipe through.
            board_reader: Optional board reader used to verify that individual
                tap gestures changed the expected cell state.

        Raises:
            ValueError: If `pixel_path` is empty.
            NotImplementedError: If adb command execution fails.
            RuntimeError: If a cell state change is not detected after tapping.

        """
        if not pixel_path:
            msg = "pixel_path must contain at least one coordinate"
            raise ValueError(msg)

        if board_reader is None:
            for coord in pixel_path:
                self.tap(coord)
                time.sleep(self._tap_delay_ms / 1000.0)
            # Confirm by tapping the last cell a second time.
            self.tap(pixel_path[-1])
            return

        before_state = board_reader.extract_state(self.capture_screen())
        for coord in pixel_path:
            self.tap(coord)
            self._wait_until_cell_state_changes(coord, board_reader, before_state)
            before_state = board_reader.extract_state(self.capture_screen())

        # Confirm by tapping the last cell a second time.
        self.tap(pixel_path[-1])

    def _wait_until_cell_state_changes(  # noqa: C901
        self,
        pixel_coord: PixelCoord,
        board_reader: BoardReader,
        before_state: BoardState,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.1,
    ) -> None:
        """Wait until the tapped cell's state differs from the previous board state."""
        if before_state.cell_states is None:
            msg = "Board reader did not provide cell state metadata for tap verification"
            raise RuntimeError(msg)

        cell_centers = getattr(board_reader, "_cell_centers", [])
        if not cell_centers:
            msg = "Board reader geometry is not available for tap verification"
            raise RuntimeError(msg)

        row_idx = -1
        col_idx = -1
        best_distance = float("inf")
        for row, row_centers in enumerate(cell_centers):
            for col, center in enumerate(row_centers):
                distance = abs(center[0] - pixel_coord[0]) + abs(center[1] - pixel_coord[1])
                if distance < best_distance:
                    best_distance = distance
                    row_idx = row
                    col_idx = col

        if row_idx < 0 or col_idx < 0:
            msg = "Could not map tap coordinate to board cell for verification"
            raise RuntimeError(msg)

        deadline = time.monotonic() + timeout_s
        previous_state = before_state.cell_states[row_idx][col_idx]

        while time.monotonic() < deadline:
            after_state = board_reader.extract_state(self.capture_screen())
            if after_state.cell_states is None:
                msg = "Board reader did not provide cell state metadata for tap verification"
                raise RuntimeError(msg)
            if row_idx >= len(after_state.cell_states) or col_idx >= len(after_state.cell_states[row_idx]):
                msg = "Board reader returned inconsistent cell state dimensions during tap verification"
                raise RuntimeError(msg)
            if after_state.cell_states[row_idx][col_idx] != previous_state:
                return
            time.sleep(poll_interval_s)

        msg = f"Timed out waiting for tapped cell at {pixel_coord} to change state"
        raise RuntimeError(msg)
