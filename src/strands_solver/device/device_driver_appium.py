"""Appium-backed device driver implementation."""

from itertools import pairwise
from typing import TYPE_CHECKING, Protocol

from strands_solver.device.device_driver import DeviceDriver

if TYPE_CHECKING:
    from strands_solver.util.util import PixelCoord


class AppiumSession(Protocol):
    """Protocol describing Appium session methods used by the driver."""

    def get_screenshot_as_png(self) -> bytes:
        """Return the current screenshot encoded as PNG bytes.

        Returns:
            PNG-encoded screenshot bytes.

        """

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int) -> None:
        """Perform one swipe gesture.

        Args:
            start_x: Start x-coordinate in pixels.
            start_y: Start y-coordinate in pixels.
            end_x: End x-coordinate in pixels.
            end_y: End y-coordinate in pixels.
            duration: Swipe duration in milliseconds.

        """


class DeviceDriverAppium(DeviceDriver):
    """Appium driver for screenshot and path-gesture execution."""

    def __init__(self, session: AppiumSession | None = None, swipe_duration_ms: int = 100) -> None:
        """Initialize the Appium-backed driver.

        Args:
            session: Optional active Appium session.
            swipe_duration_ms: Duration to use for each swipe call.

        """
        self._session = session
        self._swipe_duration_ms = swipe_duration_ms

    def attach_session(self, session: AppiumSession) -> None:
        """Attach or replace the Appium session.

        Args:
            session: Initialized Appium session.

        """
        self._session = session

    def _get_session(self) -> AppiumSession:
        """Return the configured Appium session.

        Returns:
            Active Appium session.

        Raises:
            NotImplementedError: If no session has been attached yet.

        """
        if self._session is None:
            msg = "Appium session is not configured yet"
            raise NotImplementedError(msg)
        return self._session

    def capture_screen(self) -> bytes:
        """Capture one PNG screenshot from the session.

        Returns:
            PNG-encoded screenshot bytes.

        Raises:
            NotImplementedError: If the screenshot payload is empty.

        """
        session = self._get_session()
        screenshot_bytes = session.get_screenshot_as_png()
        if screenshot_bytes:
            return screenshot_bytes
        msg = "Appium returned an empty screenshot"
        raise NotImplementedError(msg)

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        """Execute a board path as one or more Appium swipes.

        Args:
            pixel_path: Ordered pixel coordinates to swipe through.

        Raises:
            ValueError: If `pixel_path` is empty.

        """
        if not pixel_path:
            msg = "pixel_path must contain at least one coordinate"
            raise ValueError(msg)

        session = self._get_session()
        if len(pixel_path) == 1:
            start_x, start_y = pixel_path[0]
            session.swipe(start_x, start_y, start_x, start_y, self._swipe_duration_ms)
            return

        for (start_x, start_y), (end_x, end_y) in pairwise(pixel_path):
            session.swipe(start_x, start_y, end_x, end_y, self._swipe_duration_ms)

    def tap(self, coord: PixelCoord) -> None:
        """Tap a single pixel coordinate via a zero-duration swipe.

        Args:
            coord: Pixel coordinate to tap.

        Raises:
            NotImplementedError: If no Appium session is attached.

        """
        session = self._get_session()
        x, y = coord
        session.swipe(x, y, x, y, 0)
