"""Abstract device-driver interfaces."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands_solver.util.util import PixelCoord


class DeviceDriver(ABC):
    """Abstract device-control adapter."""

    @abstractmethod
    def capture_screen(self) -> bytes:
        """Capture the current device screen.

        Returns:
            Encoded screenshot bytes.

        """

    @abstractmethod
    def tap(self, coord: PixelCoord) -> None:
        """Tap a single pixel coordinate on the device.

        Args:
            coord: Pixel coordinate to tap.

        """

    @abstractmethod
    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        """Execute a gesture path in pixel coordinates.

        Args:
            pixel_path: Ordered pixel coordinates representing the move path.

        """
