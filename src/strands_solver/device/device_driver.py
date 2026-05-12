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
    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        """Execute a gesture path in pixel coordinates.

        Args:
            pixel_path: Ordered pixel coordinates representing the move path.
        """
