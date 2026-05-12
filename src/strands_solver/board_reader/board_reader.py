"""Abstract board-reader interfaces and shared state types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands_solver.util.util import BoardCoord, PixelCoord


class Highlight(StrEnum):
    """Highlight state of a board cell, also used as a move outcome."""

    NONE = "none"
    WORD = "word"
    SPANGRAM = "spangram"


type CellStateGrid = list[list[Highlight]]


@dataclass
class BoardState:
    """Snapshot of extracted board state from a device screen."""

    board: list[str]
    cell_states: CellStateGrid | None = None


class BoardReader(ABC):
    """Abstract board-reader and move-feedback analyzer."""

    @abstractmethod
    def extract_state(self, screenshot: bytes) -> BoardState:
        """Parse one screenshot into board state metadata."""

    @abstractmethod
    def classify_feedback(self, before: BoardState, after: BoardState, move: list[BoardCoord]) -> Highlight:
        """Classify move feedback as none, word, or spangram."""

    @abstractmethod
    def board_move_to_pixel_path(self, move: list[BoardCoord]) -> list[PixelCoord]:
        """Convert board coordinates to device pixel coordinates.

        Args:
            move: Board coordinate path.

        Returns:
            Corresponding device pixel coordinates.

        Raises:
            ValueError: If cell geometry not available or path invalid.
        """
