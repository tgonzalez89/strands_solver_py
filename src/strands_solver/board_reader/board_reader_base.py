"""Grid-aware base class for screenshot-based board readers."""

import zlib
from abc import abstractmethod
from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader import BoardReader, BoardState, CellStateGrid, Highlight
from strands_solver.util.util import MAX_BOARD_DIMENSION, MIN_BOARD_DIMENSION

if TYPE_CHECKING:
    from strands_solver.util.util import BoardCoord, PixelCoord


class BoardReaderBase(BoardReader):
    """Concrete base for grid-based screenshot board readers.

    Implements caching, coordinate-to-pixel conversion, feedback classification,
    and board marking as generic template logic. Subclasses provide the
    image-library-specific implementations of decoding, board detection,
    color/state sampling, and text extraction.
    """

    def __init__(self, rows: int = 8, cols: int = 6) -> None:
        """Initialize grid geometry and internal state caches.

        Args:
            rows: Expected board row count.
            cols: Expected board column count.

        Raises:
            ValueError: If grid size is outside allowed thresholds.

        """
        if rows < MIN_BOARD_DIMENSION or cols < MIN_BOARD_DIMENSION:
            msg = f"rows and cols must both be >= {MIN_BOARD_DIMENSION}"
            raise ValueError(msg)
        if rows > MAX_BOARD_DIMENSION or cols > MAX_BOARD_DIMENSION:
            msg = f"rows and cols must both be <= {MAX_BOARD_DIMENSION}"
            raise ValueError(msg)

        self._rows = rows
        self._cols = cols
        self._last_screenshot_hash: int | None = None
        self._last_state: BoardState | None = None
        self._cell_centers: list[list[PixelCoord]] = []

    # ------------------------------------------------------------------
    # Abstract hooks - implemented by concrete subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _decode_image(self, screenshot: bytes) -> object:
        """Decode raw screenshot bytes into an image object.

        Args:
            screenshot: PNG/JPEG screenshot bytes.

        Returns:
            Decoded image in the format expected by the other hooks.

        Raises:
            ValueError: If the bytes cannot be decoded into a valid image.
            NotImplementedError: If required image dependencies are not installed.

        """

    @abstractmethod
    def _extract_cell_states(self, image: object) -> CellStateGrid:
        """Determine each cell's highlight state from the image and cell centers.

        Args:
            image: Decoded image returned by `_decode_image`.

        Returns:
            2D list of cell states corresponding to the board grid.

        """

    @abstractmethod
    def _extract_cell_centers(self, image: object) -> list[list[PixelCoord]]:
        """Determine the pixel coordinates of each cell center from the image.

        Args:
            image: Decoded image returned by `_decode_image`.

        Returns:
            2D list of pixel coordinates corresponding to the center of each cell.

        """

    @abstractmethod
    def _extract_board_rows(self, image: object) -> list[str]:
        """Extract board letter rows from the image.

        Args:
            image: Decoded image returned by `_decode_image`.

        Returns:
            List of strings, one per board row, each of length ``self._cols``.

        """

    # ------------------------------------------------------------------
    # BoardReader interface - concrete generic implementations
    # ------------------------------------------------------------------

    def extract_state(self, screenshot: bytes) -> BoardState:
        """Return board state parsed from one screenshot, with caching.

        Args:
            screenshot: PNG/JPEG screenshot bytes.

        Returns:
            Parsed board state.

        Raises:
            ValueError: If the screenshot is empty or not a valid image.
            NotImplementedError: If required image dependencies are not installed.

        """
        if not screenshot:
            msg = "screenshot cannot be empty"
            raise ValueError(msg)

        screenshot_hash = zlib.crc32(screenshot)
        if self._last_screenshot_hash == screenshot_hash and self._last_state is not None:
            return self._last_state

        image = self._decode_image(screenshot)
        cell_states = self._extract_cell_states(image)

        # These two are only needed once.
        if not self._cell_centers:
            self._cell_centers = self._extract_cell_centers(image)
        board = self._extract_board_rows(image) if self._last_state is None else self._last_state.board

        state = BoardState(board, cell_states)
        self._last_screenshot_hash = screenshot_hash
        self._last_state = state
        return state

    def classify_feedback(self, before: BoardState, after: BoardState, move: list[BoardCoord]) -> Highlight:
        """Classify move outcome by comparing cell states before and after.

        Args:
            before: Board state before executing the move.
            after: Board state after executing the move.
            move: Move coordinates to evaluate.

        Returns:
            Classified move outcome as none, word, or spangram.

        """
        if not move:
            return Highlight.NONE

        before_states = before.cell_states
        after_states = after.cell_states
        if after_states is None:
            return Highlight.NONE
        if before_states is None:
            before_states = [[Highlight.NONE for _ in range(self._cols)] for _ in range(self._rows)]

        move_before_states = [before_states[row_idx][col_idx] for row_idx, col_idx in move]
        move_after_states = [after_states[row_idx][col_idx] for row_idx, col_idx in move]

        was_unselected = all(state == Highlight.NONE for state in move_before_states)
        # If any cell in the move was already selected before,
        # the feedback is unreliable and should be classified as NONE.
        if not was_unselected:
            return Highlight.NONE

        has_spangram = any(state == Highlight.SPANGRAM for state in move_after_states)
        has_word = any(state == Highlight.WORD for state in move_after_states)

        if has_spangram:
            return Highlight.SPANGRAM
        if has_word:
            return Highlight.WORD
        return Highlight.NONE

    def board_move_to_pixel_path(self, move: list[BoardCoord]) -> list[PixelCoord]:
        """Convert board coordinates to device pixel coordinates.

        Args:
            move: Board coordinate path.

        Returns:
            Corresponding device pixel coordinates.

        Raises:
            ValueError: If cell geometry is not available or a coordinate is missing.

        """
        if not self._cell_centers:
            msg = "Cell geometry not available; call extract_state first"
            raise ValueError(msg)

        return [self._cell_centers[row_idx][col_idx] for row_idx, col_idx in move]
