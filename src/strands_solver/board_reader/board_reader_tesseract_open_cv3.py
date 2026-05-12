"""OpenCV + tesserocr board reader (v2) skeleton."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader_base import BoardReaderBase

try:
    import cv2
    import numpy as np

    HAS_EXTRAS = True
except ModuleNotFoundError:
    HAS_EXTRAS = False

if TYPE_CHECKING:
    from strands_solver.board_reader.board_reader import CellStateGrid


class BoardReaderTesseractOpenCV3(BoardReaderBase):
    """Board reader implementation using OpenCV for image processing and Tesseract for OCR."""

    def __init__(self, rows: int = 8, cols: int = 6) -> None:
        """Initialize board reader."""
        if not HAS_EXTRAS:
            msg = "Extra dependencies not found."
            raise NotImplementedError(msg)
        super().__init__(rows=rows, cols=cols)

    def _decode_image(self, screenshot: bytes) -> object:
        """Decode screenshot bytes into an OpenCV BGR image.

        Args:
            screenshot: PNG/JPEG screenshot bytes.

        Returns:
            Decoded OpenCV image (`numpy.ndarray` in BGR color space).

        Raises:
            ValueError: If screenshot bytes are empty or cannot be decoded.

        """
        if not screenshot:
            msg = "screenshot cannot be empty"
            raise ValueError(msg)

        image_bytes = np.frombuffer(screenshot, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            msg = "Unable to decode screenshot bytes as an image"
            raise ValueError(msg)

        return image

    def _extract_cell_centers(self, image: object) -> list[list[tuple[int, int]]]:
        """TODO: Locate each board cell center."""
        msg = "TODO: implement cell center extraction"
        raise NotImplementedError(msg)

    def _extract_cell_states(self, image: object) -> CellStateGrid:
        """TODO: Infer each cell highlight state from the decoded image."""
        msg = "TODO: implement cell-state extraction"
        raise NotImplementedError(msg)

    def _extract_board_rows(self, image: object) -> list[str]:
        """TODO: Extract raw OCR text for teh board."""
        msg = "TODO: implement board row extraction"
        raise NotImplementedError(msg)
