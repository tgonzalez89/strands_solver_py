"""OpenCV + tesserocr board reader (v2) skeleton."""

import importlib
from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader_base import BoardReaderBase

if TYPE_CHECKING:
    from strands_solver.board_reader.board_reader import CellStateGrid


class BoardReaderTesseractOpenCV2(BoardReaderBase):
    """Board reader skeleton that will use OpenCV for vision and tesserocr for OCR.

    This class intentionally provides only the shared scaffolding plus image decoding.
    Remaining extraction hooks are left as TODOs for iterative implementation.
    """

    def __init__(self, rows: int = 8, cols: int = 6) -> None:
        """Initialize reader with board geometry and placeholder settings."""
        super().__init__(rows=rows, cols=cols)
        self.debug_mode = False
        self.debug_output_dir = ".debug"
        self.save_debug_images = False
        self.save_ocr_logs = True
        self.current_input_name: str | None = None

    def _decode_image(self, screenshot: bytes) -> object:
        """Decode screenshot bytes into an OpenCV BGR image.

        Args:
            screenshot: PNG/JPEG screenshot bytes.

        Returns:
            Decoded OpenCV image (`numpy.ndarray` in BGR color space).

        Raises:
            ValueError: If screenshot bytes are empty or cannot be decoded.
            NotImplementedError: If OpenCV or NumPy are unavailable.
        """
        if not screenshot:
            raise ValueError("screenshot cannot be empty")

        try:
            cv2 = importlib.import_module("cv2")
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise NotImplementedError("OpenCV and NumPy are required to decode screenshots") from exc

        image_bytes = np.frombuffer(screenshot, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Unable to decode screenshot bytes as an image")

        return image

    def _extract_cell_centers(self, image: object) -> list[list[tuple[int, int]]]:
        """TODO: Locate each board cell center."""
        raise NotImplementedError("TODO: implement cell center extraction")

    def _extract_cell_states(self, image: object) -> CellStateGrid:
        """TODO: Infer each cell highlight state from the decoded image."""
        raise NotImplementedError("TODO: implement cell-state extraction")

    def _extract_board_rows(self, image: object) -> list[str]:
        """TODO: Extract raw OCR text for teh board."""
        raise NotImplementedError("TODO: implement board row extraction")
