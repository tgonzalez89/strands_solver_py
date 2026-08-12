"""OpenCV + tesserocr board reader implementation."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Final, cast

from strands_solver.board_reader.board_reader import Highlight
from strands_solver.board_reader.board_reader_base import BoardReaderBase
from strands_solver.util.util import CIRCLE_COLOR_CHANNEL_TOLERANCE, board_to_text

try:
    import cv2
    import numpy as np
    import tesserocr
    from PIL import Image

    HAS_EXTRAS = True
except ModuleNotFoundError:
    HAS_EXTRAS = False

if TYPE_CHECKING:
    from strands_solver.board_reader.board_reader import CellStateGrid
    from strands_solver.device.device_driver import DeviceDriver
    from strands_solver.util.util import PixelCoord

TOP_LEFT_CELL_CENTER: Final = (155, 752)
BOTTOM_RIGHT_CELL_CENTER: Final = (920, 1791)

CIRCLE_DIAMETER: Final = 120

# RGB colors of the cell highlights (as observed in the screenshots):
NONE_COLOR_LIGHT: Final = (255, 255, 255)
NONE_COLOR_DARK: Final = (18, 18, 18)
WORD_COLOR: Final = (174, 223, 238)
SPANGRAM_COLOR: Final = (248, 205, 5)
SELECTION_COLOR: Final = (219, 216, 197)

PALETTE: Final[dict[tuple[int, int, int], Highlight]] = {
    NONE_COLOR_LIGHT: Highlight.NONE,
    NONE_COLOR_DARK: Highlight.NONE,
    WORD_COLOR: Highlight.WORD,
    SPANGRAM_COLOR: Highlight.SPANGRAM,
    SELECTION_COLOR: Highlight.SELECTED,
}

# Default Tesseract data directory, depending on the OS.
if os.name == "nt":  # Windows
    TESSDATA_DIR = rf"C:\Users\{os.getlogin()}\AppData\Local\Programs\Tesseract-OCR\tessdata"
elif os.name == "posix":
    TESSDATA_DIR = "/usr/share/tesseract-ocr/5/tessdata"
else:
    TESSDATA_DIR = "/usr/share/tesseract-ocr/5/tessdata"
TESSEDIT_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ| "
TESSEDIT_CHAR_BLACKLIST = "`~!@#$%^&*()-_=+[{]}\\;:'\",<.>/?]"


class BoardReaderTesseractOpenCV(BoardReaderBase):
    """Board reader implementation using OpenCV for image processing and Tesseract OCR."""

    def __init__(self, rows: int = 8, cols: int = 6, tessdata_dir: str | None = None) -> None:
        """Initialize board reader.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            tessdata_dir: Optional path to Tesseract tessdata directory.
                If not provided, uses `TESSDATA_PREFIX` environment variable when set,
                otherwise falls back to `TESSDATA_DIR`.

        """
        if not HAS_EXTRAS:
            msg = "Extra dependencies not found."
            raise NotImplementedError(msg)

        super().__init__(rows=rows, cols=cols)

        self._top_left_cell_center: PixelCoord = TOP_LEFT_CELL_CENTER
        self._bottom_right_cell_center: PixelCoord = BOTTOM_RIGHT_CELL_CENTER
        self._cell_height: int = 0
        self._cell_width: int = 0
        self._cell_centers: list[list[PixelCoord]] = []
        self._board_top_left: PixelCoord = (0, 0)
        self._estimated_circle_diameter: int = CIRCLE_DIAMETER  # Estimated from calibration
        self._recalculate_geometry()
        self._tessdata_dir = tessdata_dir or os.environ.get("TESSDATA_PREFIX", TESSDATA_DIR)

    def _recalculate_geometry(self) -> None:
        """Recompute cell dimensions and centers from the current corner coordinates."""
        tl = self._top_left_cell_center
        br = self._bottom_right_cell_center
        self._cell_height = (br[1] - tl[1]) // (self._rows - 1)
        self._cell_width = (br[0] - tl[0]) // (self._cols - 1)
        self._cell_centers = self._compute_cell_centers()
        self._board_top_left = (
            tl[0] - self._cell_width // 2,
            tl[1] - self._cell_height // 2,
        )

    def set_cell_corner_centers(
        self,
        top_left_cell_center: PixelCoord,
        bottom_right_cell_center: PixelCoord,
        circle_diameter: int | None = None,
    ) -> None:
        """Set board corner cell centers and recompute geometry.

        This is useful for non-interactive integrations that can provide known
        board geometry (for example synthetic/fake device renderers).

        Args:
            top_left_cell_center: Pixel center of cell (0, 0).
            bottom_right_cell_center: Pixel center of cell (rows-1, cols-1).
            circle_diameter: Optional circle diameter in pixels. If provided, overrides
                the estimated diameter. Defaults to current estimated value.

        """
        self._top_left_cell_center = top_left_cell_center
        self._bottom_right_cell_center = bottom_right_cell_center
        if circle_diameter is not None:
            self._estimated_circle_diameter = circle_diameter
        self._recalculate_geometry()

    def get_estimated_circle_diameter(self) -> int:
        """Get the circle diameter (estimated during calibration or from defaults).

        Returns:
            Estimated circle diameter in pixels.

        """
        return self._estimated_circle_diameter

    def calibrate(
        self,
        driver: DeviceDriver,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
    ) -> None:
        """Interactively calibrate cell-corner coordinates and circle diameter using the device.

        Prompts the user to tap the top-left and bottom-right board cells in turn.
        For each corner:

        1. Waits until no SELECTION_COLOR blobs are visible (asking the user to
           deselect all cells if needed).
        2. Prompts the user to tap the target cell.
        3. Polls screenshots until exactly one SELECTION_COLOR blob appears,
           recording its centroid as the corner coordinate.
        4. Taps that coordinate to deselect the cell before moving on.

        After both corners are captured, the internal geometry (cell width/height,
        cell centers, board top-left) is updated.

        The circle diameter is estimated from the detected blobs at both corners
        and averaged to produce a robust estimate.

        Args:
            driver: Device driver used to capture screenshots and send taps.
            timeout_s: Seconds to wait for the user to tap each corner.
            poll_interval_s: Seconds between successive screenshot polls.

        Raises:
            TimeoutError: If the user does not tap within `timeout_s` seconds.

        """
        corners: list[tuple[str, str]] = [
            ("TOP-LEFT", "_top_left_cell_center"),
            ("BOTTOM-RIGHT", "_bottom_right_cell_center"),
        ]

        estimated_diameters: list[float] = []

        for corner_name, attr in corners:
            # Step 1: Wait until screen is clear of any selection.
            while True:
                image = self._decode_image(driver.capture_screen())
                blobs = self._find_selection_blobs(image)
                if not blobs:
                    break
                print(f"Please deselect all cells ({len(blobs)} selected cell(s) detected)...")
                time.sleep(poll_interval_s)

            # Step 2: Prompt user.
            print(f"Please tap the {corner_name} cell...")

            # Step 3: Poll until exactly one blob appears.
            deadline = time.monotonic() + timeout_s
            center: PixelCoord | None = None
            diameter: float | None = None
            while time.monotonic() < deadline:
                time.sleep(poll_interval_s)
                image = self._decode_image(driver.capture_screen())
                blobs_with_diameter = self._find_selection_blobs_with_diameter(image)
                if len(blobs_with_diameter) == 1:
                    center, diameter = blobs_with_diameter[0]
                    break

            if center is None:
                msg = f"Calibration timed out waiting for {corner_name} tap after {timeout_s:.0f}s"
                raise TimeoutError(msg)

            if diameter is not None:
                estimated_diameters.append(diameter)
                print(f"  -> Detected {corner_name} at {center}, diameter ≈ {diameter:.1f}px")
            else:
                print(f"  -> Detected {corner_name} at {center}")

            setattr(self, attr, center)

            # Step 4: Tap to deselect.
            driver.tap(center)
            time.sleep(poll_interval_s)

        # Estimate circle diameter from both measurements and average them
        if estimated_diameters:
            self._estimated_circle_diameter = round(sum(estimated_diameters) / len(estimated_diameters))
            print(f"Estimated circle diameter: {self._estimated_circle_diameter}px")

        self._recalculate_geometry()
        print("Calibration complete.")

    @staticmethod
    def _detect_selection_contours(image: object) -> list[tuple[PixelCoord, float]]:
        """Detect SELECTION_COLOR blobs and return each blob's centroid and diameter.

        Applies a color-range mask around SELECTION_COLOR (within a tolerance),
        morphological opening for noise removal, and contour filtering by area.
        Diameter is estimated from the minimum enclosing circle of each contour.

        Args:
            image: Decoded OpenCV image (BGR, as returned by `_decode_image`).

        Returns:
            List of ``((x, y), diameter_px)`` tuples for each passing contour.

        """
        image_matlike = cast("cv2.typing.MatLike", image)

        # SELECTION_COLOR is RGB; convert to BGR for OpenCV.
        r, g, b = SELECTION_COLOR
        tol = CIRCLE_COLOR_CHANNEL_TOLERANCE
        lower = np.array([max(b - tol, 0), max(g - tol, 0), max(r - tol, 0)], dtype=np.uint8)
        upper = np.array([min(b + tol, 255), min(g + tol, 255), min(r + tol, 255)], dtype=np.uint8)

        mask = cv2.inRange(image_matlike, lower, upper)

        # Morphological opening to remove noise, then find contours.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area: a cell circle of CIRCLE_DIAMETER px has area ≈ π*(D/2)².
        min_area = 3.14159 * (CIRCLE_DIAMETER / 2) ** 2 * 0.3
        blobs: list[tuple[PixelCoord, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            moment = cv2.moments(contour)
            if moment["m00"] == 0:
                continue
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
            _, radius = cv2.minEnclosingCircle(contour)
            blobs.append(((cx, cy), 2 * radius))

        return blobs

    @staticmethod
    def _find_selection_blobs(image: object) -> list[PixelCoord]:
        """Locate blobs matching SELECTION_COLOR in a decoded image.

        Args:
            image: Decoded OpenCV image (BGR, as returned by `_decode_image`).

        Returns:
            List of (x, y) centroid coordinates for each detected blob.

        """
        return [center for center, _ in BoardReaderTesseractOpenCV._detect_selection_contours(image)]

    @staticmethod
    def _find_selection_blobs_with_diameter(image: object) -> list[tuple[PixelCoord, float]]:
        """Locate blobs matching SELECTION_COLOR and estimate their diameter.

        Args:
            image: Decoded OpenCV image (BGR, as returned by `_decode_image`).

        Returns:
            List of ((x, y), diameter) tuples for each detected blob.

        """
        return BoardReaderTesseractOpenCV._detect_selection_contours(image)

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

    def _extract_cell_centers(self, image: object) -> list[list[PixelCoord]]:
        """Locate each board cell center."""
        _ = image
        return self._cell_centers

    def _extract_cell_states(self, image: object) -> CellStateGrid:
        """Infer each cell highlight state from the decoded image."""
        image_matlike = cast("cv2.typing.MatLike", image)

        cell_states: CellStateGrid = [[Highlight.NONE for _ in range(self._cols)] for _ in range(self._rows)]

        # For each cell center:
        for row in range(self._rows):
            for col in range(self._cols):
                center = self._cell_centers[row][col]
                # Step 1: Extract a small patch of pixels around the center.
                cell_img = self._extract_cell_patch(image_matlike, center)
                # Step 2: Classify cell based on dominant background color.
                cell_states[row][col] = self._classify_cell_state(cell_img)

        return cell_states

    def _extract_board_rows(self, image: object) -> list[str]:
        """Extract raw OCR text for the board."""
        image_matlike = cast("cv2.typing.MatLike", image)
        board_img = self._preprocess_board_image(image_matlike)

        # Save the preprocessed board image for debugging.
        cv2.imwrite("debug_board.png", board_img)

        return self._ocr_board_rows(board_img)

    def _preprocess_board_image(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Build a thresholded board image suitable for OCR."""
        # Step 1: Preprocess the image to clean the background and enhance text visibility.
        #   a) Split the image into individual cell patches using the known cell centers.
        #   b) Apply thresholding to each individual cell image to isolate the letter from the background.
        #   c) Stitch the cleaned cell patches back into a single image for OCR.
        board_img_height = self._cell_height * self._rows
        board_img_width = self._cell_width * self._cols
        x_start_offset = (self._cell_width - self._patch_image_side()) // 2
        y_start_offset = (self._cell_height - self._patch_image_side()) // 2
        board_img = np.full((board_img_height, board_img_width), 255, dtype=np.uint8)
        for row in range(self._rows):
            for col in range(self._cols):
                center = self._cell_centers[row][col]
                cell_img = self._extract_cell_patch(image, center)
                cell_img_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                # Apply thresholding to create a binary image.
                # Binary image should have the letter in black and the background in white to improve OCR accuracy.
                # Use THRESH_BINARY if the letter is darker than the background (most cases).
                # Use THRESH_BINARY_INV if the letter is lighter than the background (dark mode cells, no highlight).
                thresh_flag = (
                    cv2.THRESH_BINARY_INV
                    if self._classify_cell_color(cell_img) == NONE_COLOR_DARK
                    else cv2.THRESH_BINARY
                )
                _, cell_img_binary = cv2.threshold(cell_img_gray, 0, 255, thresh_flag + cv2.THRESH_OTSU)
                # Add the binary cell image to the board image at the correct location.
                # Take into account that the cell image might be smaller than the cell size.
                x_start = col * self._cell_width + x_start_offset
                y_start = row * self._cell_height + y_start_offset
                x_end = x_start + cell_img_binary.shape[1]
                y_end = y_start + cell_img_binary.shape[0]
                board_img[y_start:y_end, x_start:x_end] = cell_img_binary

        return board_img

    def _ocr_board_rows(self, board_img: cv2.typing.MatLike) -> list[str]:
        """Run OCR against a preprocessed board image and return extracted rows."""
        # Step 2: Use Tesseract OCR to extract text from the preprocessed image.
        variables = {
            "load_system_dawg": "0",
            "load_freq_dawg": "0",
            "load_unambig_dawg": "0",
            "load_punc_dawg": "0",
            "load_number_dawg": "0",
            "load_bigram_dawg": "0",
        }

        board: list[str] = ["?" * self._cols for _ in range(self._rows)]
        result = self._ocr_cell_by_cell(board, board_img, self._tessdata_dir, variables)
        # Try again with whitelist.
        if not result:
            print("WARNING: OCR failed with initial configuration, retrying with character whitelist.")
            print(f"DEBUG:\n{board_to_text(board, ' ')}")
            result = self._ocr_cell_by_cell(
                board,
                board_img,
                self._tessdata_dir,
                variables | {"tessedit_char_whitelist": TESSEDIT_CHAR_WHITELIST},
            )
        if not result:
            msg = f"OCR failed to properly extract the letters from the board image.\n{board_to_text(board, ' ')}"
            raise RuntimeError(msg)

        return board

    def _ocr_cell_by_cell(
        self, board: list[str], board_img: cv2.typing.MatLike, tessdata: str, variables: dict[str, str]
    ) -> bool:
        """Perform OCR on each cell separately and fill the board with the results.

        Args:
            board: List of strings representing the board.
                   Must be pre-initialized with "?" for each cell to be filled by OCR.
            board_img: Preprocessed board image for OCR.
            tessdata: Path to Tesseract data files.
            variables: Tesseract configuration variables.

        Returns:
            True if all cells were successfully OCR'd, False if any cell failed OCR.

        """
        result = True

        with tesserocr.PyTessBaseAPI(path=tessdata, psm=tesserocr.PSM.SINGLE_CHAR, variables=variables) as tess_api:
            for row in range(self._rows):
                row_chars = list(board[row])
                for col in range(self._cols):
                    if board[row][col] != "?":
                        continue

                    x_start = col * self._cell_width
                    y_start = row * self._cell_height
                    cell_img = board_img[y_start : y_start + self._cell_height, x_start : x_start + self._cell_width]
                    tess_api.SetImage(Image.fromarray(cell_img))
                    ocr_result = tess_api.GetUTF8Text()
                    cleaned_cell = ocr_result.strip()
                    cleaned_cell = cleaned_cell.replace("|", "I")  # Known common OCR error.

                    if len(cleaned_cell) == 1 and cleaned_cell in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        # If OCR returns 'D', try fallback with SINGLE_BLOCK mode to see if we get 'P' instead.
                        if cleaned_cell == "D":
                            fallback_char = BoardReaderTesseractOpenCV._ocr_d_cell_fallback(
                                tess_api,
                                cell_img,
                            )
                            if fallback_char is not None:
                                cleaned_cell = fallback_char
                        row_chars[col] = cleaned_cell
                    else:
                        row_chars[col] = "?"
                        result = False
                        print(f"DEBUG: OCR failed for cell at row {row}, col {col}. {ocr_result=} {cleaned_cell=}")
                board[row] = "".join(row_chars)

        return result

    @staticmethod
    def _ocr_d_cell_fallback(tess_api: tesserocr.PyTessBaseAPI, cell_img: cv2.typing.MatLike) -> str | None:
        """Attempt fallback OCR with SINGLE_BLOCK mode when SINGLE_CHAR returns 'D'.

        The tuning experiments showed that SINGLE_BLOCK mode with appropriate preprocessing
        is more reliable for distinguishing P from D. This retry uses a tighter whitelist
        to force Tesseract to choose between P and D only.

        Args:
            tess_api: Active Tesseract API object to reuse.
            cell_img: Single cell image (binary, letter in black, background in white).

        Returns:
            'P' if fallback OCR returns 'P', None otherwise to keep original 'D'.

        """
        original_psm = tess_api.GetPageSegMode()
        original_whitelist = tess_api.GetStringVariable("tessedit_char_whitelist")
        tess_api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
        tess_api.SetVariable("tessedit_char_whitelist", TESSEDIT_CHAR_WHITELIST)

        try:
            tess_api.SetImage(Image.fromarray(cell_img))
            fallback_result = tess_api.GetUTF8Text()
            fallback_cleaned = fallback_result.strip()
            if fallback_cleaned == "P":
                return fallback_cleaned
        finally:
            tess_api.SetPageSegMode(original_psm)
            tess_api.SetVariable("tessedit_char_whitelist", original_whitelist)

        return None

    def _compute_cell_centers(self) -> list[list[PixelCoord]]:
        """Compute the pixel coordinates of each cell center based on the top-left and bottom-right corners.

        Returns:
            A 2D list of pixel coordinates for each cell center, indexed by [row][col].

        """
        cell_centers: list[list[PixelCoord]] = []

        for row in range(self._rows):
            row_centers: list[PixelCoord] = []
            for col in range(self._cols):
                center_x = self._top_left_cell_center[0] + col * self._cell_width
                center_y = self._top_left_cell_center[1] + row * self._cell_height
                row_centers.append((center_x, center_y))
            cell_centers.append(row_centers)

        return cell_centers

    def _extract_cell_patch(self, image: cv2.typing.MatLike, center: PixelCoord) -> cv2.typing.MatLike:
        """Extract a patch of pixels around the cell center.

        Args:
            image: OpenCV image in BGR color space.
            center: Pixel coordinates of the cell center.

        Returns:
            Image patch containing the cell.

        """
        img_cell_side = self._patch_image_side()
        x, y = center
        x_start = max(x - img_cell_side // 2, 0)
        y_start = max(y - img_cell_side // 2, 0)
        x_end = min(x + img_cell_side // 2, image.shape[1])
        y_end = min(y + img_cell_side // 2, image.shape[0])

        return image[y_start:y_end, x_start:x_end]

    @staticmethod
    def _classify_cell_state(cell_img: cv2.typing.MatLike) -> Highlight:
        """Classify cell by finding dominant background color via histogram.

        Args:
            cell_img: Cell patch image (numpy array).

        Returns:
            Highlight state corresponding to the dominant color.

        """
        closest = BoardReaderTesseractOpenCV._classify_cell_color(cell_img)
        return PALETTE[closest]

    def _patch_image_side(self, factor: float = 0.9) -> int:
        """Calculate the side length of the square patch image extracted around the cell center."""
        # Patch is an image with the side length of a square inscribed inside the circle,
        # to avoid capturing any colors outside the circle.
        # Side is calculated as diameter * sqrt(2) / 2 (approximated as 0.7).
        side = self._estimated_circle_diameter * 7 // 10
        # Apply additional factor to ensure we are well within the circle and
        # not capturing any background colors on the corners of the inscribed square.
        side = int(side * factor)
        # Round down to nearest even integer for symmetry.
        return side - (side % 2)

    @staticmethod
    def _classify_cell_color(cell_img: cv2.typing.MatLike) -> tuple[int, int, int]:
        # Reshape to Nx3 array of pixel colors and quantize to uint8.
        pixels = cell_img.reshape(-1, 3).astype(np.uint8)

        # Find unique colors and their counts to get the dominant background.
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        dominant_idx = np.argmax(counts)
        dominant_color_bgr = tuple(map(int, unique_colors[dominant_idx]))
        dominant_color_rgb = (dominant_color_bgr[2], dominant_color_bgr[1], dominant_color_bgr[0])

        # Map to palette via nearest Euclidean distance in RGB space.
        def squared_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

        return min(PALETTE, key=lambda known: squared_distance(dominant_color_rgb, known))
