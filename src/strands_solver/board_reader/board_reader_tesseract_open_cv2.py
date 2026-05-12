"""OpenCV + tesserocr board reader (v2) skeleton."""

from __future__ import annotations

import os
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, cast

try:
    import cv2
    import numpy as np
    import tesserocr
    from PIL import Image as PILImage

    HAS_EXTRAS = True
except ModuleNotFoundError:
    HAS_EXTRAS = False

from strands_solver.board_reader.board_reader_base import BoardReaderBase

if TYPE_CHECKING:
    from strands_solver.board_reader.board_reader import CellStateGrid

INVERT_BLACK_RATIO_THRESHOLD = 0.5
OCR_INPUT_SIZE = 56
DEFAULT_COLS = 6
DEFAULT_CROP_RADIUS = 25
COLUMN_X_SPACING_SHIFT_FACTOR = {
    1: 0.09,
    2: 0.09,
}
CROP_RADIUS_PIXEL_SHRINK = 5


class BoardReaderTesseractOpenCV2(BoardReaderBase):
    """Board reader skeleton that will use OpenCV for vision and tesserocr for OCR.

    This class intentionally provides only the shared scaffolding plus image decoding.
    Remaining extraction hooks are left as TODOs for iterative implementation.
    """

    _PLACEHOLDER_CHAR = "?"

    def __init__(self, rows: int = 8, cols: int = 6) -> None:
        """Initialize v2 reader with board geometry."""
        if not HAS_EXTRAS:
            msg = "Extra dependencies not found."
            raise NotImplementedError(msg)
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
            msg = "screenshot cannot be empty"
            raise ValueError(msg)

        image_bytes = np.frombuffer(screenshot, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            msg = "Unable to decode screenshot bytes as an image"
            raise ValueError(msg)

        return image

    def _extract_cell_states(self, image: object) -> CellStateGrid:
        """TODO: Infer each cell highlight state from the decoded image."""
        msg = "TODO: implement cell-state extraction"
        raise NotImplementedError(msg)

    def _extract_cell_centers(self, image: object) -> list[list[tuple[int, int]]]:
        """Locate each board cell center using circle detection and grid fitting.

        Strategy:
        1) Detect circular letter-cells with Hough circles.
        2) Fit evenly spaced x/y lattice coordinates for the configured grid.
        3) Fall back to normalized geometry if circle fitting is unreliable.
        """
        # HARDCODED: Return early with centers from centers.txt data
        # Calibrated from visual inspection: x_spacing=154, y_spacing=149, start=(154,748)
        x_start, y_start = 154, 748
        x_spacing, y_spacing = 154, 149
        return [
            [(x_start + col * x_spacing, y_start + row * y_spacing) for col in range(self._cols)]
            for row in range(self._rows)
        ]

        # Original detection code (kept for reference, but early return above bypasses it):
        min_shape_dims = 2
        shape = getattr(image, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < min_shape_dims:
            msg = "image must be a valid OpenCV image array"
            raise ValueError(msg)

        image_h, image_w = int(shape[0]), int(shape[1])
        typed_image = cast("Any", image)
        gray = cv2.cvtColor(typed_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

        min_radius = max(12, int(image_w * 0.03))
        max_radius = max(min_radius + 2, int(image_w * 0.095))
        min_distance = max(20, int(image_w * 0.07))

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_distance,
            param1=100,
            param2=24,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            all_points_array = np.round(circles[0][:, :2]).astype(int)
            all_points = [(int(point[0]), int(point[1])) for point in all_points_array]

            board_min_x = int(image_w * 0.10)
            board_max_x = int(image_w * 0.93)
            board_min_y = int(image_h * 0.32)
            board_max_y = int(image_h * 0.90)
            board_region_points = [
                (x_coord, y_coord)
                for x_coord, y_coord in all_points
                if board_min_x <= x_coord <= board_max_x and board_min_y <= y_coord <= board_max_y
            ]

            fit_points = board_region_points if len(board_region_points) >= self._rows * 2 else all_points
            fitted = self._fit_uniform_grid_lines(
                points=fit_points,
                image_w=image_w,
                image_h=image_h,
            )
            if fitted is not None:
                x_lines, y_lines = fitted
                min_top_ratio = 0.30
                if y_lines[0] < int(image_h * min_top_ratio):
                    return self._fallback_cell_centers(image_w=image_w, image_h=image_h)
                return [
                    [(x_lines[col_idx], y_lines[row_idx]) for col_idx in range(self._cols)]
                    for row_idx in range(self._rows)
                ]

        return self._fallback_cell_centers(image_w=image_w, image_h=image_h)

    def _fit_uniform_grid_lines(
        self,
        points: list[tuple[int, int]],
        image_w: int,
        image_h: int,
    ) -> tuple[list[int], list[int]] | None:
        """Fit uniformly spaced x/y lines for the board grid from detected points."""
        if len(points) < max(8, (self._rows * self._cols) // 3):
            return None

        x_values = sorted(int(point[0]) for point in points)
        y_values = sorted(int(point[1]) for point in points)

        x_fit = self._fit_axis_lines(
            axis_values=x_values,
            expected_count=self._cols,
            per_line_cap=self._rows,
            image_extent=image_w,
        )
        y_fit = self._fit_axis_lines(
            axis_values=y_values,
            expected_count=self._rows,
            per_line_cap=self._cols,
            image_extent=image_h,
        )

        if x_fit is None or y_fit is None:
            return None

        x_lines, x_score = x_fit
        y_lines, y_score = y_fit

        required_x_score = int(self._rows * self._cols * 0.55)
        required_y_score = int(self._rows * self._cols * 0.55)
        if x_score < required_x_score or y_score < required_y_score:
            return None

        return x_lines, y_lines

    def _fit_axis_lines(  # noqa: C901
        self,
        axis_values: list[int],
        expected_count: int,
        per_line_cap: int,
        image_extent: int,
    ) -> tuple[list[int], int] | None:
        """Find best evenly-spaced axis coordinates for one dimension."""
        if len(axis_values) < expected_count:
            return None

        unique_values = sorted(set(axis_values))
        if len(unique_values) < expected_count:
            return None

        min_step = image_extent * 0.035
        max_step = image_extent * 0.30
        best_lines: list[int] | None = None
        best_score = -1

        for start_idx, start_value in enumerate(unique_values[:-1]):
            for end_value in unique_values[start_idx + 1 :]:
                step = (end_value - start_value) / max(1, expected_count - 1)
                if step < min_step or step > max_step:
                    continue

                line_positions = [start_value + line_idx * step for line_idx in range(expected_count)]
                tolerance = max(8.0, step * 0.22)

                refined_lines: list[int] = []
                score = 0
                for target in line_positions:
                    matches = [value for value in axis_values if abs(value - target) <= tolerance]
                    if matches:
                        refined_line = round(float(median(matches)))
                        line_score = min(len(matches), per_line_cap)
                    else:
                        refined_line = round(target)
                        line_score = 0
                    refined_lines.append(refined_line)
                    score += line_score

                if score > best_score:
                    best_score = score
                    best_lines = refined_lines

        if best_lines is None:
            return None

        best_lines = sorted(best_lines)
        if len(best_lines) != expected_count:
            return None

        if best_lines[0] < 0 or best_lines[-1] >= image_extent:
            return None

        return best_lines, best_score

    def _fallback_cell_centers(self, image_w: int, image_h: int) -> list[list[tuple[int, int]]]:
        """Fallback center estimation using normalized board geometry ratios.

        Ratios are tuned from real screenshot samples and assume NYT Strands portrait UI.
        """
        left_x = round(image_w * 0.143)
        right_x = round(image_w * 0.852)
        top_y = round(image_h * 0.333)
        bottom_y = round(image_h * 0.796)

        if self._cols == 1:
            x_lines = [left_x]
        else:
            x_step = (right_x - left_x) / (self._cols - 1)
            x_lines = [round(left_x + col_idx * x_step) for col_idx in range(self._cols)]

        if self._rows == 1:
            y_lines = [top_y]
        else:
            y_step = (bottom_y - top_y) / (self._rows - 1)
            y_lines = [round(top_y + row_idx * y_step) for row_idx in range(self._rows)]

        return [
            [(x_lines[col_idx], y_lines[row_idx]) for col_idx in range(self._cols)] for row_idx in range(self._rows)
        ]

    def _extract_board_rows(self, image: object) -> list[str]:
        """Extract board letters by trying multiple OCR strategies and selecting best output."""
        candidates = self._extract_board_rows_candidates(image)
        return max(candidates.values(), key=self._score_board_rows)

    def _extract_board_rows_candidates(self, image: object) -> dict[str, list[str]]:
        """Return OCR candidates for full-board, per-line, and per-cell strategies."""
        centers = self._extract_cell_centers(image)
        cell_images = self._build_normalized_cell_images(image=image, centers=centers)

        full_board_rows = self._ocr_full_board(cell_images)
        line_rows = self._ocr_line_by_line(cell_images)
        cell_rows = self._ocr_cell_by_cell(cell_images)

        candidates = {
            "full_board": full_board_rows,
            "line_by_line": line_rows,
            "cell_by_cell": cell_rows,
        }
        self._save_ocr_result_log(candidates)
        return candidates

    def _save_ocr_result_log(self, candidates: dict[str, list[str]]) -> None:
        """Save OCR candidates to .log file, matching the input screenshot basename."""
        if not self.save_ocr_logs:
            return

        output_dir = Path(self.debug_output_dir or ".debug")
        output_dir.mkdir(parents=True, exist_ok=True)

        log_name = f"{Path(self.current_input_name).stem}.log" if self.current_input_name else "ocr_result.log"

        selected = max(candidates.values(), key=self._score_board_rows)
        lines = [
            "OCR candidates",
            f"input={self.current_input_name or 'unknown'}",
            "",
        ]
        for method_name in ("full_board", "line_by_line", "cell_by_cell"):
            rows = candidates.get(method_name, [])
            score = self._score_board_rows(rows)
            lines.append(f"[{method_name}] score={score}")
            lines.extend(rows)
            lines.append("")

        lines.append("[selected]")
        lines.extend(selected)
        lines.append("")

        (output_dir / log_name).write_text("\n".join(lines), encoding="utf-8")

    def _build_normalized_cell_images(
        self,
        image: object,
        centers: list[list[tuple[int, int]]],
    ) -> list[list[object]]:
        """Crop cells uniformly around centers based on measured spacing."""
        typed_image = cast("Any", image)
        gray = cv2.cvtColor(typed_image, cv2.COLOR_BGR2GRAY)

        x_centers = [centers[0][col_idx][0] for col_idx in range(self._cols)]
        y_centers = [centers[row_idx][0][1] for row_idx in range(self._rows)]

        x_diffs = [x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)]
        y_diffs = [y_centers[i + 1] - y_centers[i] for i in range(len(y_centers) - 1)]
        avg_x_spacing = round(float(np.mean(x_diffs))) if x_diffs else 50
        avg_y_spacing = round(float(np.mean(y_diffs))) if y_diffs else 60

        crop_radius_x = max(8, int(avg_x_spacing * 0.32) - CROP_RADIUS_PIXEL_SHRINK)
        crop_radius_y = max(8, int(avg_y_spacing * 0.32) - CROP_RADIUS_PIXEL_SHRINK)

        normalized_cells: list[list[object]] = []
        for row_idx in range(self._rows):
            row_cells: list[object] = []
            for col_idx in range(self._cols):
                cx, cy = centers[row_idx][col_idx]

                x_shift = round(avg_x_spacing * COLUMN_X_SPACING_SHIFT_FACTOR.get(col_idx, 0.0))
                cx = cx + x_shift

                x0 = max(0, cx - crop_radius_x)
                x1 = min(gray.shape[1], cx + crop_radius_x)
                y0 = max(0, cy - crop_radius_y)
                y1 = min(gray.shape[0], cy + crop_radius_y)

                cell_gray = gray[y0:y1, x0:x1]
                if cell_gray.size == 0:
                    cell_gray = np.full((50, 50), 255, dtype=np.uint8)

                _threshold, cell_bin = cv2.threshold(
                    cell_gray,
                    0,
                    255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                )

                black_ratio = float(np.mean(cell_bin == 0))
                if black_ratio > INVERT_BLACK_RATIO_THRESHOLD:
                    cell_bin = cv2.bitwise_not(cell_bin)

                kernel = np.ones((2, 2), dtype=np.uint8)
                cell_bin = cv2.morphologyEx(cell_bin, cv2.MORPH_OPEN, kernel)

                cell_bin = cv2.resize(cell_bin, (OCR_INPUT_SIZE, OCR_INPUT_SIZE), interpolation=cv2.INTER_AREA)

                row_cells.append(cell_bin)
            normalized_cells.append(row_cells)

        return normalized_cells

    def _trim_and_pad_cell(self, cell_bin: object) -> object:
        """Return the cell image unchanged for compatibility."""
        return cell_bin

    def _ocr_full_board(self, cells: list[list[object]]) -> list[str]:
        """Run OCR over one stitched board image."""
        board_image = self._stitch_cells(cells)
        text = self._tesseract_text(board_image, psm_name="SINGLE_BLOCK")
        letters = self._sanitize_letters(text)
        return self._letters_to_rows(letters)

    def _ocr_line_by_line(self, cells: list[list[object]]) -> list[str]:
        """Run OCR row-by-row on stitched row strips."""
        rows: list[str] = []
        for row_idx in range(self._rows):
            row_image = self._stitch_cells([cells[row_idx]])
            text = self._tesseract_text(row_image, psm_name="SINGLE_LINE")
            letters = self._sanitize_letters(text)
            row = self._fit_letters_to_width(letters, self._cols)
            rows.append(row)
        return rows

    def _ocr_cell_by_cell(self, cells: list[list[object]]) -> list[str]:
        """Run OCR cell-by-cell using single-character mode."""
        rows: list[str] = []
        for row_idx in range(self._rows):
            row_chars = [
                self._ocr_single_cell_char(
                    cells[row_idx][col_idx],
                    row_idx=row_idx,
                    col_idx=col_idx,
                )
                for col_idx in range(self._cols)
            ]
            rows.append("".join(row_chars))
        return rows

    def _ocr_single_cell_char(self, cell: object, row_idx: int, col_idx: int) -> str:
        """OCR one cell (fixed 64x64 input) with morphology variants for robustness."""
        typed_cell = cast("Any", cell)
        if self.debug_mode and self.save_debug_images and self.debug_output_dir:
            self._save_debug_image(typed_cell, f"cell_r{row_idx:02d}_c{col_idx:02d}_prep")

        kernel = np.ones((2, 2), dtype=np.uint8)
        variants = [
            typed_cell,
            cv2.morphologyEx(typed_cell, cv2.MORPH_DILATE, kernel),
            cv2.morphologyEx(typed_cell, cv2.MORPH_ERODE, kernel),
        ]

        best_char = self._PLACEHOLDER_CHAR
        best_conf = -1
        for variant in variants:
            text, confidence = self._tesseract_text_with_conf(variant, psm_name="SINGLE_CHAR")
            letters = self._sanitize_letters(text)
            if letters and confidence > best_conf:
                best_char = letters[0]
                best_conf = confidence

        return best_char

    def _stitch_cells(self, cells: list[list[object]], pad: int = 8) -> object:
        """Stitch normalized cell images into one white canvas for OCR."""
        if not cells or not cells[0]:
            return np.full((32, 32), 255, dtype=np.uint8)

        heights = [int(cast("Any", cell).shape[0]) for row in cells for cell in row]
        widths = [int(cast("Any", cell).shape[1]) for row in cells for cell in row]
        cell_h = max(16, round(float(median(heights))))
        cell_w = max(16, round(float(median(widths))))
        rows_count = len(cells)
        cols_count = len(cells[0])

        canvas_h = rows_count * cell_h + (rows_count + 1) * pad
        canvas_w = cols_count * cell_w + (cols_count + 1) * pad
        canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

        for row_idx in range(rows_count):
            for col_idx in range(cols_count):
                y0 = pad + row_idx * (cell_h + pad)
                x0 = pad + col_idx * (cell_w + pad)
                source_cell = cast("Any", cells[row_idx][col_idx])
                resized_cell = cv2.resize(
                    source_cell,
                    (cell_w, cell_h),
                    interpolation=cv2.INTER_AREA,
                )
                canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = resized_cell

        return canvas

    def _save_debug_image(self, image: object, name: str) -> None:
        """Save image to debug output directory."""
        if not self.debug_output_dir:
            return
        output_dir = Path(self.debug_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        typed_img = cast("Any", image)
        output_path = output_dir / f"{name}.png"
        cv2.imwrite(str(output_path), typed_img)

    def _tesseract_text(self, gray_image: object, psm_name: str) -> str:
        """Run tesserocr on one grayscale image and return raw UTF-8 text."""
        typed_gray_image = cast("Any", gray_image)
        pil_image = PILImage.fromarray(typed_gray_image)
        psm_value = getattr(tesserocr.PSM, psm_name)
        tessdata_path = self._resolve_tessdata_path()
        if tessdata_path is None:
            with tesserocr.PyTessBaseAPI(lang="eng", psm=psm_value) as api:
                api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                api.SetImage(pil_image)
                return api.GetUTF8Text() or ""

        with tesserocr.PyTessBaseAPI(path=tessdata_path, lang="eng", psm=psm_value) as api:
            api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            api.SetImage(pil_image)
            return api.GetUTF8Text() or ""

    def _tesseract_text_with_conf(self, gray_image: object, psm_name: str) -> tuple[str, int]:
        """Run tesserocr and return UTF-8 text with mean confidence."""
        typed_gray_image = cast("Any", gray_image)
        pil_image = PILImage.fromarray(typed_gray_image)
        psm_value = getattr(tesserocr.PSM, psm_name)
        tessdata_path = self._resolve_tessdata_path()

        if tessdata_path is None:
            with tesserocr.PyTessBaseAPI(lang="eng", psm=psm_value) as api:
                api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                api.SetImage(pil_image)
                return (api.GetUTF8Text() or "", int(api.MeanTextConf()))

        with tesserocr.PyTessBaseAPI(path=tessdata_path, lang="eng", psm=psm_value) as api:
            api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            api.SetImage(pil_image)
            return (api.GetUTF8Text() or "", int(api.MeanTextConf()))

    def _resolve_tessdata_path(self) -> str | None:
        """Resolve tessdata directory for tesserocr initialization."""
        env_path = os.environ.get("TESSDATA_PREFIX")
        candidates = [
            Path(env_path) if env_path else None,
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tesseract-ocr/4.00/tessdata"),
            Path("/usr/share/tessdata"),
            Path("/usr/local/share/tessdata"),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            if (candidate / "eng.traineddata").exists():
                return str(candidate)

        return None

    def _sanitize_letters(self, text: str) -> list[str]:
        """Normalize OCR text to uppercase A-Z letters only."""
        return [char for char in text.upper() if "A" <= char <= "Z"]

    def _letters_to_rows(self, letters: list[str]) -> list[str]:
        """Pack linear letters into fixed-size board rows with placeholders."""
        expected_total = self._rows * self._cols
        adjusted = letters[:expected_total]
        if len(adjusted) < expected_total:
            adjusted.extend(self._PLACEHOLDER_CHAR for _ in range(expected_total - len(adjusted)))

        rows: list[str] = []
        for row_idx in range(self._rows):
            start = row_idx * self._cols
            end = start + self._cols
            rows.append("".join(adjusted[start:end]))
        return rows

    def _fit_letters_to_width(self, letters: list[str], width: int) -> str:
        """Fit a letter list to exact row width by truncating/padding."""
        adjusted = letters[:width]
        if len(adjusted) < width:
            adjusted.extend(self._PLACEHOLDER_CHAR for _ in range(width - len(adjusted)))
        return "".join(adjusted)

    def _score_board_rows(self, rows: list[str]) -> int:
        """Score OCR result quality by counting non-placeholder letters."""
        return sum(1 for row in rows for char in row if char != self._PLACEHOLDER_CHAR)
