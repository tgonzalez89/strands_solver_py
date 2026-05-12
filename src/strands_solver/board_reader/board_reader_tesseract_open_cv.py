"""OpenCV + Tesseract board-reader implementation."""

from __future__ import annotations

import colorsys
import re
from typing import TYPE_CHECKING, Any, cast

try:
    import cv2
    import numpy as np
    import tesserocr
    from PIL import Image as PILImage

    HAS_EXTRAS = True
except ModuleNotFoundError:
    HAS_EXTRAS = False

from strands_solver.board_reader.board_reader import CellStateGrid, Highlight
from strands_solver.board_reader.board_reader_base import BoardReaderBase

if TYPE_CHECKING:
    from strands_solver.util.util import BoardCoord, PixelCoord


class BoardReaderTesseractOpenCv(BoardReaderBase):
    """Board reader using OpenCV for image analysis and Tesseract for OCR."""

    _YELLOW_HUE_LOW = 0.10
    _YELLOW_HUE_HIGH = 0.18
    _YELLOW_MIN_SAT = 0.35
    _YELLOW_MIN_VAL = 0.35
    _BLUE_HUE_LOW = 0.52
    _BLUE_HUE_HIGH = 0.72
    _BLUE_MIN_SAT = 0.25
    _BLUE_MIN_VAL = 0.25
    _MIN_BOARD_AREA_FRACTION = 0.12
    _FALLBACK_X_RATIO = 0.13
    _FALLBACK_Y_RATIO = 0.275
    _FALLBACK_WIDTH_RATIO = 0.74
    _FALLBACK_HEIGHT_RATIO = 0.53
    _OCR_INSET_RATIO = 0.20
    _OCR_FALLBACK_INSET_RATIOS = (0.20, 0.12, 0.16, 0.24)
    _OCR_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _OCR_MIN_GOOD_CONFIDENCE = 75
    _OCR_BOARD_VARIANT_INSETS = (0.00, 0.02)
    _TEXT_GUIDED_MIN_COMPONENTS = 12
    _TEXT_GUIDED_MIN_MASK_FILL_RATIO = 0.004
    _TEXT_GUIDED_MAX_MASK_FILL_RATIO = 0.35
    _TEXT_GUIDED_MIN_COMPONENT_AREA_RATIO = 0.010
    _TEXT_GUIDED_MAX_COMPONENT_AREA_RATIO = 0.45
    _TEXT_GUIDED_MIN_COMPONENT_WIDTH_RATIO = 0.03
    _TEXT_GUIDED_MAX_COMPONENT_WIDTH_RATIO = 0.80
    _TEXT_GUIDED_MIN_COMPONENT_HEIGHT_RATIO = 0.12
    _TEXT_GUIDED_MAX_COMPONENT_HEIGHT_RATIO = 0.90
    _TEXT_GUIDED_MAX_SPACING_CV = 0.35
    _TEXT_GUIDED_MIN_BOARD_COVERAGE = 0.70
    _TEXT_GUIDED_MAX_BOARD_COVERAGE = 1.05
    _MIN_SPACING_SAMPLES = 2
    _PLACEHOLDER_CHAR = "?"

    # ------------------------------------------------------------------
    # BoardReaderBase abstract hooks
    # ------------------------------------------------------------------

    def _decode_image(self, screenshot: bytes) -> object:
        """Decode screenshot bytes into a NumPy BGR image array via OpenCV."""
        cv2_module, np_module = self._load_cv_modules()
        cv2_any = cast("Any", cv2_module)
        np_any = cast("Any", np_module)
        buffer = np_any.frombuffer(screenshot, dtype=np_any.uint8)
        image = cv2_any.imdecode(buffer, cv2_any.IMREAD_COLOR)
        if image is None:
            msg = "screenshot is not a valid image payload"
            raise ValueError(msg)
        return image

    def _estimate_board_rect(self, image: object) -> tuple[int, int, int, int]:
        """Detect board bounding rectangle using Canny edges and contour scoring."""
        cv2_module, _np_module = self._load_cv_modules()
        return self._estimate_board_rect_cv(image, cv2_module)

    def _extract_cell_states(self, image: object) -> CellStateGrid:
        """Sample colors at each cell center to determine highlight states."""
        board_rect = self._estimate_board_rect(image)
        cell_centers = self._compute_cell_centers(board_rect)
        return self._compute_cell_states(image, cell_centers)

    def _extract_cell_centers(self, image: object) -> list[list[PixelCoord]]:
        """Compute pixel centers for every board cell."""
        board_rect = self._estimate_board_rect(image)
        return self._compute_cell_centers(board_rect)

    def _extract_board_rows(self, image: object) -> list[str]:
        """Run Tesseract OCR to extract board letter rows from the image."""
        board_rect = self._estimate_board_rect(image)
        return self._ocr_board(image, board_rect)

    # ------------------------------------------------------------------
    # Grid geometry helpers
    # ------------------------------------------------------------------

    def _default_grid_geometry(self, board_rect: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        board_x, board_y, board_width, board_height = board_rect
        return (float(board_x), float(board_y), board_width / self._cols, board_height / self._rows)

    def _compute_cell_centers(self, board_rect: tuple[int, int, int, int]) -> list[list[PixelCoord]]:
        """Compute per-cell center pixels from board geometry."""
        return self._compute_cell_centers_from_geometry(self._default_grid_geometry(board_rect))

    def _compute_cell_centers_from_geometry(
        self,
        grid_geometry: tuple[float, float, float, float],
    ) -> list[list[PixelCoord]]:
        x, y, cell_width, cell_height = grid_geometry
        centers: list[list[PixelCoord]] = []
        for row in range(self._rows):
            row_centers: list[PixelCoord] = []
            for col in range(self._cols):
                center_x = int(x + (col + 0.5) * cell_width)
                center_y = int(y + (row + 0.5) * cell_height)
                row_centers.append((center_x, center_y))
            centers.append(row_centers)
        return centers

    @staticmethod
    def _selected_coords_from_states(cell_states: CellStateGrid) -> set[BoardCoord]:
        selected_coords: set[BoardCoord] = set()
        for row_idx, row_states in enumerate(cell_states):
            for col_idx, state in enumerate(row_states):
                if state != Highlight.NONE:
                    selected_coords.add((row_idx, col_idx))
        return selected_coords

    @staticmethod
    def _load_ocr_module() -> object:
        """Import OCR dependency module.

        Returns:
            Imported `tesserocr` module.

        Raises:
            NotImplementedError: If `tesserocr` is not installed.

        """
        if not HAS_EXTRAS:  # pragma: no cover - env dependent
            msg = "OpenCV reader requires device extras"
            raise NotImplementedError(msg)
        return cast("Any", tesserocr)

    @staticmethod
    def _load_pil() -> object:
        """Import Pillow image module for OCR conversion.

        Returns:
            Imported `PIL.Image` module.

        Raises:
            NotImplementedError: If Pillow is not installed.

        """
        if not HAS_EXTRAS:  # pragma: no cover - env dependent
            msg = "OpenCV reader requires device extras"
            raise NotImplementedError(msg)
        return cast("Any", PILImage)

    def _ocr_board(self, image: object, board_rect: tuple[int, int, int, int]) -> list[str]:
        """Run character OCR across all board cells.

        Args:
            image: Decoded screenshot image array.
            board_rect: Board rectangle as `(x, y, width, height)`.

        Returns:
            OCR-extracted board rows.

        """
        tesserocr = cast("Any", self._load_ocr_module())
        pil_image_module = cast("Any", self._load_pil())
        cv2_any = cast("Any", cv2)
        np_any = cast("Any", np)
        psm_single_char = tesserocr.PSM.SINGLE_CHAR
        psm_single_word = getattr(tesserocr.PSM, "SINGLE_WORD", psm_single_char)
        psm_single_line = getattr(tesserocr.PSM, "SINGLE_LINE", psm_single_word)

        image_array = cast("Any", image)
        grid_geometry = self._resolve_grid_geometry(image_array, board_rect, cv2_any, np_any)

        with tesserocr.PyTessBaseAPI(psm=psm_single_line, lang="eng") as api:
            api.SetVariable("tessedit_char_whitelist", self._OCR_CHAR_WHITELIST)
            api.SetVariable("load_system_dawg", "0")
            api.SetVariable("load_freq_dawg", "0")

            board_rows = self._ocr_board_by_rows(image_array, grid_geometry, api, pil_image_module, cv2_any)
            if board_rows is not None:
                return board_rows

        with tesserocr.PyTessBaseAPI(psm=psm_single_word, lang="eng") as api:
            api.SetVariable("tessedit_char_whitelist", self._OCR_CHAR_WHITELIST)
            api.SetVariable("load_system_dawg", "0")
            api.SetVariable("load_freq_dawg", "0")

            rows: list[str] = []
            for row in range(self._rows):
                row_letters = []
                for col in range(self._cols):
                    letter = self._ocr_cell_with_fallback(
                        image_array,
                        grid_geometry,
                        (row, col),
                        api,
                        pil_image_module,
                    )
                    row_letters.append(letter)
                rows.append("".join(row_letters))

        return rows

    def _ocr_board_by_rows(  # noqa: C901
        self,
        image: object,
        grid_geometry: tuple[float, float, float, float],
        api: object,
        pil_image_module: object,
        cv2_module: object,
    ) -> list[str] | None:
        board_x, board_y, cell_width, cell_height = grid_geometry
        x = int(board_x)
        y = int(board_y)
        width = max(1, int(cell_width * self._cols))
        height = max(1, int(cell_height * self._rows))

        img_array = cast("Any", image)
        board_bgr = img_array[y : y + height, x : x + width]
        if board_bgr.size == 0:
            return None

        cv2_any = cast("Any", cv2_module)
        pil_mod = cast("Any", pil_image_module)
        api_obj = cast("Any", api)
        row_height = cell_height
        parsed_rows: list[str] = []

        for row_idx in range(self._rows):
            y0 = int(row_idx * row_height)
            y1 = int((row_idx + 1) * row_height)
            row_crop = board_bgr[y0:y1, :]
            if row_crop.size == 0:
                return None

            row_text: str | None = None
            for inset_ratio in self._OCR_BOARD_VARIANT_INSETS:
                crop = row_crop
                if inset_ratio > 0:
                    inset_x = int(width * inset_ratio)
                    inset_y = int(row_crop.shape[0] * inset_ratio)
                    x0 = max(0, inset_x)
                    y0_local = max(0, inset_y)
                    x1 = max(x0 + 1, row_crop.shape[1] - inset_x)
                    y1_local = max(y0_local + 1, row_crop.shape[0] - inset_y)
                    crop = row_crop[y0_local:y1_local, x0:x1]
                    if crop.size == 0:
                        continue

                for variant in self._build_ocr_variants(crop, cv2_any):
                    variant_any = cast("Any", variant)
                    pil_img = pil_mod.fromarray(variant_any.astype("uint8"))
                    api_obj.SetImage(pil_img)
                    text = self._normalize_ocr_text(api_obj.GetUTF8Text())
                    confidence = self._ocr_confidence(api_obj)
                    letters = "".join(ch.lower() for ch in text if ch.isalpha())
                    if len(letters) >= self._cols and confidence >= self._OCR_MIN_GOOD_CONFIDENCE:
                        row_text = letters[: self._cols]
                        break

                if row_text is not None:
                    break

            if row_text is None:
                return None

            parsed_rows.append(row_text)

        return parsed_rows

    def _parse_board_text(self, text: str) -> list[str] | None:
        lines = ["".join(ch.lower() for ch in line if ch.isalpha()) for line in text.splitlines()]
        lines = [line for line in lines if line]

        if len(lines) >= self._rows:
            trimmed: list[str] = []
            for line in lines[: self._rows]:
                if len(line) < self._cols:
                    return None
                trimmed.append(line[: self._cols])
            return trimmed

        letters = "".join(ch.lower() for ch in text if ch.isalpha())
        expected_len = self._rows * self._cols
        if len(letters) < expected_len:
            return None

        letters = letters[:expected_len]
        return [letters[row * self._cols : (row + 1) * self._cols] for row in range(self._rows)]

    def _ocr_cell_with_fallback(
        self,
        image: object,
        grid_geometry: tuple[float, float, float, float],
        coord: tuple[int, int],
        api: object,
        pil_image_module: object,
    ) -> str:
        row, col = coord
        best_letter = self._PLACEHOLDER_CHAR
        best_confidence = -1

        for inset_ratio in self._OCR_FALLBACK_INSET_RATIOS:
            cell_rect = self._cell_rect_from_geometry(grid_geometry, row, col, inset_ratio)
            letter, confidence = self._ocr_cell(image, cell_rect, api, pil_image_module)

            if letter != self._PLACEHOLDER_CHAR and confidence >= best_confidence:
                best_letter = letter
                best_confidence = confidence
                if confidence >= self._OCR_MIN_GOOD_CONFIDENCE:
                    break

        return best_letter

    @staticmethod
    def _ocr_cell(
        image: object,
        cell_rect: tuple[int, int, int, int],
        api: object,
        pil_image_module: object,
        cv2_module: object | None = None,
    ) -> tuple[str, int]:
        """OCR a single cell region into one lowercase character.

        Args:
            image: Decoded screenshot image array.
            cell_rect: Cell rectangle `(x0, y0, x1, y1)`.
            api: Active tesserocr API instance.
            pil_image_module: Imported Pillow image module.
            cv2_module: Optional imported OpenCV module override.

        Returns:
            Tuple of normalized character and OCR confidence.

        """
        x0, y0, x1, y1 = cell_rect
        img_array = cast("Any", image)
        cell_bgr = img_array[y0:y1, x0:x1]
        if cell_bgr.size == 0:
            return ("", -1)

        cv2_any = cast("Any", cv2_module if cv2_module is not None else cv2)
        pil_mod = cast("Any", pil_image_module)

        api_obj = cast("Any", api)
        best_letter = "?"
        best_confidence = -1

        for variant in BoardReaderTesseractOpenCv._build_ocr_variants(cell_bgr, cv2_any):
            variant_any = cast("Any", variant)
            pil_img = pil_mod.fromarray(variant_any.astype("uint8"))
            api_obj.SetImage(pil_img)

            text = BoardReaderTesseractOpenCv._normalize_ocr_text(api_obj.GetUTF8Text())
            confidence = BoardReaderTesseractOpenCv._ocr_confidence(api_obj)
            letter = BoardReaderTesseractOpenCv._extract_alpha_char(text)

            if letter != "?" and confidence >= best_confidence:
                best_letter = letter
                best_confidence = confidence
                if confidence >= BoardReaderTesseractOpenCv._OCR_MIN_GOOD_CONFIDENCE:
                    break

        return (best_letter, best_confidence)

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_alpha_char(text: str) -> str:
        for char in text:
            if char.isalpha():
                return char.lower()
        return "?"

    @staticmethod
    def _build_ocr_variants(cell_bgr: object, cv2_module: object) -> list[object]:
        cv2_any = cast("Any", cv2_module)
        cell_arr = cast("Any", cell_bgr)

        cell_rgb = cell_arr[:, :, ::-1]
        gray = cv2_any.cvtColor(cell_arr, cv2_any.COLOR_BGR2GRAY)
        _threshold_inv, binary_inv = cv2_any.threshold(gray, 0, 255, cv2_any.THRESH_BINARY_INV + cv2_any.THRESH_OTSU)
        _threshold, binary = cv2_any.threshold(gray, 0, 255, cv2_any.THRESH_BINARY + cv2_any.THRESH_OTSU)

        return [
            cell_rgb,
            gray,
            binary_inv,
            binary,
        ]

    @staticmethod
    def _ocr_confidence(api: object) -> int:
        api_obj = cast("Any", api)
        confidence = api_obj.MeanTextConf()
        return confidence if isinstance(confidence, int) else -1

    def _resolve_grid_geometry(
        self,
        image: object,
        board_rect: tuple[int, int, int, int],
        cv2_module: object,
        np_module: object,
    ) -> tuple[float, float, float, float]:
        if board_rect == self._fallback_board_rect(cast("Any", image).shape[:2]):
            return self._default_grid_geometry(board_rect)

        refined = self._refine_grid_geometry_from_text(image, board_rect, cv2_module, np_module)
        if refined is not None:
            return refined
        return self._default_grid_geometry(board_rect)

    def _refine_grid_geometry_from_text(  # noqa: C901
        self,
        image: object,
        board_rect: tuple[int, int, int, int],
        cv2_module: object,
        np_module: object,
    ) -> tuple[float, float, float, float] | None:
        try:
            cv2_any = cast("Any", cv2_module)
            np_any = cast("Any", np_module)
            image_array = cast("Any", image)
            board_x, board_y, board_width, board_height = board_rect
            board_bgr = image_array[board_y : board_y + board_height, board_x : board_x + board_width]
            if getattr(board_bgr, "size", 0) == 0:
                return None

            default_cell_width = board_width / self._cols
            default_cell_height = board_height / self._rows
            best_geometry: tuple[float, float, float, float] | None = None
            best_score = float("-inf")

            for mask in self._build_text_guided_masks(board_bgr, cv2_any, np_any):
                mask_any = cast("Any", mask)
                fill_ratio = float(np_any.count_nonzero(mask_any)) / float(mask_any.size)
                if not (self._TEXT_GUIDED_MIN_MASK_FILL_RATIO <= fill_ratio <= self._TEXT_GUIDED_MAX_MASK_FILL_RATIO):
                    continue

                centers = self._extract_text_component_centers(mask, default_cell_width, default_cell_height, cv2_any)
                if len(centers) < self._TEXT_GUIDED_MIN_COMPONENTS:
                    continue

                x_positions = self._cluster_axis_positions(
                    [center_x for center_x, _ in centers],
                    self._cols,
                    cv2_any,
                    np_any,
                )
                y_positions = self._cluster_axis_positions(
                    [center_y for _, center_y in centers],
                    self._rows,
                    cv2_any,
                    np_any,
                )
                if x_positions is None or y_positions is None:
                    continue

                cell_width = self._median_spacing(x_positions, np_any)
                cell_height = self._median_spacing(y_positions, np_any)
                if cell_width is None or cell_height is None:
                    continue

                refined_width = cell_width * self._cols
                refined_height = cell_height * self._rows
                width_coverage = refined_width / board_width
                height_coverage = refined_height / board_height
                if not (
                    self._TEXT_GUIDED_MIN_BOARD_COVERAGE <= width_coverage <= self._TEXT_GUIDED_MAX_BOARD_COVERAGE
                    and self._TEXT_GUIDED_MIN_BOARD_COVERAGE <= height_coverage <= self._TEXT_GUIDED_MAX_BOARD_COVERAGE
                ):
                    continue

                origin_x = board_x + x_positions[0] - cell_width / 2
                origin_y = board_y + y_positions[0] - cell_height / 2
                max_origin_x = board_x + board_width - refined_width
                max_origin_y = board_y + board_height - refined_height
                origin_x = min(max(float(board_x), origin_x), float(max_origin_x))
                origin_y = min(max(float(board_y), origin_y), float(max_origin_y))

                x_cv = self._spacing_coefficient_of_variation(x_positions, np_any)
                y_cv = self._spacing_coefficient_of_variation(y_positions, np_any)
                score = (
                    len(centers) - (x_cv + y_cv) * 25 - abs(1.0 - width_coverage) * 10 - abs(1.0 - height_coverage) * 10
                )
                if score > best_score:
                    best_score = score
                    best_geometry = (origin_x, origin_y, cell_width, cell_height)

        except Exception:  # noqa: BLE001
            return None
        else:
            return best_geometry

    def _build_text_guided_masks(self, board_bgr: object, cv2_module: object, np_module: object) -> list[object]:
        cv2_any = cast("Any", cv2_module)
        np_any = cast("Any", np_module)
        board_arr = cast("Any", board_bgr)

        gray = cv2_any.cvtColor(board_arr, cv2_any.COLOR_BGR2GRAY)
        blurred = cv2_any.GaussianBlur(gray, (5, 5), 0)
        _threshold_dark, mask_dark = cv2_any.threshold(blurred, 0, 255, cv2_any.THRESH_BINARY_INV + cv2_any.THRESH_OTSU)
        _threshold_light, mask_light = cv2_any.threshold(blurred, 0, 255, cv2_any.THRESH_BINARY + cv2_any.THRESH_OTSU)
        kernel = np_any.ones((3, 3), dtype=np_any.uint8)

        masks: list[object] = []
        for mask in (mask_dark, mask_light):
            opened = cv2_any.morphologyEx(mask, cv2_any.MORPH_OPEN, kernel)
            dilated = cv2_any.dilate(opened, kernel, iterations=1)
            masks.extend([opened, dilated])
        return masks

    def _extract_text_component_centers(
        self,
        mask: object,
        cell_width: float,
        cell_height: float,
        cv2_module: object,
    ) -> list[tuple[float, float]]:
        cv2_any = cast("Any", cv2_module)
        component_count, _labels, stats, centroids = cv2_any.connectedComponentsWithStats(mask, connectivity=8)
        cell_area = cell_width * cell_height
        min_area = cell_area * self._TEXT_GUIDED_MIN_COMPONENT_AREA_RATIO
        max_area = cell_area * self._TEXT_GUIDED_MAX_COMPONENT_AREA_RATIO
        min_width = cell_width * self._TEXT_GUIDED_MIN_COMPONENT_WIDTH_RATIO
        max_width = cell_width * self._TEXT_GUIDED_MAX_COMPONENT_WIDTH_RATIO
        min_height = cell_height * self._TEXT_GUIDED_MIN_COMPONENT_HEIGHT_RATIO
        max_height = cell_height * self._TEXT_GUIDED_MAX_COMPONENT_HEIGHT_RATIO
        mask_height, mask_width = cast("Any", mask).shape[:2]

        centers: list[tuple[float, float]] = []
        for index in range(1, component_count):
            x = int(stats[index, cv2_any.CC_STAT_LEFT])
            y = int(stats[index, cv2_any.CC_STAT_TOP])
            width = int(stats[index, cv2_any.CC_STAT_WIDTH])
            height = int(stats[index, cv2_any.CC_STAT_HEIGHT])
            area = int(stats[index, cv2_any.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            if width < min_width or width > max_width:
                continue
            if height < min_height or height > max_height:
                continue
            if x <= 0 or y <= 0 or x + width >= mask_width - 1 or y + height >= mask_height - 1:
                continue

            center_x = float(centroids[index][0])
            center_y = float(centroids[index][1])
            centers.append((center_x, center_y))

        return centers

    def _cluster_axis_positions(
        self,
        values: list[float],
        cluster_count: int,
        cv2_module: object,
        np_module: object,
    ) -> list[float] | None:
        if len(values) < cluster_count:
            return None

        cv2_any = cast("Any", cv2_module)
        np_any = cast("Any", np_module)
        samples = np_any.array(values, dtype=np_any.float32).reshape(-1, 1)
        criteria = (cv2_any.TERM_CRITERIA_EPS + cv2_any.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _compactness, labels, centers = cv2_any.kmeans(
            samples,
            cluster_count,
            None,
            criteria,
            10,
            cv2_any.KMEANS_PP_CENTERS,
        )

        positions = sorted(float(center[0]) for center in centers)
        if len(positions) != cluster_count:
            return None
        if self._median_spacing(positions, np_any) is None:
            return None
        if self._spacing_coefficient_of_variation(positions, np_any) > self._TEXT_GUIDED_MAX_SPACING_CV:
            return None

        counts = [0] * cluster_count
        for label in labels.reshape(-1):
            counts[int(label)] += 1
        if any(count == 0 for count in counts):
            return None

        return positions

    @staticmethod
    def _median_spacing(positions: list[float], np_module: object) -> float | None:
        if len(positions) < BoardReaderTesseractOpenCv._MIN_SPACING_SAMPLES:
            return None

        np_any = cast("Any", np_module)
        diffs = np_any.diff(np_any.array(positions, dtype=np_any.float32))
        if diffs.size == 0 or float(np_any.min(diffs)) <= 0:
            return None
        return float(np_any.median(diffs))

    @staticmethod
    def _spacing_coefficient_of_variation(positions: list[float], np_module: object) -> float:
        spacing = BoardReaderTesseractOpenCv._median_spacing(positions, np_module)
        if spacing is None or spacing <= 0:
            return float("inf")

        np_any = cast("Any", np_module)
        diffs = np_any.diff(np_any.array(positions, dtype=np_any.float32))
        return float(np_any.std(diffs) / spacing)

    def _cell_rect_from_geometry(
        self,
        grid_geometry: tuple[float, float, float, float],
        row: int,
        col: int,
        inset_ratio: float,
    ) -> tuple[int, int, int, int]:
        board_x, board_y, cell_w, cell_h = grid_geometry
        inset_x = cell_w * inset_ratio
        inset_y = cell_h * inset_ratio
        return (
            int(board_x + col * cell_w + inset_x),
            int(board_y + row * cell_h + inset_y),
            int(board_x + (col + 1) * cell_w - inset_x),
            int(board_y + (row + 1) * cell_h - inset_y),
        )

    def _cell_rect(
        self,
        board_rect: tuple[int, int, int, int],
        row: int,
        col: int,
        inset_ratio: float,
    ) -> tuple[int, int, int, int]:
        return self._cell_rect_from_geometry(self._default_grid_geometry(board_rect), row, col, inset_ratio)

    def _get_cell_state(self, bgr: tuple[int, int, int]) -> Highlight:
        """Determine cell state from BGR color."""
        hue, saturation, value = self._bgr_to_hsv_unit(bgr)
        is_yellow = (
            self._YELLOW_HUE_LOW <= hue <= self._YELLOW_HUE_HIGH
            and saturation >= self._YELLOW_MIN_SAT
            and value >= self._YELLOW_MIN_VAL
        )
        is_blue = (
            self._BLUE_HUE_LOW <= hue <= self._BLUE_HUE_HIGH
            and saturation >= self._BLUE_MIN_SAT
            and value >= self._BLUE_MIN_VAL
        )
        if is_yellow:
            return Highlight.SPANGRAM
        if is_blue:
            return Highlight.WORD
        return Highlight.NONE

    def _compute_cell_states(self, image: object, cell_centers: list[list[PixelCoord]]) -> CellStateGrid:
        """Compute cell states from image colors at cell centers."""
        cell_colors_bgr = self._sample_cell_colors(image, cell_centers)
        return [[self._get_cell_state(bgr) for bgr in row_colors] for row_colors in cell_colors_bgr]

    def _estimate_board_rect_cv(self, image: object, cv2_module: object) -> tuple[int, int, int, int]:
        """Estimate board rectangle from contours with geometry heuristics.

        Args:
            image: Decoded screenshot image array.
            cv2_module: Imported OpenCV module.

        Returns:
            Board rectangle as `(x, y, width, height)`.

        """
        cv2 = cast("Any", cv2_module)
        image_array = cast("Any", image)
        img_height, img_width = image_array.shape[:2]
        min_candidate_area = img_width * img_height * self._MIN_BOARD_AREA_FRACTION
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        expected_ratio = self._cols / self._rows
        best_rect: tuple[int, int, int, int] | None = None
        best_score = float("-inf")

        image_center_x = img_width / 2
        image_center_y = img_height / 2

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area <= 0 or area < min_candidate_area:
                continue

            ratio = width / height if height > 0 else 0.0
            ratio_error = abs(ratio - expected_ratio)

            center_x = x + width / 2
            center_y = y + height / 2
            center_distance = abs(center_x - image_center_x) + abs(center_y - image_center_y)

            score = area - (ratio_error * 150_000) - (center_distance * 40)
            if score > best_score:
                best_score = score
                best_rect = (x, y, width, height)

        if best_rect is not None:
            return best_rect

        return self._fallback_board_rect(cast("Any", image).shape[:2])

    def _fallback_board_rect(self, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        img_height, img_width = image_shape
        fallback_x = int(img_width * self._FALLBACK_X_RATIO)
        fallback_y = int(img_height * self._FALLBACK_Y_RATIO)
        fallback_width = int(img_width * self._FALLBACK_WIDTH_RATIO)
        fallback_height = int(img_height * self._FALLBACK_HEIGHT_RATIO)
        return (fallback_x, fallback_y, fallback_width, fallback_height)

    @staticmethod
    def _load_cv_modules() -> tuple[object, object]:
        """Import OpenCV and NumPy dependencies.

        Returns:
            Tuple of ``(cv2_module, numpy_module)``.

        Raises:
            NotImplementedError: If OpenCV or NumPy is not installed.

        """
        if not HAS_EXTRAS:  # pragma: no cover - env dependent
            msg = "OpenCV reader requires device extras"
            raise NotImplementedError(msg)

        return cast("Any", cv2), cast("Any", np)

    @staticmethod
    def _sample_cell_colors(image: object, cell_centers: list[list[PixelCoord]]) -> list[list[tuple[int, int, int]]]:
        """Sample average BGR values around each cell center.

        Args:
            image: Decoded screenshot image array.
            cell_centers: Mapping of board coordinates to pixel centers.

        Returns:
            Per-cell sampled mean BGR colors aligned to board coordinates.

        """
        image_array = cast("Any", image)
        image_height, image_width = image_array.shape[:2]
        radius = max(2, min(image_height, image_width) // 180)
        offset = max(3, radius * 3)
        sample_offsets = ((-offset, 0), (offset, 0), (0, -offset), (0, offset))

        colors: list[list[tuple[int, int, int]]] = []
        for row_centers in cell_centers:
            row_colors: list[tuple[int, int, int]] = []
            for center_x, center_y in row_centers:
                sample_patches: list[object] = []
                for delta_x, delta_y in sample_offsets:
                    sample_x = min(max(center_x + delta_x, 0), image_width - 1)
                    sample_y = min(max(center_y + delta_y, 0), image_height - 1)

                    x_min = max(0, sample_x - radius)
                    x_max = min(image_width, sample_x + radius + 1)
                    y_min = max(0, sample_y - radius)
                    y_max = min(image_height, sample_y + radius + 1)

                    patch = image_array[y_min:y_max, x_min:x_max]
                    if patch.size != 0:
                        sample_patches.append(patch)

                if not sample_patches:
                    row_colors.append((0, 0, 0))
                    continue

                channel_sums = [0.0, 0.0, 0.0]
                total_pixels = 0
                for patch in sample_patches:
                    patch_any = cast("Any", patch)
                    flat_patch = patch_any.reshape(-1, 3)
                    total_pixels += len(flat_patch)
                    channel_sum = flat_patch.sum(axis=0)
                    channel_sums[0] += float(channel_sum[0])
                    channel_sums[1] += float(channel_sum[1])
                    channel_sums[2] += float(channel_sum[2])

                mean_bgr = (
                    channel_sums[0] / total_pixels,
                    channel_sums[1] / total_pixels,
                    channel_sums[2] / total_pixels,
                )
                row_colors.append((int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])))
            colors.append(row_colors)

        return colors

    @staticmethod
    def _mean_hsv(colors_bgr: list[tuple[int, int, int]]) -> tuple[float, float, float]:
        """Compute mean HSV values from BGR color samples.

        Args:
            colors_bgr: BGR color tuples.

        Returns:
            Mean `(hue, saturation, value)` tuple.

        """
        hues: list[float] = []
        sats: list[float] = []
        vals: list[float] = []

        for blue, green, red in colors_bgr:
            hue, sat, val = BoardReaderTesseractOpenCv._bgr_to_hsv_unit((blue, green, red))
            hues.append(hue)
            sats.append(sat)
            vals.append(val)

        count = len(colors_bgr)
        return (sum(hues) / count, sum(sats) / count, sum(vals) / count)

    @staticmethod
    def _bgr_to_hsv_unit(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
        blue, green, red = bgr
        return colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
