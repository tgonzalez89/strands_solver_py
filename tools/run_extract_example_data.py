#!/usr/bin/env python3
"""Run OCR board-row and cell-state extraction for `data/example*` directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader import Highlight
from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCV
from strands_solver.device.device_driver import DeviceDriver

if TYPE_CHECKING:
    from strands_solver.util.util import PixelCoord

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / ".debug"
TEMP_DEBUG_IMAGE = REPO_ROOT / "debug_board.png"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
STATE_SYMBOL_TO_HIGHLIGHT = {
    "N": Highlight.NONE,
    "W": Highlight.WORD,
    "S": Highlight.SPANGRAM,
}


def iter_example_dirs(data_dir: Path) -> list[Path]:
    """Return sorted example directories under the data folder."""
    return sorted(path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("example"))


def iter_board_images(input_dir: Path) -> list[Path]:
    """Return sorted board image paths, preferring synthetic.png then Screenshot*.png."""
    # Try synthetic.png first
    synth = input_dir / "synthetic.png"
    if synth.exists():
        return [synth]

    # Fall back to Screenshot images
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.name.startswith("Screenshot") and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_reference_board(example_dir: Path) -> list[str]:
    """Read expected board rows from `board.txt`."""
    board_path = example_dir / "board.txt"
    lines = [line.strip() for line in board_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def read_reference_cell_states(example_dir: Path, rows: int, cols: int) -> list[list[Highlight]] | None:
    """Read expected cell-state grid from cell_states.txt if present."""
    states_path = example_dir / "cell_states.txt"
    if not states_path.exists():
        return None

    raw_lines = [line.strip().upper() for line in states_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_lines) != rows:
        msg = f"Expected {rows} rows in {states_path}, found {len(raw_lines)}"
        raise ValueError(msg)

    parsed: list[list[Highlight]] = []
    for row_idx, line in enumerate(raw_lines):
        symbols = line.split() if " " in line else list(line)
        if len(symbols) != cols:
            msg = f"Expected {cols} columns in {states_path} row {row_idx + 1}, found {len(symbols)}"
            raise ValueError(msg)
        if any(symbol not in STATE_SYMBOL_TO_HIGHLIGHT for symbol in symbols):
            msg = f"Invalid symbol in {states_path} row {row_idx + 1}. Allowed: N, W, S"
            raise ValueError(msg)
        parsed.append([STATE_SYMBOL_TO_HIGHLIGHT[symbol] for symbol in symbols])
    return parsed


def read_reference_centers(example_dir: Path) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Read all cell centers from centers.txt if present.

    Expected line format: `row,col : x,y`.

    Returns:
        Mapping from (row, col) to (x, y) pixel center, or None if file missing.

    """
    centers_path = example_dir / "centers.txt"
    if not centers_path.exists():
        return None

    center_by_coord: dict[tuple[int, int], tuple[int, int]] = {}
    for line_number, raw_line in enumerate(centers_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        if ":" not in line:
            msg = f"Invalid line in {centers_path} at {line_number}: expected 'row,col : x,y'"
            raise ValueError(msg)

        board_coord_text, pixel_coord_text = [part.strip() for part in line.split(":", maxsplit=1)]
        try:
            row_text, col_text = [part.strip() for part in board_coord_text.split(",", maxsplit=1)]
            x_text, y_text = [part.strip() for part in pixel_coord_text.split(",", maxsplit=1)]
            board_coord = (int(row_text), int(col_text))
            pixel_coord = (int(x_text), int(y_text))
        except (ValueError, TypeError) as error:
            msg = f"Invalid line in {centers_path} at {line_number}: expected integers in 'row,col : x,y'"
            raise ValueError(msg) from error

        center_by_coord[board_coord] = pixel_coord

    return center_by_coord or None


class ReplayDriver(DeviceDriver):
    """Device driver that replays a prerecorded screenshot sequence."""

    def __init__(self, screenshots: list[bytes]) -> None:
        """Initialize with a list of screenshot bytes."""
        self._screenshots = screenshots
        self._index = 0

    def capture_screen(self) -> bytes:
        """Return the next screenshot; keep returning the last one after sequence ends."""
        if not self._screenshots:
            msg = "Screenshot sequence is empty"
            raise RuntimeError(msg)

        if self._index < len(self._screenshots):
            screenshot = self._screenshots[self._index]
            self._index += 1
            return screenshot

        return self._screenshots[-1]

    def tap(self, coord: PixelCoord) -> None:
        """Accept taps without side effects."""
        _ = coord

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        """Ignore path execution."""
        _ = pixel_path


def try_calibrate_and_validate(
    reader: BoardReaderTesseractOpenCV,
    example_dir: Path,
) -> tuple[bool, str]:
    """Attempt to calibrate reader from disk images and validate against centers.txt.

    Returns:
        Tuple of (success, log_message).

    """
    clear_path = example_dir / "clear.png"
    top_left_path = example_dir / "top_left.png"
    bottom_right_path = example_dir / "bottom_right.png"

    if not (clear_path.exists() and top_left_path.exists() and bottom_right_path.exists()):
        return True, ""  # No calibration frames available; skip silently

    # Load calibration frames
    clear_bytes = clear_path.read_bytes()
    top_left_bytes = top_left_path.read_bytes()
    bottom_right_bytes = bottom_right_path.read_bytes()

    # Run calibration via replay
    driver = ReplayDriver(screenshots=[clear_bytes, top_left_bytes, clear_bytes, bottom_right_bytes])
    try:
        reader.calibrate(driver, timeout_s=2.0, poll_interval_s=0.0)
    except (TimeoutError, ValueError, RuntimeError) as err:
        msg = f"Calibration failed: {err}"
        return False, msg

    # Try to validate against centers.txt
    rows = reader._rows  # noqa: SLF001
    cols = reader._cols  # noqa: SLF001
    reference_centers = read_reference_centers(example_dir)
    if reference_centers is None:
        return True, "Calibration succeeded (no centers.txt for validation)"

    # Compare calibrated corners vs reference
    tol = 3
    tl_ref = reference_centers[(0, 0)]
    br_ref = reference_centers[(rows - 1, cols - 1)]
    tl_calib = reader._top_left_cell_center  # noqa: SLF001
    br_calib = reader._bottom_right_cell_center  # noqa: SLF001

    tl_match = abs(tl_calib[0] - tl_ref[0]) <= tol and abs(tl_calib[1] - tl_ref[1]) <= tol
    br_match = abs(br_calib[0] - br_ref[0]) <= tol and abs(br_calib[1] - br_ref[1]) <= tol

    if tl_match and br_match:
        msg = f"Calibration validated: TL {tl_calib} vs {tl_ref}, BR {br_calib} vs {br_ref}"
        return True, msg

    msg = f"Calibration mismatch: TL {tl_calib} vs {tl_ref}, BR {br_calib} vs {br_ref}"
    return False, msg


def to_symbol(state: Highlight) -> str:
    """Map highlight enum value to output symbol."""
    if state == Highlight.WORD:
        return "W"
    if state == Highlight.SPANGRAM:
        return "S"
    return "N"


def format_grid(cell_states: list[list[Highlight]]) -> str:
    """Format a state grid with space-separated N/W/S symbols."""
    return "\n".join(" ".join(to_symbol(cell) for cell in row) for row in cell_states)


def format_board_log_entry(image_name: str, board_rows: list[str], reference_rows: list[str]) -> list[str]:
    """Build one OCR board log entry."""
    matched_rows = sum(1 for actual, expected in zip(board_rows, reference_rows, strict=False) if actual == expected)
    exact_match = board_rows == reference_rows

    rows = len(reference_rows)
    cols = len(reference_rows[0])
    total_letters = rows * cols
    identified_letters = 0
    correct_letters = 0
    for row_idx in range(rows):
        extracted_row = board_rows[row_idx] if row_idx < len(board_rows) else ""
        reference_row = reference_rows[row_idx]
        for col_idx in range(cols):
            extracted_char = extracted_row[col_idx] if col_idx < len(extracted_row) else "?"
            if extracted_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                identified_letters += 1
            if extracted_char == reference_row[col_idx]:
                correct_letters += 1

    complete_ocr_failure = identified_letters == 0

    lines = [image_name, "Extracted:"]
    lines.extend(board_rows or ["<no OCR output>"])
    lines.append("Reference:")
    lines.extend(reference_rows)
    lines.append(f"matched_rows={matched_rows}/{len(reference_rows)}")
    lines.append(f"identified_letters={identified_letters}/{total_letters}")
    lines.append(f"correct_letters={correct_letters}/{total_letters}")
    lines.append(f"complete_ocr_failure={'yes' if complete_ocr_failure else 'no'}")
    lines.append(f"exact_match={'yes' if exact_match else 'no'}")
    lines.append("")
    return lines


def score_board_rows(board_rows: list[str], reference_rows: list[str]) -> tuple[int, int, bool, bool]:
    """Compute OCR board-level score metrics."""
    rows = len(reference_rows)
    cols = len(reference_rows[0])
    total_letters = rows * cols
    identified_letters = 0

    for row_idx in range(rows):
        extracted_row = board_rows[row_idx] if row_idx < len(board_rows) else ""
        for col_idx in range(cols):
            extracted_char = extracted_row[col_idx] if col_idx < len(extracted_row) else "?"
            if extracted_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                identified_letters += 1

    exact_match = board_rows == reference_rows
    complete_failure = identified_letters == 0
    return identified_letters, total_letters, exact_match, complete_failure


def score_cell_states(
    extracted_states: list[list[Highlight]],
    expected_states: list[list[Highlight]],
) -> tuple[int, int, bool]:
    """Compute cell-state score metrics."""
    total_cells = len(expected_states) * len(expected_states[0]) if expected_states else 0
    matched_cells = 0
    for row_idx in range(len(expected_states)):
        for col_idx in range(len(expected_states[0])):
            if extracted_states[row_idx][col_idx] == expected_states[row_idx][col_idx]:
                matched_cells += 1
    return matched_cells, total_cells, matched_cells == total_cells


def copy_debug_image(output_path: Path) -> None:
    """Copy the temporary OCR debug image to its final destination."""
    if not TEMP_DEBUG_IMAGE.exists():
        msg = f"expected {TEMP_DEBUG_IMAGE} was not generated"
        raise RuntimeError(msg)

    shutil.copy2(TEMP_DEBUG_IMAGE, output_path)


@dataclass
class ImageExtractionResult:
    """Hold extraction outputs and errors for one screenshot."""

    board_rows: list[str]
    cell_states: list[list[Highlight]] | None
    board_error: Exception | None
    cell_error: Exception | None


def extract_image_data(
    reader: BoardReaderTesseractOpenCV,
    image_path: Path,
    example_name: str,
) -> ImageExtractionResult:
    """Extract board rows and cell states for one screenshot path."""
    image: object | None = None
    board_error: Exception | None = None
    cell_error: Exception | None = None
    board_rows: list[str] = []
    cell_states: list[list[Highlight]] | None = None

    try:
        screenshot = image_path.read_bytes()
        image = reader._decode_image(screenshot)  # noqa: SLF001
    except (OSError, ValueError, RuntimeError) as error:
        board_error = error
        cell_error = error

    if image is None:
        return ImageExtractionResult(
            board_rows=board_rows,
            cell_states=cell_states,
            board_error=board_error,
            cell_error=cell_error,
        )

    try:
        cell_states = reader._extract_cell_states(image)  # noqa: SLF001
    except (OSError, ValueError, RuntimeError) as error:
        cell_error = error

    try:
        board_rows = reader._extract_board_rows(image)  # noqa: SLF001
        debug_image_path = OUTPUT_DIR / f"{example_name}_{image_path.stem}_debug_board.png"
        copy_debug_image(debug_image_path)
    except (OSError, ValueError, RuntimeError) as error:
        board_error = error

    return ImageExtractionResult(
        board_rows=board_rows,
        cell_states=cell_states,
        board_error=board_error,
        cell_error=cell_error,
    )


def append_board_result(
    board_log_lines: list[str],
    image_path: Path,
    extraction: ImageExtractionResult,
    reference_rows: list[str],
) -> tuple[str, str]:
    """Append OCR board extraction output and return status summary."""
    if extraction.board_error is None:
        board_log_lines.extend(format_board_log_entry(image_path.name, extraction.board_rows, reference_rows))
        identified_letters, total_letters, ocr_exact_match, complete_failure = score_board_rows(
            extraction.board_rows,
            reference_rows,
        )
        ocr_status = "PASS" if ocr_exact_match else "FAIL"
        if complete_failure:
            ocr_score_text = f"letters=0/{total_letters} (complete OCR failure)"
        else:
            ocr_score_text = f"letters={identified_letters}/{total_letters}"
        return ocr_status, ocr_score_text

    board_log_lines.append(image_path.name)
    board_log_lines.append(f"ERROR: {extraction.board_error}")
    board_log_lines.append("")
    return "ERROR", str(extraction.board_error)


def append_cell_result(
    cell_log_lines: list[str],
    image_path: Path,
    extraction: ImageExtractionResult,
    expected_cell_states: list[list[Highlight]] | None,
) -> tuple[str, str]:
    """Append cell-state extraction output and return status summary."""
    cell_log_lines.append(image_path.name)
    if extraction.cell_error is None and extraction.cell_states is not None:
        cell_log_lines.append(format_grid(extraction.cell_states))
        if expected_cell_states is not None:
            matched_cells, total_cells, cell_exact_match = score_cell_states(
                extraction.cell_states,
                expected_cell_states,
            )
            cell_log_lines.append("Expected:")
            cell_log_lines.append(format_grid(expected_cell_states))
            cell_log_lines.append(f"matched_cells={matched_cells}/{total_cells}")
            cell_log_lines.append(f"exact_match={'yes' if cell_exact_match else 'no'}")
            cell_log_lines.append("")
            cell_status = "PASS" if cell_exact_match else "FAIL"
            cell_score_text = f"cells={matched_cells}/{total_cells}"
            return cell_status, cell_score_text

        cell_log_lines.append("Expected: <not provided>")
        cell_log_lines.append("")
        return "N/A", "cells=<no reference>"

    cell_log_lines.append(f"ERROR: {extraction.cell_error}")
    cell_log_lines.append("")
    return "ERROR", str(extraction.cell_error)


def process_example_dir(example_dir: Path) -> tuple[int, int, int, int]:
    """Process one example directory.

    Returns:
        Tuple of (successful_extractions, total_extractions, successful_calibrations, total_calibration_attempts).

    """
    example_name = example_dir.name
    image_paths = iter_board_images(example_dir)
    if not image_paths:
        print(f"SKIP {example_name}: no board images found")
        return 0, 0, 0, 0

    try:
        reference_rows = read_reference_board(example_dir)
        rows, cols = len(reference_rows), len(reference_rows[0])
    except (OSError, ValueError) as err:
        print(f"SKIP {example_name}: {err}")
        return 0, 0, 0, 0

    expected_cell_states = read_reference_cell_states(example_dir, rows, cols)
    reader = BoardReaderTesseractOpenCV(rows=rows, cols=cols)

    # Try calibration if frames exist
    calib_success, calib_msg = try_calibrate_and_validate(reader, example_dir)
    calib_attempts = 1 if (example_dir / "clear.png").exists() else 0
    calib_count = 1 if calib_success and calib_attempts == 1 else 0

    if calib_msg:
        if calib_success:
            print(f"  {example_name}: {calib_msg}")
        else:
            print(f"  {example_name}: CALIB FAIL: {calib_msg}")

    board_log_path = OUTPUT_DIR / f"{example_name}_board_rows.log"
    cell_log_path = OUTPUT_DIR / f"{example_name}_cell_states.log"
    calib_log_path = OUTPUT_DIR / f"{example_name}_calibration.log"
    board_log_lines: list[str] = []
    cell_log_lines: list[str] = []
    calib_log_lines: list[str] = []

    # Write calibration log if frames were present
    if calib_attempts > 0:
        calib_log_lines.append(f"Calibration attempt for {example_name}")
        calib_log_lines.append(f"Status: {'PASS' if calib_success else 'FAIL'}")
        calib_log_lines.append(f"Message: {calib_msg}")
        calib_log_lines.append("")

    success_count = 0
    for image_path in image_paths:
        extraction = extract_image_data(reader, image_path, example_name)

        ocr_status, ocr_score_text = append_board_result(
            board_log_lines,
            image_path,
            extraction,
            reference_rows,
        )

        cell_status, cell_score_text = append_cell_result(
            cell_log_lines,
            image_path,
            extraction,
            expected_cell_states,
        )

        print(
            f"{ocr_status} OCR {example_name}/{image_path.name} {ocr_score_text}; {cell_status} CELL {cell_score_text}"
        )

        if extraction.board_error is None and extraction.cell_error is None:
            success_count += 1

    board_log_path.write_text("\n".join(board_log_lines), encoding="utf-8")
    cell_log_path.write_text("\n".join(cell_log_lines), encoding="utf-8")
    if calib_log_lines:
        calib_log_path.write_text("\n".join(calib_log_lines), encoding="utf-8")

    return success_count, len(image_paths), calib_count, calib_attempts


def main() -> int:
    """Process all example directories under `data/`."""
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"Input directory not found: {DATA_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    example_dirs = iter_example_dirs(DATA_DIR)
    if not example_dirs:
        print(f"No example directories found in {DATA_DIR}")
        return 1

    total_success = 0
    total_images = 0
    total_calib_success = 0
    total_calib_attempts = 0

    try:
        for example_dir in example_dirs:
            success_count, image_count, calib_success, calib_attempts = process_example_dir(example_dir)
            total_success += success_count
            total_images += image_count
            total_calib_success += calib_success
            total_calib_attempts += calib_attempts
    finally:
        if TEMP_DEBUG_IMAGE.exists():
            TEMP_DEBUG_IMAGE.unlink()

    print(f"Processed {total_success}/{total_images} images across {len(example_dirs)} example directories")
    if total_calib_attempts > 0:
        print(f"Calibration: {total_calib_success}/{total_calib_attempts} examples")
    calib_all_pass = total_calib_attempts in (0, total_calib_success)
    return 0 if total_success == total_images and calib_all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
