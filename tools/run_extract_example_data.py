#!/usr/bin/env python3
"""Run OCR board-row and cell-state extraction for `data/example*` directories."""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader import Highlight
from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCV
from strands_solver.device.device_driver import DeviceDriver
from strands_solver.util.util import CENTER_TOLERANCE_PX, CIRCLE_DIAMETER_TOLERANCE_PX

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print internal calibration and OCR output that is suppressed by default.",
    )
    return parser.parse_args()


def maybe_suppress_stdout(*, verbose: bool) -> contextlib.AbstractContextManager[io.StringIO | None]:
    """Return a context manager that suppresses stdout unless verbose mode is enabled."""
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


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


def read_reference_circle_diameter(example_dir: Path) -> int | None:
    """Read ground-truth circle diameter from circle_diameter.txt if present.

    Returns:
        Circle diameter in pixels, or None if file missing.

    """
    diameter_path = example_dir / "circle_diameter.txt"
    if not diameter_path.exists():
        return None

    text = diameter_path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    try:
        return int(text)
    except ValueError as error:
        msg = f"Invalid value in {diameter_path}: expected integer diameter, got '{text}'"
        raise ValueError(msg) from error


def validate_circle_diameter(
    estimated: int | None,
    reference: int | None,
) -> tuple[str, str]:
    """Compare estimated and reference circle diameters.

    Returns:
        Tuple (status, message) where status is PASS/FAIL/NA.

    """
    if reference is None:
        return "NA", "No circle_diameter.txt reference"
    if estimated is None:
        return "NA", "No estimated circle diameter from calibration"

    diff = abs(estimated - reference)
    if diff <= CIRCLE_DIAMETER_TOLERANCE_PX:
        msg = (
            f"Diameter validated: estimated={estimated}px, "
            f"reference={reference}px, tol=±{CIRCLE_DIAMETER_TOLERANCE_PX}px"
        )
        return "PASS", msg

    msg = (
        f"Diameter mismatch: estimated={estimated}px, reference={reference}px, "
        f"diff={diff}px, tol=±{CIRCLE_DIAMETER_TOLERANCE_PX}px"
    )
    return "FAIL", msg


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
    *,
    verbose: bool = False,
) -> CalibrationResult:
    """Attempt to calibrate reader from disk images and validate against centers.txt.

    Returns:
        CalibrationResult with status PASS/FAIL/NA and detail message.

    """
    clear_path = example_dir / "clear.png"
    top_left_path = example_dir / "top_left.png"
    bottom_right_path = example_dir / "bottom_right.png"

    if not (clear_path.exists() and top_left_path.exists() and bottom_right_path.exists()):
        return CalibrationResult(status="NA", message="Calibration frames missing")

    # Load calibration frames
    clear_bytes = clear_path.read_bytes()
    top_left_bytes = top_left_path.read_bytes()
    bottom_right_bytes = bottom_right_path.read_bytes()

    # Run calibration via replay
    driver = ReplayDriver(screenshots=[clear_bytes, top_left_bytes, clear_bytes, bottom_right_bytes])
    try:
        with maybe_suppress_stdout(verbose=verbose):
            reader.calibrate(driver, timeout_s=2.0, poll_interval_s=0.0)
    except (TimeoutError, ValueError, RuntimeError) as err:
        msg = f"Calibration failed: {err}"
        return CalibrationResult(status="FAIL", message=msg)

    # Try to validate against centers.txt
    rows = reader._rows  # noqa: SLF001
    cols = reader._cols  # noqa: SLF001
    reference_centers = read_reference_centers(example_dir)
    if reference_centers is None:
        msg = "Calibration succeeded but centers.txt is missing"
        return CalibrationResult(status="NA", message=msg)

    # Compare calibrated corners vs reference
    tl_ref = reference_centers[(0, 0)]
    br_ref = reference_centers[(rows - 1, cols - 1)]
    tl_calib = reader._top_left_cell_center  # noqa: SLF001
    br_calib = reader._bottom_right_cell_center  # noqa: SLF001

    tl_match = (
        abs(tl_calib[0] - tl_ref[0]) <= CENTER_TOLERANCE_PX and abs(tl_calib[1] - tl_ref[1]) <= CENTER_TOLERANCE_PX
    )
    br_match = (
        abs(br_calib[0] - br_ref[0]) <= CENTER_TOLERANCE_PX and abs(br_calib[1] - br_ref[1]) <= CENTER_TOLERANCE_PX
    )

    if tl_match and br_match:
        msg = f"Calibration validated: TL {tl_calib} vs {tl_ref}, BR {br_calib} vs {br_ref}"
        estimated_diameter = reader.get_estimated_circle_diameter()
        return CalibrationResult(status="PASS", message=msg, estimated_diameter_px=estimated_diameter)

    msg = f"Calibration mismatch: TL {tl_calib} vs {tl_ref}, BR {br_calib} vs {br_ref}"
    return CalibrationResult(status="FAIL", message=msg)


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


@dataclass
class CalibrationResult:
    """Hold calibration status and message for one example."""

    status: str  # PASS | FAIL | NA
    message: str
    estimated_diameter_px: int | None = None


@dataclass
class ExampleResult:
    """Hold final test statuses for one example."""

    example_name: str
    overall_status: str  # PASS | FAIL
    ocr_status: str  # PASS | FAIL | NA
    centers_status: str  # PASS | FAIL | NA
    states_status: str  # PASS | FAIL | NA
    diameter_status: str  # PASS | FAIL | NA


def combine_statuses(statuses: list[str]) -> str:
    """Combine statuses into one PASS/FAIL/NA summary."""
    if not statuses:
        return "NA"
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if all(status == "NA" for status in statuses):
        return "NA"
    if all(status == "PASS" for status in statuses if status != "NA"):
        return "PASS"
    return "FAIL"


def format_status_with_count(status: str, passing_count: int, total_count: int) -> str:
    """Format one summary status with pass/total counts when applicable."""
    if status == "NA":
        return "NA"
    return f"{status} ({passing_count}/{total_count})"


def extract_image_data(
    reader: BoardReaderTesseractOpenCV,
    image_path: Path,
    example_name: str,
    *,
    verbose: bool = False,
) -> ImageExtractionResult:
    """Extract board rows and cell states for one image.

    Returns:
        ImageExtractionResult with extracted values and captured errors.

    """
    board_error: Exception | None = None
    cell_error: Exception | None = None
    board_rows: list[str] = []
    cell_states: list[list[Highlight]] | None = None
    image = None

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
        with maybe_suppress_stdout(verbose=verbose):
            cell_states = reader._extract_cell_states(image)  # noqa: SLF001
    except (OSError, ValueError, RuntimeError) as error:
        cell_error = error

    try:
        with maybe_suppress_stdout(verbose=verbose):
            board_rows = reader._extract_board_rows(image)  # noqa: SLF001
    except (OSError, ValueError, RuntimeError) as error:
        board_error = error

    # Always copy debug image for diagnostic purposes, regardless of extraction success
    debug_image_path = OUTPUT_DIR / f"{example_name}_{image_path.stem}_debug_board.png"
    with contextlib.suppress(RuntimeError):
        copy_debug_image(debug_image_path)

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
    return "FAIL", str(extraction.board_error)


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
    return "FAIL", str(extraction.cell_error)


def _init_example_logs(
    *,
    example_name: str,
    calibration: CalibrationResult,
    diameter_status: str,
    diameter_message: str,
) -> tuple[Path, Path, Path, list[str], list[str], list[str]]:
    """Create per-example log paths and initial log lines."""
    board_log_path = OUTPUT_DIR / f"{example_name}_board_rows.log"
    cell_log_path = OUTPUT_DIR / f"{example_name}_cell_states.log"
    calib_log_path = OUTPUT_DIR / f"{example_name}_calibration.log"
    board_log_lines: list[str] = []
    cell_log_lines: list[str] = []
    calib_log_lines = [
        f"Calibration attempt for {example_name}",
        f"Status: {calibration.status}",
        f"Message: {calibration.message}",
        f"Estimated diameter: {calibration.estimated_diameter_px}",
        f"Diameter status: {diameter_status}",
        f"Diameter message: {diameter_message}",
        "",
    ]
    return board_log_path, cell_log_path, calib_log_path, board_log_lines, cell_log_lines, calib_log_lines


def process_example_dir(example_dir: Path, *, verbose: bool = False) -> ExampleResult | None:
    """Process one example directory and print structured test output."""
    example_name = example_dir.name
    image_paths = iter_board_images(example_dir)
    if not image_paths:
        print(f"\n=== Example: {example_name} ===")
        print("Result: SKIP (no board images found)")
        return None

    try:
        reference_rows = read_reference_board(example_dir)
        rows, cols = len(reference_rows), len(reference_rows[0])
    except (OSError, ValueError) as err:
        print(f"\n=== Example: {example_name} ===")
        print(f"Result: SKIP ({err})")
        return None

    expected_cell_states = read_reference_cell_states(example_dir, rows, cols)
    reader = BoardReaderTesseractOpenCV(rows=rows, cols=cols)

    calibration = try_calibrate_and_validate(reader, example_dir, verbose=verbose)
    reference_diameter = read_reference_circle_diameter(example_dir)
    diameter_status, diameter_message = validate_circle_diameter(
        calibration.estimated_diameter_px,
        reference_diameter,
    )

    print(f"\n=== Example: {example_name} ===")
    print(f"Images: {len(image_paths)}")
    print(f"  Centers: {calibration.status} ({calibration.message})")
    print(f"  Diameter: {diameter_status} ({diameter_message})")

    board_log_path, cell_log_path, calib_log_path, board_log_lines, cell_log_lines, calib_log_lines = (
        _init_example_logs(
            example_name=example_name,
            calibration=calibration,
            diameter_status=diameter_status,
            diameter_message=diameter_message,
        )
    )

    ocr_statuses: list[str] = []
    state_statuses: list[str] = []

    for image_path in image_paths:
        print(f"  Image: {image_path.name}")
        extraction = extract_image_data(reader, image_path, example_name, verbose=verbose)

        ocr_status, ocr_score_text = append_board_result(
            board_log_lines,
            image_path,
            extraction,
            reference_rows,
        )
        ocr_statuses.append(ocr_status)
        print(f"    OCR(board): {ocr_status} ({ocr_score_text})")

        cell_status, cell_score_text = append_cell_result(
            cell_log_lines,
            image_path,
            extraction,
            expected_cell_states,
        )
        if expected_cell_states is not None:
            state_statuses.append(cell_status)
        print(f"    States: {cell_status} ({cell_score_text})")

    ocr_overall = combine_statuses(ocr_statuses)
    centers_overall = calibration.status
    states_overall = "NA" if expected_cell_states is None else combine_statuses(state_statuses)

    ocr_pass_count = sum(1 for status in ocr_statuses if status == "PASS")
    ocr_total_count = len(ocr_statuses)
    states_pass_count = sum(1 for status in state_statuses if status == "PASS")
    states_total_count = len(state_statuses)

    overall_status = (
        "FAIL"
        if any(status == "FAIL" for status in [ocr_overall, centers_overall, states_overall, diameter_status])
        else "PASS"
    )

    print(
        "  Example Summary: "
        f"Centers={centers_overall}, "
        f"Diameter={diameter_status}, "
        f"OCR={format_status_with_count(ocr_overall, ocr_pass_count, ocr_total_count)}, "
        f"States={format_status_with_count(states_overall, states_pass_count, states_total_count)}"
    )
    print(f"  Overall: {overall_status}")

    board_log_path.write_text("\n".join(board_log_lines), encoding="utf-8")
    cell_log_path.write_text("\n".join(cell_log_lines), encoding="utf-8")
    calib_log_path.write_text("\n".join(calib_log_lines), encoding="utf-8")

    return ExampleResult(
        example_name=example_name,
        overall_status=overall_status,
        ocr_status=ocr_overall,
        centers_status=centers_overall,
        states_status=states_overall,
        diameter_status=diameter_status,
    )


def main() -> int:
    """Process all example directories under `data/`."""
    args = parse_args()

    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"Input directory not found: {DATA_DIR}")
        return 1

    # Remove OUTPUT_DIR if it exists to start fresh, then recreate it
    if OUTPUT_DIR.exists():
        if not OUTPUT_DIR.is_dir():
            print(f"Output path exists and is not a directory: {OUTPUT_DIR}")
            return 1
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    example_dirs = iter_example_dirs(DATA_DIR)
    if not example_dirs:
        print(f"No example directories found in {DATA_DIR}")
        return 1

    results: list[ExampleResult] = []
    skipped_examples = 0

    try:
        for example_dir in example_dirs:
            result = process_example_dir(example_dir, verbose=args.verbose)
            if result is None:
                skipped_examples += 1
            else:
                results.append(result)
    finally:
        if TEMP_DEBUG_IMAGE.exists():
            TEMP_DEBUG_IMAGE.unlink()

    pass_examples = sum(1 for result in results if result.overall_status == "PASS")
    fail_examples = sum(1 for result in results if result.overall_status == "FAIL")

    def summarize_test(statuses: list[str]) -> tuple[int, int, int]:
        return (
            sum(1 for status in statuses if status == "PASS"),
            sum(1 for status in statuses if status == "FAIL"),
            sum(1 for status in statuses if status == "NA"),
        )

    ocr_pass, ocr_fail, ocr_na = summarize_test([result.ocr_status for result in results])
    centers_pass, centers_fail, centers_na = summarize_test([result.centers_status for result in results])
    diameter_pass, diameter_fail, diameter_na = summarize_test([result.diameter_status for result in results])
    states_pass, states_fail, states_na = summarize_test([result.states_status for result in results])

    print("\n=== Final Summary ===")
    print(f"Examples: PASS={pass_examples}, FAIL={fail_examples}, SKIP={skipped_examples}, TOTAL={len(example_dirs)}")
    print(f"OCR(board): PASS={ocr_pass}, FAIL={ocr_fail}, NA={ocr_na}")
    print(f"Centers: PASS={centers_pass}, FAIL={centers_fail}, NA={centers_na}")
    print(f"Diameter: PASS={diameter_pass}, FAIL={diameter_fail}, NA={diameter_na}")
    print(f"States: PASS={states_pass}, FAIL={states_fail}, NA={states_na}")

    return 0 if fail_examples == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
