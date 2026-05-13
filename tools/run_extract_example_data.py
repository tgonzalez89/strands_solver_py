#!/usr/bin/env python3
"""Run OCR board-row and cell-state extraction for `data/example*` directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from strands_solver.board_reader.board_reader import Highlight
from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCV

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
    """Return sorted example directories under the data folder.

    Args:
        data_dir: Base data directory.

    Returns:
        Sorted example directories whose names start with `example`.

    """
    return sorted(path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("example"))


def iter_images(input_dir: Path) -> list[Path]:
    """Return sorted image paths from one example directory.

    Args:
        input_dir: Example directory to scan.

    Returns:
        Sorted image files with known screenshot extensions.

    """
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_reference_board(example_dir: Path) -> list[str]:
    """Read expected board rows from `board.txt`.

    Args:
        example_dir: Example directory containing `board.txt`.

    Returns:
        Non-empty stripped board rows.

    """
    board_path = example_dir / "board.txt"
    lines = [line.strip() for line in board_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def read_reference_cell_states(example_dir: Path, rows: int, cols: int) -> list[list[Highlight]] | None:
    """Read expected cell-state grid from cell_states.txt if present.

    Format supports either space-separated symbols (e.g. "N W S N W S")
    or compact rows (e.g. "NWSNWS"). Supported symbols: N, W, S.
    """
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


def validate_reference_board(reference_rows: list[str], example_dir: Path) -> tuple[int, int]:
    """Validate reference board shape and return rows and columns.

    Args:
        reference_rows: Board rows loaded from fixture.
        example_dir: Directory used for contextual error messages.

    Returns:
        Tuple of `(rows, cols)`.

    Raises:
        ValueError: If the board is empty or has inconsistent row lengths.

    """
    if not reference_rows:
        msg = f"Reference board is empty in {example_dir / 'board.txt'}"
        raise ValueError(msg)

    cols = len(reference_rows[0])
    if any(len(row) != cols for row in reference_rows):
        msg = f"Reference board has inconsistent row lengths in {example_dir / 'board.txt'}"
        raise ValueError(msg)

    return len(reference_rows), cols


def to_symbol(state: Highlight) -> str:
    """Map highlight enum value to output symbol.

    Args:
        state: Highlight value to convert.

    Returns:
        `"N"`, `"W"`, or `"S"`.

    """
    if state == Highlight.WORD:
        return "W"
    if state == Highlight.SPANGRAM:
        return "S"
    return "N"


def format_grid(cell_states: list[list[Highlight]]) -> str:
    """Format a state grid with space-separated N/W/S symbols.

    Args:
        cell_states: Cell-state grid to format.

    Returns:
        Multiline text representation of the grid.

    """
    return "\n".join(" ".join(to_symbol(cell) for cell in row) for row in cell_states)


def format_board_log_entry(image_name: str, board_rows: list[str], reference_rows: list[str]) -> list[str]:
    """Build one OCR board log entry including reference comparison.

    Args:
        image_name: Source image file name.
        board_rows: OCR-extracted board rows.
        reference_rows: Expected board rows from fixture.

    Returns:
        Log lines for one image entry.

    """
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
    """Compute OCR board-level score metrics.

    Args:
        board_rows: OCR-extracted board rows.
        reference_rows: Expected board rows from fixture.

    Returns:
        Tuple of `(identified_letters, total_letters, exact_match, complete_failure)`.

    """
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
    """Compute cell-state score metrics.

    Args:
        extracted_states: Extracted cell-state grid.
        expected_states: Expected cell-state grid.

    Returns:
        Tuple of `(matched_cells, total_cells, exact_match)`.

    """
    total_cells = len(expected_states) * len(expected_states[0]) if expected_states else 0
    matched_cells = 0
    for row_idx in range(len(expected_states)):
        for col_idx in range(len(expected_states[0])):
            if extracted_states[row_idx][col_idx] == expected_states[row_idx][col_idx]:
                matched_cells += 1
    return matched_cells, total_cells, matched_cells == total_cells


def copy_debug_image(output_path: Path) -> None:
    """Copy the temporary OCR debug image to its final destination.

    Args:
        output_path: Destination path for the copied debug image.

    Raises:
        RuntimeError: If the temporary debug image does not exist.

    """
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


def process_example_dir(example_dir: Path) -> tuple[int, int]:
    """Process one example directory and write board/cell-state debug outputs.

    Args:
        example_dir: Example directory containing images and references.

    Returns:
        Tuple of `(successful_images, total_images)`.

    """
    example_name = example_dir.name
    image_paths = iter_images(example_dir)
    if not image_paths:
        print(f"SKIP {example_name}: no images found")
        return 0, 0

    reference_rows = read_reference_board(example_dir)
    rows, cols = validate_reference_board(reference_rows, example_dir)
    expected_cell_states = read_reference_cell_states(example_dir, rows, cols)
    reader = BoardReaderTesseractOpenCV(rows=rows, cols=cols)

    board_log_path = OUTPUT_DIR / f"{example_name}_board_rows.log"
    cell_log_path = OUTPUT_DIR / f"{example_name}_cell_states.log"
    board_log_lines: list[str] = []
    cell_log_lines: list[str] = []

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

    print(f"Wrote {board_log_path}")
    print(f"Wrote {cell_log_path}")
    return success_count, len(image_paths)


def main() -> int:
    """Process all example directories under `data/`.

    Returns:
        Process exit code (`0` if all processed images succeeded).

    """
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
    try:
        for example_dir in example_dirs:
            success_count, image_count = process_example_dir(example_dir)
            total_success += success_count
            total_images += image_count
    finally:
        if TEMP_DEBUG_IMAGE.exists():
            TEMP_DEBUG_IMAGE.unlink()

    print(f"Processed {total_success}/{total_images} images across {len(example_dirs)} example directories")
    return 0 if total_success == total_images else 1


if __name__ == "__main__":
    raise SystemExit(main())
