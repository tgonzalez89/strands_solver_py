#!/usr/bin/env python3
"""Generate synthetic Strands example datasets for OCR/cell-state evaluation."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from string import ascii_uppercase

sys.path.insert(0, str(Path(__file__).parent / "src"))

from strands_solver.image_renderer.board_image_renderer import RenderConfig, render_board_png

DATA_DIR = Path("data")

ROWS = 8
COLS = 6
ALL_COORDS = {(row, col) for row in range(ROWS) for col in range(COLS)}


def build_baseline_board_rows() -> list[str]:
    """Build baseline board with A-Z then only A's to fill remaining cells."""
    letters = list(ascii_uppercase)
    letters.extend(["A"] * (ROWS * COLS - len(letters)))
    return ["".join(letters[row * COLS : (row + 1) * COLS]) for row in range(ROWS)]


def build_random_board_rows(rng: random.Random) -> list[str]:
    """Build random uppercase board rows for one synthetic mixed example."""
    letters = [rng.choice(ascii_uppercase) for _ in range(ROWS * COLS)]
    return ["".join(letters[row * COLS : (row + 1) * COLS]) for row in range(ROWS)]


def write_board_file(example_dir: Path, board_rows: list[str]) -> None:
    """Write board.txt for one example directory."""
    (example_dir / "board.txt").write_text("\n".join(board_rows) + "\n", encoding="utf-8")


def write_cell_states_file(example_dir: Path, state_symbols: list[list[str]]) -> None:
    """Write cell_states.txt using N/W/S symbols (space-separated)."""
    lines = [" ".join(row) for row in state_symbols]
    (example_dir / "cell_states.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_example(  # noqa: PLR0913
    example_dir: Path,
    board_rows: list[str],
    image_name: str,
    mode: str,
    word_coords: set[tuple[int, int]],
    spangram_coords: set[tuple[int, int]],
    selection_coords: set[tuple[int, int]],
    expected_symbols: list[list[str]],
) -> None:
    """Render and save one synthetic example image and references."""
    example_dir.mkdir(parents=True, exist_ok=True)
    write_board_file(example_dir, board_rows)
    write_cell_states_file(example_dir, expected_symbols)

    png_bytes, _ = render_board_png(
        board_rows,
        mode=mode,
        word_coords=word_coords,
        spangram_coords=spangram_coords,
        selection_coords=selection_coords,
        config=RenderConfig.board_reader_v3(),
    )
    (example_dir / image_name).write_bytes(png_bytes)


def make_uniform_symbols(symbol: str) -> list[list[str]]:
    """Return an rows x cols grid filled with one symbol."""
    return [[symbol for _ in range(COLS)] for _ in range(ROWS)]


def make_expected_symbols_with_selection_as_none(
    word_coords: set[tuple[int, int]],
    spangram_coords: set[tuple[int, int]],
    selection_coords: set[tuple[int, int]],
) -> list[list[str]]:
    """Build expected N/W/S grid; selection cells are expected as N."""
    symbols = [["N" for _ in range(COLS)] for _ in range(ROWS)]
    for row, col in word_coords:
        symbols[row][col] = "W"
    for row, col in spangram_coords:
        symbols[row][col] = "S"
    for row, col in selection_coords:
        if symbols[row][col] == "N":
            symbols[row][col] = "N"
    return symbols


def generate_baseline_cases() -> None:
    """Generate baseline examples with all cells in one visual state/color."""
    baseline_board_rows = build_baseline_board_rows()
    cases = [
        ("example_synth_baseline_none_light", "light", set(), set(), set(), make_uniform_symbols("N")),
        ("example_synth_baseline_none_dark", "dark", set(), set(), set(), make_uniform_symbols("N")),
        ("example_synth_baseline_word", "light", ALL_COORDS, set(), set(), make_uniform_symbols("W")),
        ("example_synth_baseline_word_dark", "dark", ALL_COORDS, set(), set(), make_uniform_symbols("W")),
        ("example_synth_baseline_spangram", "light", set(), ALL_COORDS, set(), make_uniform_symbols("S")),
        (
            "example_synth_baseline_spangram_dark",
            "dark",
            set(),
            ALL_COORDS,
            set(),
            make_uniform_symbols("S"),
        ),
        (
            "example_synth_baseline_selection",
            "light",
            set(),
            set(),
            ALL_COORDS,
            make_uniform_symbols("N"),
        ),
        (
            "example_synth_baseline_selection_dark",
            "dark",
            set(),
            set(),
            ALL_COORDS,
            make_uniform_symbols("N"),
        ),
    ]

    for name, mode, word_coords, spangram_coords, selection_coords, expected_symbols in cases:
        render_example(
            DATA_DIR / name,
            baseline_board_rows,
            "synthetic.png",
            mode,
            word_coords,
            spangram_coords,
            selection_coords,
            expected_symbols,
        )


def generate_mixed_cases(count: int = 12, seed: int = 20260513) -> None:
    """Generate mixed-highlight synthetic examples with deterministic randomness."""
    rng = random.Random(seed)
    coords = sorted(ALL_COORDS)

    for idx in range(count):
        example_name = f"example_synth_mixed_{idx + 1:02d}"
        example_dir = DATA_DIR / example_name
        board_rows = build_random_board_rows(rng)

        shuffled = coords.copy()
        rng.shuffle(shuffled)

        word_count = rng.randint(8, 16)
        spangram_count = rng.randint(6, 12)
        selection_count = rng.randint(6, 14)

        word_coords = set(shuffled[:word_count])
        spangram_candidates = [coord for coord in shuffled[word_count:] if coord not in word_coords]
        spangram_coords = set(spangram_candidates[:spangram_count])
        selection_candidates = [
            coord
            for coord in shuffled[word_count + spangram_count :]
            if coord not in word_coords and coord not in spangram_coords
        ]
        selection_coords = set(selection_candidates[:selection_count])

        expected_symbols = make_expected_symbols_with_selection_as_none(
            word_coords,
            spangram_coords,
            selection_coords,
        )

        mode = "light" if idx % 2 == 0 else "dark"
        render_example(
            example_dir,
            board_rows,
            "synthetic.png",
            mode,
            word_coords,
            spangram_coords,
            selection_coords,
            expected_symbols,
        )


def clean_old_synthetic_dirs() -> None:
    """Remove previously generated synthetic example directories."""
    for path in DATA_DIR.glob("example_synth_*"):
        if path.is_dir():
            shutil.rmtree(path)


def main() -> int:
    """Generate synthetic datasets under data/example_synth_*."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_old_synthetic_dirs()
    generate_baseline_cases()
    generate_mixed_cases()
    print("Generated synthetic datasets under data/example_synth_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
