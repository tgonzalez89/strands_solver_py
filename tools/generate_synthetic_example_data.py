#!/usr/bin/env python3
"""Generate synthetic Strands datasets for OCR and cell-state evaluation."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase

from strands_solver.image_renderer.board_image_renderer import RenderConfig, render_board_png

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"

ROWS = 8
COLS = 6
ALL_COORDS = {(row, col) for row in range(ROWS) for col in range(COLS)}


@dataclass(frozen=True)
class RenderExampleSpec:
    """Describe one synthetic render invocation."""

    image_name: str
    mode: str
    word_coords: set[tuple[int, int]]
    spangram_coords: set[tuple[int, int]]
    selection_coords: set[tuple[int, int]]
    expected_symbols: list[list[str]]


def build_baseline_board_rows() -> list[str]:
    """Build baseline board rows with A-Z followed by A-fill.

    Returns:
        Eight board rows (6 columns each) where cells are filled with
        `A..Z` first and remaining cells with `A`.

    """
    letters = list(ascii_uppercase)
    letters.extend(["A"] * (ROWS * COLS - len(letters)))
    return ["".join(letters[row * COLS : (row + 1) * COLS]) for row in range(ROWS)]


def build_random_board_rows(rng: random.Random) -> list[str]:
    """Build random uppercase board rows for one synthetic mixed example.

    Args:
        rng: Deterministic pseudo-random generator.

    Returns:
        Eight randomized board rows (6 columns each).

    """
    letters = [rng.choice(ascii_uppercase) for _ in range(ROWS * COLS)]
    return ["".join(letters[row * COLS : (row + 1) * COLS]) for row in range(ROWS)]


def write_board_file(example_dir: Path, board_rows: list[str]) -> None:
    """Write `board.txt` for one example directory.

    Args:
        example_dir: Target example directory.
        board_rows: Board rows to persist.

    """
    (example_dir / "board.txt").write_text("\n".join(board_rows) + "\n", encoding="utf-8")


def write_cell_states_file(example_dir: Path, state_symbols: list[list[str]]) -> None:
    """Write `cell_states.txt` using space-separated N/W/S symbols.

    Args:
        example_dir: Target example directory.
        state_symbols: Expected state grid using N/W/S symbols.

    """
    lines = [" ".join(row) for row in state_symbols]
    (example_dir / "cell_states.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_example(
    example_dir: Path,
    board_rows: list[str],
    spec: RenderExampleSpec,
) -> None:
    """Render and persist one synthetic example image and references.

    Args:
        example_dir: Directory to create/update for this example.
        board_rows: Board rows rendered into the screenshot.
        spec: Render parameters and expected N/W/S reference grid.

    """
    example_dir.mkdir(parents=True, exist_ok=True)
    write_board_file(example_dir, board_rows)
    write_cell_states_file(example_dir, spec.expected_symbols)

    png_bytes, _ = render_board_png(
        board_rows,
        mode=spec.mode,
        word_coords=spec.word_coords,
        spangram_coords=spec.spangram_coords,
        selection_coords=spec.selection_coords,
        config=RenderConfig(),
    )
    (example_dir / spec.image_name).write_bytes(png_bytes)


def make_uniform_symbols(symbol: str) -> list[list[str]]:
    """Return a rows-by-cols grid filled with one symbol.

    Args:
        symbol: Symbol to fill every cell with.

    Returns:
        Uniform symbol grid with shape `ROWS x COLS`.

    """
    return [[symbol for _ in range(COLS)] for _ in range(ROWS)]


def make_expected_symbols_with_selection_as_none(
    word_coords: set[tuple[int, int]],
    spangram_coords: set[tuple[int, int]],
    selection_coords: set[tuple[int, int]],
) -> list[list[str]]:
    """Build expected N/W/S grid; selection cells remain expected as N.

    Args:
        word_coords: Coordinates expected as word cells.
        spangram_coords: Coordinates expected as spangram cells.
        selection_coords: Coordinates currently selected in UI state.

    Returns:
        Expected N/W/S state grid for evaluation.

    """
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
    """Generate baseline examples with uniform highlight patterns."""
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
            RenderExampleSpec(
                image_name="synthetic.png",
                mode=mode,
                word_coords=word_coords,
                spangram_coords=spangram_coords,
                selection_coords=selection_coords,
                expected_symbols=expected_symbols,
            ),
        )


def generate_mixed_cases(count: int = 12, seed: int = 20260513) -> None:
    """Generate mixed-highlight synthetic examples with deterministic randomness.

    Args:
        count: Number of mixed examples to generate.
        seed: Seed for deterministic pseudo-random generation.

    """
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
            RenderExampleSpec(
                image_name="synthetic.png",
                mode=mode,
                word_coords=word_coords,
                spangram_coords=spangram_coords,
                selection_coords=selection_coords,
                expected_symbols=expected_symbols,
            ),
        )


def clean_old_synthetic_dirs() -> None:
    """Remove previously generated synthetic example directories."""
    for path in DATA_DIR.glob("example_synth_*"):
        if path.is_dir():
            shutil.rmtree(path)


def main() -> int:
    """Generate synthetic datasets under `data/example_synth_*`.

    Returns:
        Process exit code (`0` on success).

    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_old_synthetic_dirs()
    generate_baseline_cases()
    generate_mixed_cases()
    print("Generated synthetic datasets under data/example_synth_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
