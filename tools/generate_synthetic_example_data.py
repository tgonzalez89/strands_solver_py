#!/usr/bin/env python3
"""Generate synthetic Strands datasets for OCR and cell-state evaluation."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from string import ascii_uppercase

from strands_solver.image_renderer.board_image_renderer import (
    DARK_THEME,
    LIGHT_THEME,
    RenderConfig,
    Theme,
    render_board_png,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"

ROWS = 8
COLS = 6
ALL_COORDS = {(row, col) for row in range(ROWS) for col in range(COLS)}


def _jitter_channel(value: int, delta: int, rng: random.Random) -> int:
    """Return one RGB channel value with bounded random jitter."""
    return max(0, min(255, value + rng.randint(-delta, delta)))


def _jitter_color(color: tuple[int, int, int], delta: int, rng: random.Random) -> tuple[int, int, int]:
    """Apply bounded jitter to an RGB color tuple."""
    return (
        _jitter_channel(color[0], delta, rng),
        _jitter_channel(color[1], delta, rng),
        _jitter_channel(color[2], delta, rng),
    )


def build_randomized_theme(base_theme: Theme, rng: random.Random) -> Theme:
    """Build a randomized color theme while staying calibration/OCR compatible.

    Notes:
    - Selection fill color is only jittered by a small amount.
    - Other colors are jittered more to increase synthetic diversity.

    """
    return Theme(
        background_color=_jitter_color(base_theme.background_color, 5, rng),
        unselected_letter_color=_jitter_color(base_theme.unselected_letter_color, 5, rng),
        word_fill_color=_jitter_color(base_theme.word_fill_color, 5, rng),
        word_letter_color=_jitter_color(base_theme.word_letter_color, 5, rng),
        spangram_fill_color=_jitter_color(base_theme.spangram_fill_color, 5, rng),
        spangram_letter_color=_jitter_color(base_theme.spangram_letter_color, 5, rng),
        selection_fill_color=_jitter_color(base_theme.selection_fill_color, 5, rng),
        selection_letter_color=_jitter_color(base_theme.selection_letter_color, 5, rng),
    )


@dataclass(frozen=True)
class RenderExampleSpec:
    """Describe one synthetic render invocation."""

    image_name: str
    word_coords: set[tuple[int, int]]
    spangram_coords: set[tuple[int, int]]
    selection_coords: set[tuple[int, int]]
    expected_symbols: list[list[str]]
    config: RenderConfig | None = None
    circle_diameter_px: int | None = None


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


def write_centers_file(example_dir: Path, centers: dict[tuple[int, int], tuple[int, int]]) -> None:
    """Write `centers.txt` with all cell centers.

    Args:
        example_dir: Target example directory.
        centers: Mapping from (row, col) to (x, y) pixel center.

    """
    lines = []
    for row in range(ROWS):
        for col in range(COLS):
            x, y = centers[(row, col)]
            lines.append(f"{row},{col} : {x},{y}")
    (example_dir / "centers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_circle_diameter_file(example_dir: Path, diameter_px: int) -> None:
    """Write `circle_diameter.txt` with the ground-truth circle diameter.

    Args:
        example_dir: Target example directory.
        diameter_px: Circle diameter in pixels.

    """
    (example_dir / "circle_diameter.txt").write_text(str(diameter_px) + "\n", encoding="utf-8")


def render_example(
    example_dir: Path,
    board_rows: list[str],
    spec: RenderExampleSpec,
) -> None:
    """Render and persist one synthetic example image and references.

    Generates:
    - synthetic.png: main screenshot with all highlights
    - clear.png: same board with full word/spangram overlays but no selection
    - top_left.png: word/spangram overlays but (0,0) is ONLY selection (no word/spangram)
    - bottom_right.png: word/spangram overlays but (rows-1,cols-1) is ONLY selection (no word/spangram)
    - centers.txt: all cell centers for validation

    Calibration frames are "noisy" (with word/spangram highlights visible elsewhere) but
    have clean selection blobs at target corners by removing those coords from word/spangram.

    Args:
        example_dir: Directory to create/update for this example.
        board_rows: Board rows rendered into the screenshot.
        spec: Render parameters and expected N/W/S reference grid.

    """
    example_dir.mkdir(parents=True, exist_ok=True)
    write_board_file(example_dir, board_rows)
    write_cell_states_file(example_dir, spec.expected_symbols)

    # Use provided config or default
    config = spec.config or RenderConfig(theme=LIGHT_THEME)
    if config.theme is None:
        config = replace(config, theme=LIGHT_THEME)

    # Render main screenshot with original spec
    png_bytes, centers = render_board_png(
        board_rows,
        word_coords=spec.word_coords,
        spangram_coords=spec.spangram_coords,
        selection_coords=spec.selection_coords,
        config=config,
    )
    (example_dir / spec.image_name).write_bytes(png_bytes)

    # Render clear.png: full highlights but no selection
    clear_bytes, _ = render_board_png(
        board_rows,
        word_coords=spec.word_coords,
        spangram_coords=spec.spangram_coords,
        selection_coords=set(),
        config=config,
    )
    (example_dir / "clear.png").write_bytes(clear_bytes)

    # Render top_left.png: remove (0,0) from word/spangram to give selection precedence
    tl_word_coords = spec.word_coords - {(0, 0)}
    tl_spangram_coords = spec.spangram_coords - {(0, 0)}
    top_left_bytes, _ = render_board_png(
        board_rows,
        word_coords=tl_word_coords,
        spangram_coords=tl_spangram_coords,
        selection_coords={(0, 0)},
        config=config,
    )
    (example_dir / "top_left.png").write_bytes(top_left_bytes)

    # Render bottom_right.png: remove (rows-1,cols-1) from word/spangram
    br_coord = (ROWS - 1, COLS - 1)
    br_word_coords = spec.word_coords - {br_coord}
    br_spangram_coords = spec.spangram_coords - {br_coord}
    bottom_right_bytes, _ = render_board_png(
        board_rows,
        word_coords=br_word_coords,
        spangram_coords=br_spangram_coords,
        selection_coords={br_coord},
        config=config,
    )
    (example_dir / "bottom_right.png").write_bytes(bottom_right_bytes)

    # Write all cell centers for validation
    write_centers_file(example_dir, centers)

    # Write ground-truth circle diameter for validation.
    # Renderer computes radius with int(min(cell_w, cell_h) * ratio), so mirror that logic.
    effective_circle_diameter_px = (
        spec.circle_diameter_px
        if spec.circle_diameter_px is not None
        else 2 * int(min(config.cell_width_px, config.cell_height_px) * config.cell_radius_ratio)
    )
    write_circle_diameter_file(example_dir, effective_circle_diameter_px)


def _fit_board_to_bounds(
    width: int,
    height: int,
    board_aspect_ratio: float,
    max_board_fill_ratio: float,
) -> tuple[float, float]:
    """Compute the largest board that fits inside max fill bounds."""
    max_board_width = width * max_board_fill_ratio
    max_board_height = height * max_board_fill_ratio

    if max_board_width / board_aspect_ratio <= max_board_height:
        fit_board_width = max_board_width
        fit_board_height = fit_board_width / board_aspect_ratio
    else:
        fit_board_height = max_board_height
        fit_board_width = fit_board_height * board_aspect_ratio

    return fit_board_width, fit_board_height


def _choose_board_position(
    rng: random.Random,
    image_size: tuple[int, int],
    board_size: tuple[float, float],
    image_margin_ratio: float,
) -> tuple[int, int]:
    """Choose board top-left position fully in-bounds with preferred margin."""
    width, height = image_size
    board_width, board_height = board_size
    preferred_margin_x = int(width * image_margin_ratio)
    preferred_margin_y = int(height * image_margin_ratio)
    horizontal_slack = max(0, round(width - board_width))
    vertical_slack = max(0, round(height - board_height))

    min_left = min(preferred_margin_x, horizontal_slack)
    min_top = min(preferred_margin_y, vertical_slack)
    max_left = max(min_left, horizontal_slack - preferred_margin_x)
    max_top = max(min_top, vertical_slack - preferred_margin_y)

    left_margin = rng.randint(min_left, max_left) if max_left > min_left else min_left
    top_margin = rng.randint(min_top, max_top) if max_top > min_top else min_top
    return left_margin, top_margin


def generate_randomized_render_config(rng: random.Random) -> tuple[RenderConfig, int]:
    """Generate a randomized RenderConfig to simulate diverse devices.

        Randomizes image geometry first, then derives board and cell geometry in raw
        pixels so the full board always fits in-bounds with meaningful variation.

        Strategy:
        - Choose a base image resolution where the smaller dimension is 1000-2000px.
        - Choose an image aspect ratio between 1:2 and 2:1.
        - Choose a near-square cell aspect ratio between 1:1.25 and 1.25:1.
        - Size the board to fit within 50%-95% of the image where possible, while
            always respecting the 95% max-fit constraint.
        - Place the board with a 2.5% image margin when possible.
        - Derive circle/font ratios from pixel targets relative to cell size.

    Args:
        rng: Deterministic pseudo-random generator.

    Returns:
        Tuple containing:
        - A randomized RenderConfig instance with validated bounds.
        - The circle diameter in pixels (ground truth for the rendered selection circles).

    """
    image_margin_ratio = 0.025
    min_board_fill_ratio = 0.70
    max_board_fill_ratio = 0.95

    # Step 1: Choose image size from smaller dimension + aspect ratio.
    base_resolution = rng.randint(1000, 2000)
    image_aspect_ratio = rng.uniform(0.5, 2.0)
    if image_aspect_ratio >= 1.0:
        width = int(base_resolution * image_aspect_ratio)
        height = base_resolution
    else:
        width = base_resolution
        height = int(base_resolution / image_aspect_ratio)

    # Step 2: Choose a near-square cell aspect ratio and derive board aspect.
    cell_aspect_ratio = rng.uniform(0.9, 1.1)  # width / height
    board_aspect_ratio = (COLS * cell_aspect_ratio) / ROWS

    # Step 3: Compute the largest board that fits inside the 95% bounds.
    fit_board_width, fit_board_height = _fit_board_to_bounds(
        width,
        height,
        board_aspect_ratio,
        max_board_fill_ratio,
    )

    # Step 4: Randomly size the board, keeping at least MIN_BOARD_FILL_RATIO% fill when possible.
    min_board_width = width * min_board_fill_ratio
    min_board_height = height * min_board_fill_ratio

    min_fill_scale_width = min_board_width / fit_board_width
    min_fill_scale_height = min_board_height / fit_board_height
    min_fill_scale = max(min_fill_scale_width, min_fill_scale_height)

    board_scale = rng.uniform(min_fill_scale, 1.0) if min_fill_scale <= 1.0 else 1.0

    board_width = fit_board_width * board_scale
    board_height = fit_board_height * board_scale

    # Step 5: Derive cell dimensions from board size.
    cell_width_px = board_width / COLS
    cell_height_px = board_height / ROWS

    # Step 6: Place the board fully in-bounds with a preferred 2.5% margin.
    left_margin, top_margin = _choose_board_position(
        rng,
        (width, height),
        (board_width, board_height),
        image_margin_ratio,
    )

    # Step 7: Circle diameter and font size as ratios of the smaller cell dimension.
    min_cell_dimension = min(cell_width_px, cell_height_px)
    min_circle_diameter_px = 0.60 * min_cell_dimension
    max_circle_diameter_px = 0.90 * min_cell_dimension
    circle_diameter_px = rng.uniform(min_circle_diameter_px, max_circle_diameter_px)
    font_size_px = rng.uniform(0.35 * min_cell_dimension, 0.40 * min_cell_dimension)

    cell_radius_ratio = (circle_diameter_px / 2.0) / min_cell_dimension
    font_size_ratio = font_size_px / min_cell_dimension

    config = RenderConfig(
        width=width,
        height=height,
        board_left_px=left_margin,
        board_top_px=top_margin,
        cell_width_px=cell_width_px,
        cell_height_px=cell_height_px,
        cell_radius_ratio=cell_radius_ratio,
        font_size_ratio=font_size_ratio,
    )

    # Return the config and the ground-truth circle diameter in pixels
    return config, round(circle_diameter_px)


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
        ("example_synth_baseline_none_light", LIGHT_THEME, set(), set(), set(), make_uniform_symbols("N")),
        ("example_synth_baseline_none_dark", DARK_THEME, set(), set(), set(), make_uniform_symbols("N")),
        ("example_synth_baseline_word", LIGHT_THEME, ALL_COORDS, set(), set(), make_uniform_symbols("W")),
        ("example_synth_baseline_word_dark", DARK_THEME, ALL_COORDS, set(), set(), make_uniform_symbols("W")),
        ("example_synth_baseline_spangram", LIGHT_THEME, set(), ALL_COORDS, set(), make_uniform_symbols("S")),
        ("example_synth_baseline_spangram_dark", DARK_THEME, set(), ALL_COORDS, set(), make_uniform_symbols("S")),
        ("example_synth_baseline_selection", LIGHT_THEME, set(), set(), ALL_COORDS, make_uniform_symbols("N")),
        ("example_synth_baseline_selection_dark", DARK_THEME, set(), set(), ALL_COORDS, make_uniform_symbols("N")),
    ]

    for name, theme, word_coords, spangram_coords, selection_coords, expected_symbols in cases:
        render_example(
            DATA_DIR / name,
            baseline_board_rows,
            RenderExampleSpec(
                image_name="synthetic.png",
                word_coords=word_coords,
                spangram_coords=spangram_coords,
                selection_coords=selection_coords,
                expected_symbols=expected_symbols,
                config=RenderConfig(theme=theme),
            ),
        )


def generate_mixed_cases(count: int = 16, seed: int = 20260513) -> None:
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

        base_theme = LIGHT_THEME if idx % 2 == 0 else DARK_THEME
        randomized_config, circle_diameter_px = generate_randomized_render_config(rng)
        randomized_theme = build_randomized_theme(base_theme, rng)
        randomized_config = replace(randomized_config, theme=randomized_theme)
        render_example(
            example_dir,
            board_rows,
            RenderExampleSpec(
                image_name="synthetic.png",
                word_coords=word_coords,
                spangram_coords=spangram_coords,
                selection_coords=selection_coords,
                expected_symbols=expected_symbols,
                config=randomized_config,
                circle_diameter_px=circle_diameter_px,
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
