"""Reusable Strands board image rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw as PILImageDraw
    from PIL import ImageFont as PILImageFont

    HAS_EXTRAS = True
except ModuleNotFoundError:
    HAS_EXTRAS = False

if TYPE_CHECKING:
    from pathlib import Path

    from strands_solver.util.util import BoardCoord, PixelCoord


@dataclass(frozen=True)
class Theme:
    """Color theme for board rendering."""

    canvas_color: tuple[int, int, int]
    board_color: tuple[int, int, int]
    unselected_letter_color: tuple[int, int, int]
    word_fill_color: tuple[int, int, int]
    word_letter_color: tuple[int, int, int]
    spangram_fill_color: tuple[int, int, int]
    spangram_letter_color: tuple[int, int, int]


LIGHT_THEME = Theme(
    canvas_color=(255, 255, 255),
    board_color=(255, 255, 255),
    unselected_letter_color=(18, 18, 18),
    word_fill_color=(174, 223, 238),
    word_letter_color=(18, 18, 18),
    spangram_fill_color=(248, 205, 5),
    spangram_letter_color=(18, 18, 18),
)

DARK_THEME = Theme(
    canvas_color=(18, 18, 18),
    board_color=(18, 18, 18),
    unselected_letter_color=(248, 248, 248),
    word_fill_color=(174, 223, 238),
    word_letter_color=(18, 18, 18),
    spangram_fill_color=(248, 205, 5),
    spangram_letter_color=(18, 18, 18),
)


@dataclass(frozen=True)
class RenderConfig:
    """Visual layout configuration for board rendering."""

    width: int = 1080
    height: int = 2246
    board_width_ratio: float = 0.74
    board_height_ratio: float = 0.53
    board_center_y_ratio: float = 0.54
    cell_radius_ratio: float = 0.42
    font_size_ratio: float = 0.48
    font_path: Path | None = None


def pick_theme(mode: str) -> Theme:
    """Select the render theme for a mode string.

    Args:
        mode: Rendering mode, typically `"light"` or `"dark"`.

    Returns:
        Theme configuration for the selected mode.

    """
    if mode == "light":
        return LIGHT_THEME
    return DARK_THEME


def validate_board_shape(board_rows: list[str]) -> None:
    """Validate board shape constraints required by the renderer.

    Args:
        board_rows: Board rows to validate.

    Raises:
        ValueError: If the board is empty, has empty first row, or has
            inconsistent row widths.

    """
    if not board_rows:
        msg = "Board is empty."
        raise ValueError(msg)

    width = len(board_rows[0])
    if width == 0:
        msg = "Board has an empty first row."
        raise ValueError(msg)

    for row_num, row in enumerate(board_rows, start=1):
        if len(row) != width:
            msg = f"Board row {row_num} has width {len(row)} but expected {width}."
            raise ValueError(msg)


def build_coord_sets(
    moves: list[list[BoardCoord]],
    spangram_indexes: set[int],
    row_count: int,
    col_count: int,
) -> tuple[set[BoardCoord], set[BoardCoord]]:
    """Build word and spangram highlight sets from indexed move paths.

    Args:
        moves: Move paths to translate into highlight coordinates.
        spangram_indexes: Move indexes that should be highlighted as spangram.
        row_count: Board row count for bounds filtering.
        col_count: Board column count for bounds filtering.

    Returns:
        Tuple of `(word_coords, spangram_coords)` highlight sets.

    """
    word_coords: set[BoardCoord] = set()
    spangram_coords: set[BoardCoord] = set()

    for move_idx, move in enumerate(moves):
        target_set = spangram_coords if move_idx in spangram_indexes else word_coords
        for row_idx, col_idx in move:
            if 0 <= row_idx < row_count and 0 <= col_idx < col_count:
                target_set.add((row_idx, col_idx))

    return word_coords, spangram_coords


def _load_font(config: RenderConfig, font_size: int) -> object:
    """Load a font object suitable for board lettering.

    Args:
        config: Render configuration containing optional font path.
        font_size: Requested font size in pixels.

    Returns:
        Pillow font object.

    Raises:
        ModuleNotFoundError: If Pillow is not installed.

    """
    if not HAS_EXTRAS:
        msg = (
            "Pillow is required. Install optional deps with "
            "`uv sync --extra device`, `pip install .[device]`, "
            "or `pip install pillow`."
        )
        raise ModuleNotFoundError(
            msg,
        )

    image_font = cast("Any", PILImageFont)

    if config.font_path is not None:
        return image_font.truetype(str(config.font_path), size=font_size)

    try:
        return image_font.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        return image_font.load_default()


def render_board_png(  # noqa: PLR0915
    board_rows: list[str],
    *,
    mode: str,
    word_coords: set[BoardCoord] | None = None,
    spangram_coords: set[BoardCoord] | None = None,
    config: RenderConfig | None = None,
) -> tuple[bytes, dict[BoardCoord, PixelCoord]]:
    """Render a board PNG and return image bytes plus cell-center mapping.

    Args:
        board_rows: Board rows to render.
        mode: Rendering mode, `"light"` or `"dark"`.
        word_coords: Coordinates to highlight as regular word cells.
        spangram_coords: Coordinates to highlight as spangram cells.
        config: Optional rendering configuration override.

    Returns:
        Tuple containing PNG bytes and a coordinate-to-pixel-center map.

    Raises:
        ModuleNotFoundError: If Pillow is not installed.
        ValueError: If board shape or render dimensions are invalid.

    """
    if not HAS_EXTRAS:
        msg = (
            "Pillow is required. Install optional deps with "
            "`uv sync --extra device`, `pip install .[device]`, "
            "or `pip install pillow`."
        )
        raise ModuleNotFoundError(
            msg,
        )

    normalized_rows = [row.upper() for row in board_rows]
    validate_board_shape(normalized_rows)

    cfg = config or RenderConfig()
    if cfg.width <= 0 or cfg.height <= 0:
        msg = "Render width and height must be positive integers."
        raise ValueError(msg)

    rows = len(normalized_rows)
    cols = len(normalized_rows[0])
    active_word_coords = word_coords or set()
    active_spangram_coords = spangram_coords or set()

    theme = pick_theme(mode)
    image_module = cast("Any", PILImage)
    draw_module = cast("Any", PILImageDraw)
    image = image_module.new("RGB", (cfg.width, cfg.height), color=theme.canvas_color)
    draw = draw_module.Draw(image)

    board_width = int(cfg.width * cfg.board_width_ratio)
    board_height = int(cfg.height * cfg.board_height_ratio)
    board_left = (cfg.width - board_width) // 2
    board_top = int(cfg.height * cfg.board_center_y_ratio - board_height / 2)
    board_right = board_left + board_width
    board_bottom = board_top + board_height

    draw.rounded_rectangle(
        [(board_left, board_top), (board_right, board_bottom)],
        radius=max(8, int(min(board_width, board_height) * 0.03)),
        fill=theme.board_color,
    )

    cell_width = board_width / cols
    cell_height = board_height / rows
    cell_radius = int(min(cell_width, cell_height) * cfg.cell_radius_ratio)
    font_size = max(10, int(min(cell_width, cell_height) * cfg.font_size_ratio))
    font: Any = _load_font(cfg, font_size)

    centers: dict[BoardCoord, PixelCoord] = {}
    for row_idx, row_text in enumerate(normalized_rows):
        for col_idx, letter in enumerate(row_text):
            center_x = int(board_left + (col_idx + 0.5) * cell_width)
            center_y = int(board_top + (row_idx + 0.5) * cell_height)
            cell_coord = (row_idx, col_idx)
            centers[cell_coord] = (center_x, center_y)

            fill_color: tuple[int, int, int] | None
            letter_color = theme.unselected_letter_color
            if cell_coord in active_spangram_coords:
                fill_color = theme.spangram_fill_color
                letter_color = theme.spangram_letter_color
            elif cell_coord in active_word_coords:
                fill_color = theme.word_fill_color
                letter_color = theme.word_letter_color
            else:
                fill_color = None

            if fill_color is not None:
                draw.ellipse(
                    [
                        (center_x - cell_radius, center_y - cell_radius),
                        (center_x + cell_radius, center_y + cell_radius),
                    ],
                    fill=fill_color,
                )

            draw.text((center_x, center_y), letter, fill=letter_color, font=font, anchor="mm")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return (buffer.getvalue(), centers)
