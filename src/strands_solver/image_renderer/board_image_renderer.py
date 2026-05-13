"""Reusable Strands board image rendering utilities."""

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

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

from strands_solver.util.util import validate_board


@dataclass(frozen=True)
class Theme:
    """Color theme for board rendering."""

    background_color: tuple[int, int, int]
    unselected_letter_color: tuple[int, int, int]
    word_fill_color: tuple[int, int, int]
    word_letter_color: tuple[int, int, int]
    spangram_fill_color: tuple[int, int, int]
    spangram_letter_color: tuple[int, int, int]
    selection_fill_color: tuple[int, int, int]
    selection_letter_color: tuple[int, int, int]


LIGHT_THEME = Theme(
    background_color=(255, 255, 255),
    unselected_letter_color=(18, 18, 18),
    word_fill_color=(174, 223, 238),
    word_letter_color=(18, 18, 18),
    spangram_fill_color=(248, 205, 5),
    spangram_letter_color=(18, 18, 18),
    selection_fill_color=(219, 216, 197),
    selection_letter_color=(18, 18, 18),
)

DARK_THEME = Theme(
    background_color=(18, 18, 18),
    unselected_letter_color=(248, 248, 248),
    word_fill_color=(174, 223, 238),
    word_letter_color=(18, 18, 18),
    spangram_fill_color=(248, 205, 5),
    spangram_letter_color=(18, 18, 18),
    selection_fill_color=(219, 216, 197),
    selection_letter_color=(18, 18, 18),
)


@dataclass(frozen=True)
class RenderConfig:
    """Visual layout configuration for board rendering."""

    width: int = 1080
    height: int = 2246
    board_left_px: int = 79
    board_top_px: int = 678
    cell_width_px: float = 153.0
    cell_height_px: float = 148.5
    cell_radius_ratio: float = 0.42
    font_size_ratio: float = 0.44
    font_path: Path | None = None


@dataclass(frozen=True)
class CellDrawContext:
    """Context bundle for drawing board cells."""

    draw: PILImageDraw.ImageDraw
    board_left: int
    board_top: int
    cell_width: float
    cell_height: float
    cell_radius: int
    theme: Theme
    font: PILImageFont.ImageFont | PILImageFont.FreeTypeFont | PILImageFont.TransposedFont
    active_word_coords: set[BoardCoord]
    active_spangram_coords: set[BoardCoord]
    active_selection_coords: set[BoardCoord]


THEMES: dict[str, Theme] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


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


def _load_font(
    config: RenderConfig,
    font_size: int,
) -> PILImageFont.ImageFont | PILImageFont.FreeTypeFont | PILImageFont.TransposedFont:
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

    if config.font_path is not None:
        return PILImageFont.truetype(str(config.font_path), size=font_size)

    try:
        return PILImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except OSError:
        try:
            return PILImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
        except OSError:
            return PILImageFont.load_default()


def _resolve_board_geometry(cfg: RenderConfig, rows: int, cols: int) -> tuple[float, float, int, int, float, float]:
    """Resolve board placement and cell geometry.

    Args:
        cfg: Renderer geometry configuration.
        rows: Number of board rows.
        cols: Number of board columns.

    Returns:
        Tuple of `(board_width, board_height, board_left, board_top, cell_width, cell_height)`.

    """
    cell_width = cfg.cell_width_px
    cell_height = cfg.cell_height_px
    board_width = cell_width * cols
    board_height = cell_height * rows
    board_left = cfg.board_left_px
    board_top = cfg.board_top_px

    return board_width, board_height, board_left, board_top, cell_width, cell_height


def _resolve_cell_colors(
    cell_coord: BoardCoord,
    theme: Theme,
    active_word_coords: set[BoardCoord],
    active_spangram_coords: set[BoardCoord],
    active_selection_coords: set[BoardCoord],
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int]]:
    """Resolve fill and letter colors for one cell coordinate.

    Returns:
        Tuple of `(fill_color, letter_color)` where `fill_color` is optional.

    """
    if cell_coord in active_spangram_coords:
        return theme.spangram_fill_color, theme.spangram_letter_color
    if cell_coord in active_word_coords:
        return theme.word_fill_color, theme.word_letter_color
    if cell_coord in active_selection_coords:
        return theme.selection_fill_color, theme.selection_letter_color
    return None, theme.unselected_letter_color


def _draw_board_cells(normalized_rows: list[str], context: CellDrawContext) -> dict[BoardCoord, PixelCoord]:
    """Draw all board cells and return center coordinates.

    Args:
        normalized_rows: Uppercased board rows to draw.
        context: Shared drawing context and style values.

    Returns:
        Mapping from board coordinates to rendered pixel centers.

    """
    draw = context.draw
    font = context.font

    centers: dict[BoardCoord, PixelCoord] = {}
    for row_idx, row_text in enumerate(normalized_rows):
        for col_idx, letter in enumerate(row_text):
            center_x = int(context.board_left + (col_idx + 0.5) * context.cell_width)
            center_y = int(context.board_top + (row_idx + 0.5) * context.cell_height)
            cell_coord = (row_idx, col_idx)
            centers[cell_coord] = (center_x, center_y)

            fill_color, letter_color = _resolve_cell_colors(
                cell_coord,
                context.theme,
                context.active_word_coords,
                context.active_spangram_coords,
                context.active_selection_coords,
            )

            if fill_color is not None:
                draw.ellipse(
                    [
                        (center_x - context.cell_radius, center_y - context.cell_radius),
                        (center_x + context.cell_radius, center_y + context.cell_radius),
                    ],
                    fill=fill_color,
                )

            draw.text((center_x, center_y), letter, fill=letter_color, font=font, anchor="mm")

    return centers


def render_board_png(
    board_rows: list[str],
    *,
    mode: str,
    config: RenderConfig | None = None,
    **highlight_coords: set[BoardCoord] | None,
) -> tuple[bytes, dict[BoardCoord, PixelCoord]]:
    """Render a board PNG and return image bytes plus cell-center mapping.

    Args:
        board_rows: Board rows to render.
        mode: Rendering mode, `"light"` or `"dark"`.
        highlight_coords: Optional keyword highlight sets using keys
            `word_coords`, `spangram_coords`, and `selection_coords`.
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
    validate_board([row.lower() for row in board_rows])
    normalized_rows = [row.upper() for row in board_rows]

    cfg = config or RenderConfig()
    if cfg.width <= 0 or cfg.height <= 0:
        msg = "Render width and height must be positive integers."
        raise ValueError(msg)

    rows = len(normalized_rows)
    cols = len(normalized_rows[0])
    active_word_coords = highlight_coords.get("word_coords") or set()
    active_spangram_coords = highlight_coords.get("spangram_coords") or set()
    active_selection_coords = highlight_coords.get("selection_coords") or set()

    theme = THEMES.get(mode, DARK_THEME)
    image = PILImage.new("RGB", (cfg.width, cfg.height), color=theme.background_color)
    draw = PILImageDraw.Draw(image)

    board_width, board_height, board_left, board_top, cell_width, cell_height = _resolve_board_geometry(cfg, rows, cols)

    board_right = board_left + board_width
    board_bottom = board_top + board_height

    draw.rounded_rectangle(
        [(board_left, board_top), (board_right, board_bottom)],
        radius=max(8, int(min(board_width, board_height) * 0.03)),
        fill=theme.background_color,
    )

    cell_radius = int(min(cell_width, cell_height) * cfg.cell_radius_ratio)
    font_size = max(10, int(min(cell_width, cell_height) * cfg.font_size_ratio))
    font = _load_font(cfg, font_size)

    draw_context = CellDrawContext(
        draw,
        board_left,
        board_top,
        cell_width,
        cell_height,
        cell_radius,
        theme,
        font,
        active_word_coords,
        active_spangram_coords,
        active_selection_coords,
    )

    centers = _draw_board_cells(
        normalized_rows,
        draw_context,
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return (buffer.getvalue(), centers)
