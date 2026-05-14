"""Fake device driver that generates synthetic Strands screenshots."""

from dataclasses import replace
from pathlib import Path

from strands_solver.board_reader.board_reader import Highlight
from strands_solver.device.device_driver import DeviceDriver
from strands_solver.image_renderer.board_image_renderer import LIGHT_THEME, RenderConfig, render_board_png
from strands_solver.util.util import (
    PIXEL_COORDINATE_MATCH_TOLERANCE_PX,
    BoardCoord,
    PixelCoord,
    load_board,
    load_moves,
    validate_board,
    validate_move_paths,
)


class DeviceDriverFake(DeviceDriver):
    """In-memory fake device that renders screenshots from board state."""

    def __init__(
        self,
        board: list[str] | Path,
        valid_moves: list[list[BoardCoord]] | Path,
        *,
        spangram_indexes: set[int] | None = None,
        render_config: RenderConfig | None = None,
    ) -> None:
        """Initialize a fake screenshot driver from fixtures.

        Args:
            board: Board rows or path to board fixture.
            valid_moves: Valid move paths or path to move fixture.
            spangram_indexes: Move indexes classified as spangram.
            render_config: Optional rendering configuration.

        Raises:
            ValueError: If board shape or move coordinates are invalid.

        """
        loaded_board = load_board(board) if isinstance(board, Path) else board.copy()
        loaded_moves = (
            load_moves(valid_moves) if isinstance(valid_moves, Path) else [move.copy() for move in valid_moves]
        )

        validate_board(loaded_board)
        validate_move_paths(loaded_moves, loaded_board)

        selected_spangram_indexes = spangram_indexes or set()

        self._board = [list(row) for row in loaded_board]
        self._initial_board = loaded_board.copy()
        self._remaining_feedback: dict[tuple[BoardCoord, ...], Highlight] = {}
        self._expected_move_count = len(loaded_moves)
        self._render_config = render_config or RenderConfig(theme=LIGHT_THEME)
        if self._render_config.theme is None:
            self._render_config = replace(self._render_config, theme=LIGHT_THEME)
        self._word_coords: set[BoardCoord] = set()
        self._spangram_coords: set[BoardCoord] = set()
        self._coord_by_pixel: dict[PixelCoord, BoardCoord] = {}
        self._cached_screenshot: bytes | None = None
        self._cached_centers: dict[BoardCoord, PixelCoord] | None = None
        self._is_dirty = True

        for move_idx, move in enumerate(loaded_moves):
            move_key = tuple(move)
            feedback = Highlight.SPANGRAM if move_idx in selected_spangram_indexes else Highlight.WORD
            self._remaining_feedback[move_key] = feedback

    @property
    def expected_move_count(self) -> int:
        """Return expected number of accepted moves."""
        return self._expected_move_count

    @property
    def initial_board(self) -> list[str]:
        """Return the immutable initial board configured for this driver."""
        return self._initial_board.copy()

    def corner_cell_centers(self) -> tuple[PixelCoord, PixelCoord]:
        """Return top-left and bottom-right rendered cell centers.

        Returns:
            Tuple of ((0, 0) center, (rows-1, cols-1) center).

        """
        if self._cached_centers is None:
            self.capture_screen()
        if self._cached_centers is None:
            msg = "unable to resolve rendered cell centers"
            raise ValueError(msg)

        row_count = len(self._board)
        col_count = len(self._board[0])
        top_left = self._cached_centers[(0, 0)]
        bottom_right = self._cached_centers[(row_count - 1, col_count - 1)]
        return top_left, bottom_right

    def capture_screen(self) -> bytes:
        """Generate PNG bytes for current board-state snapshot.

        Returns:
            PNG-encoded screenshot bytes.

        """
        if not self._is_dirty and self._cached_screenshot is not None and self._cached_centers is not None:
            self._coord_by_pixel = {pixel: coord for coord, pixel in self._cached_centers.items()}
            return self._cached_screenshot

        png_bytes, cell_centers = render_board_png(
            ["".join(row) for row in self._board],
            word_coords=self._word_coords,
            spangram_coords=self._spangram_coords,
            config=self._render_config,
        )
        self._cached_screenshot = png_bytes
        self._cached_centers = cell_centers
        self._is_dirty = False
        self._coord_by_pixel = {pixel: coord for coord, pixel in cell_centers.items()}
        return png_bytes

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        """Apply a move encoded as pixel centers if configured as valid.

        Args:
            pixel_path: Path of pixel centers corresponding to cell centers.

        Raises:
            ValueError: If the path is empty or contains unknown pixels.

        """
        if not pixel_path:
            msg = "pixel_path must contain at least one coordinate"
            raise ValueError(msg)

        move_coords: list[BoardCoord] = []
        for pixel in pixel_path:
            coord = self._coord_by_pixel.get(pixel)
            if coord is None:
                nearest_pixel, nearest_coord = min(
                    self._coord_by_pixel.items(),
                    key=lambda item: abs(item[0][0] - pixel[0]) + abs(item[0][1] - pixel[1]),
                )
                nearest_distance = abs(nearest_pixel[0] - pixel[0]) + abs(nearest_pixel[1] - pixel[1])
                if nearest_distance > PIXEL_COORDINATE_MATCH_TOLERANCE_PX:
                    msg = f"pixel not found in current board centers: {pixel}"
                    raise ValueError(msg)
                coord = nearest_coord
            move_coords.append(coord)

        move_key = tuple(move_coords)
        feedback = self._remaining_feedback.pop(move_key, None)
        if feedback is None:
            return

        highlight_coords = self._spangram_coords if feedback is Highlight.SPANGRAM else self._word_coords
        for row_idx, col_idx in move_coords:
            highlight_coords.add((row_idx, col_idx))

        self._is_dirty = True

    def tap(self, coord: PixelCoord) -> None:
        """No-op tap for the fake driver.

        Args:
            coord: Pixel coordinate to tap (ignored).

        """
