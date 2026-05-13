"""Fake device-backed bot for OCR and OpenCV integration testing."""

from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCV
from strands_solver.bot.bot_device import BotDevice
from strands_solver.device.device_driver_fake import DeviceDriverFake

if TYPE_CHECKING:
    from pathlib import Path

    from strands_solver.image_renderer.board_image_renderer import RenderConfig
    from strands_solver.util.util import BoardCoord


class BotDeviceFake(BotDevice):
    """Device bot backed by generated screenshots and real OCR reader."""

    def __init__(
        self,
        board: list[str] | Path,
        valid_moves: list[list[BoardCoord]] | Path,
        *,
        spangram_indexes: set[int] | None = None,
        mode: str = "light",
        render_config: RenderConfig | None = None,
    ) -> None:
        """Initialize a fake-device bot for OCR integration testing.

        Args:
            board: Board rows or path to board fixture.
            valid_moves: Valid move paths or path to move fixture.
            spangram_indexes: Move indexes classified as spangram.
            mode: Render mode for generated screenshots.
            render_config: Optional rendering configuration.

        """
        self._device_driver_fake = DeviceDriverFake(
            board=board,
            valid_moves=valid_moves,
            spangram_indexes=spangram_indexes,
            mode=mode,
            render_config=render_config,
        )
        initial_board = self._device_driver_fake.initial_board
        super().__init__(
            driver=self._device_driver_fake,
            reader=BoardReaderTesseractOpenCV(rows=len(initial_board), cols=len(initial_board[0])),
        )
        self._validate_initial_ocr_matches_expected_board()

    @property
    def expected_move_count(self) -> int:
        """Return expected number of successful moves from configured fixture."""
        return self._device_driver_fake.expected_move_count

    def _validate_initial_ocr_matches_expected_board(self) -> None:
        expected = [row.lower() for row in self._device_driver_fake.initial_board]
        observed = self.get_board()

        if observed == expected:
            return

        # Collect mismatch details for error message if OCR output does not match expected board.
        mismatch_count = 0
        mismatch_preview: list[str] = []
        for row_idx, (expected_row, observed_row) in enumerate(zip(expected, observed, strict=True), start=1):
            if expected_row == observed_row:
                continue

            for col_idx, (expected_char, observed_char) in enumerate(
                zip(expected_row, observed_row, strict=True),
                start=1,
            ):
                if expected_char != observed_char:
                    mismatch_count += 1
                    mismatch_preview.append(
                        f"(r{row_idx},c{col_idx}): expected '{expected_char}' got '{observed_char}'",
                    )

        details = ", ".join(mismatch_preview) if mismatch_preview else "row-level mismatch"
        msg = f"Initial OCR board does not match configured fake board ({mismatch_count} mismatched cells). {details}"
        raise InitialOcrMismatchError(
            msg,
        )


class InitialOcrMismatchError(ValueError):
    """Raised when fake-mode initial OCR output does not match configured board."""
