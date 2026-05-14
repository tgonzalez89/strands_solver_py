"""Device-backed bot implementation."""

from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader import BoardReader, BoardState, Highlight
from strands_solver.bot.bot import Bot
from strands_solver.util.util import BLOCKED_CELL, validate_board

if TYPE_CHECKING:
    from strands_solver.device.device_driver import DeviceDriver
    from strands_solver.util.util import BoardCoord


class BotDevice(Bot):
    """Concrete bot for real-device or emulator integrations."""

    def __init__(self, driver: DeviceDriver, reader: BoardReader) -> None:
        """Initialize the device bot with a driver and board reader.

        Args:
            driver: Device interaction implementation.
            reader: Screenshot parser and feedback classifier.

        """
        self._driver = driver
        self._reader = reader
        self._state: BoardState | None = None

    def refresh_state(self) -> BoardState:
        """Capture and parse the latest board state.

        Returns:
            Most recent board state snapshot.

        Raises:
            ValueError: If extracted board violates board validation rules.

        """
        screenshot = self._driver.capture_screen()
        state = self._reader.extract_state(screenshot)
        validate_board(self._solver_board_from_state(state))
        self._state = state
        return state

    def get_board(self) -> list[str]:
        """Return board rows from cached or refreshed device state.

        Returns:
            Current board rows.

        """
        state = self._state if self._state is not None else self.refresh_state()
        return self._solver_board_from_state(state)

    @staticmethod
    def _solver_board_from_state(state: BoardState) -> list[str]:
        """Convert a BoardState into the solver's expected board format.

        It is a list of lowercase strings with blocked cells replaced by BLOCKED_CELL.
        """
        if state.cell_states is None:
            return [row.lower() for row in state.board]

        board_chars = [list(row.lower()) for row in state.board]
        for row_idx, row_states in enumerate(state.cell_states):
            for col_idx, highlight in enumerate(row_states):
                if highlight == Highlight.NONE:
                    continue
                if row_idx >= len(board_chars) or col_idx >= len(board_chars[row_idx]):
                    continue
                board_chars[row_idx][col_idx] = BLOCKED_CELL

        return ["".join(row) for row in board_chars]

    def apply_move(self, move: list[BoardCoord]) -> bool:
        """Execute and verify a move against the target device.

        Args:
            move: Coordinate path to execute.

        Returns:
            True if move execution is verified; otherwise False.

        """
        before_state = self._state if self._state is not None else self.refresh_state()
        pixel_path = self._reader.board_move_to_pixel_path(move)
        self._driver.execute_path(pixel_path)
        after_state = self.refresh_state()
        feedback = self._reader.classify_feedback(before_state, after_state, move)
        return feedback in {Highlight.WORD, Highlight.SPANGRAM}
