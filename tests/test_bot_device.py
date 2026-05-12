from strands_solver.board_reader.board_reader import BoardReader, BoardState, Highlight
from strands_solver.bot.bot_device import BotDevice
from strands_solver.device.device_driver import DeviceDriver
from strands_solver.util.util import BoardCoord, PixelCoord


class MockDriver(DeviceDriver):
    def __init__(self, screens: list[bytes]) -> None:
        self._screens = screens
        self._capture_idx = 0
        self.executed_paths: list[list[PixelCoord]] = []

    def capture_screen(self) -> bytes:
        screenshot = self._screens[self._capture_idx]
        if self._capture_idx < len(self._screens) - 1:
            self._capture_idx += 1
        return screenshot

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        self.executed_paths.append(pixel_path)


class MockReader(BoardReader):
    def __init__(self, states_by_screen: dict[bytes, BoardState], feedback: Highlight) -> None:
        self._states_by_screen = states_by_screen
        self._feedback = feedback
        self.feedback_calls: list[tuple[BoardState, BoardState, list[BoardCoord]]] = []
        self._cell_centers: dict[BoardCoord, PixelCoord] = {
            (0, 0): (100, 200),
            (0, 1): (200, 200),
        }

    def extract_state(self, screenshot: bytes) -> BoardState:
        return self._states_by_screen[screenshot]

    def classify_feedback(self, before: BoardState, after: BoardState, move: list[BoardCoord]) -> Highlight:
        self.feedback_calls.append((before, after, move.copy()))
        return self._feedback

    def board_move_to_pixel_path(self, move: list[BoardCoord]) -> list[PixelCoord]:
        return [self._cell_centers[coord] for coord in move]


def _state(board: list[str]) -> BoardState:
    return BoardState(board=board)


def _state_with_cells(board: list[str], cell_states: list[list[Highlight]]) -> BoardState:
    return BoardState(board=board, cell_states=cell_states)


def test_device_bot_apply_move_executes_pixel_path_and_accepts_word_feedback() -> None:
    before = _state(["test", "abcd", "rate", "wxyz"])
    after = _state(["##st", "abcd", "rate", "wxyz"])
    driver = MockDriver([b"before", b"after"])
    reader = MockReader({b"before": before, b"after": after}, feedback=Highlight.WORD)
    bot = BotDevice(driver=driver, reader=reader)
    move: list[BoardCoord] = [(0, 0), (0, 1)]

    assert bot.apply_move(move) is True
    assert driver.executed_paths == [[(100, 200), (200, 200)]]
    assert len(reader.feedback_calls) == 1


def test_device_bot_apply_move_accepts_spangram_feedback() -> None:
    before = _state(["test", "abcd", "rate", "wxyz"])
    after = _state(["##st", "abcd", "rate", "wxyz"])
    driver = MockDriver([b"before", b"after"])
    reader = MockReader({b"before": before, b"after": after}, feedback=Highlight.SPANGRAM)
    bot = BotDevice(driver=driver, reader=reader)

    assert bot.apply_move([(0, 0), (0, 1)]) is True


def test_device_bot_apply_move_rejects_invalid_feedback() -> None:
    before = _state(["test", "abcd", "rate", "wxyz"])
    after = _state(["test", "abcd", "rate", "wxyz"])
    driver = MockDriver([b"before", b"after"])
    reader = MockReader({b"before": before, b"after": after}, feedback=Highlight.NONE)
    bot = BotDevice(driver=driver, reader=reader)

    assert bot.apply_move([(0, 0), (0, 1)]) is False


def test_device_bot_get_board_reads_from_driver_reader_pipeline() -> None:
    state = _state(["test", "abcd", "rate", "wxyz"])
    driver = MockDriver([b"screen"])
    reader = MockReader({b"screen": state}, feedback=Highlight.NONE)
    bot = BotDevice(driver=driver, reader=reader)

    assert bot.get_board() == ["test", "abcd", "rate", "wxyz"]


def test_device_bot_get_board_masks_using_cell_states() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    cell_states = [
        [Highlight.WORD, Highlight.SPANGRAM, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
    ]
    state = _state_with_cells(board, cell_states)
    driver = MockDriver([b"screen"])
    reader = MockReader({b"screen": state}, feedback=Highlight.NONE)
    bot = BotDevice(driver=driver, reader=reader)

    assert bot.get_board() == ["##st", "abcd", "rate", "wxyz"]
