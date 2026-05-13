import pytest

from strands_solver.board_reader import board_reader_tesseract_open_cv as board_reader_module
from strands_solver.board_reader.board_reader import BoardState
from strands_solver.bot.bot_device_fake import BotDeviceFake, InitialOcrMismatchError


def _board() -> list[str]:
    return [
        "abcdef",
        "ghijkl",
        "mnopqr",
        "stuvwx",
        "yzabcd",
        "efghij",
        "klmnop",
        "qrstuv",
    ]


def _moves() -> list[list[tuple[int, int]]]:
    return [[(0, 0), (0, 1), (0, 2), (0, 3)], [(1, 0), (1, 1), (1, 2), (1, 3)]]


def test_bot_device_fake_accepts_matching_initial_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()

    def fake_extract_state(self: object, screenshot: bytes) -> BoardState:
        _ = self, screenshot
        return BoardState(board=board)

    monkeypatch.setattr(board_reader_module.BoardReaderTesseractOpenCV, "extract_state", fake_extract_state)

    bot = BotDeviceFake(board=board, valid_moves=_moves(), mode="light")

    assert bot.expected_move_count == 2


def test_bot_device_fake_raises_on_initial_ocr_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()
    mismatched = list(board)
    mismatched[0] = "xbcdef"

    def fake_extract_state(self: object, screenshot: bytes) -> BoardState:
        _ = self, screenshot
        return BoardState(board=mismatched)

    monkeypatch.setattr(board_reader_module.BoardReaderTesseractOpenCV, "extract_state", fake_extract_state)

    with pytest.raises(InitialOcrMismatchError, match="Initial OCR board does not match configured fake board"):
        BotDeviceFake(board=board, valid_moves=_moves(), mode="light")
