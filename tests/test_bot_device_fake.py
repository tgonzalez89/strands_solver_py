import pytest

from strands_solver.board_reader.board_reader import Highlight
from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCv
from strands_solver.bot.bot_device_fake import BotDeviceFake
from strands_solver.device.device_driver_fake import DeviceDriverFake
from strands_solver.image_renderer.board_image_renderer import RenderConfig
from strands_solver.util.util import BoardCoord, PixelCoord


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


def _moves() -> list[list[BoardCoord]]:
    return [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ]


def _render_board_rect() -> tuple[int, int, int, int]:
    cfg = RenderConfig()
    board_width = int(cfg.width * cfg.board_width_ratio)
    board_height = int(cfg.height * cfg.board_height_ratio)
    board_left = (cfg.width - board_width) // 2
    board_top = int(cfg.height * cfg.board_center_y_ratio - board_height / 2)
    return (board_left, board_top, board_width, board_height)


def _renderer_centers(
    reader: BoardReaderTesseractOpenCv, board_rect: tuple[int, int, int, int]
) -> list[list[PixelCoord]]:
    x, y, width, height = board_rect
    cell_width = width / reader._cols
    cell_height = height / reader._rows
    return [
        [(int(x + (col + 0.5) * cell_width), int(y + (row + 0.5) * cell_height)) for col in range(reader._cols)]
        for row in range(reader._rows)
    ]


def test_fake_device_bot_get_board_uses_tesseract_open_cv_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()
    move_paths = _moves()

    ocr_calls: list[int] = []

    def fake_ocr_board(
        self: BoardReaderTesseractOpenCv,
        image: object,
    ) -> list[str]:
        _ = image
        ocr_calls.append(len(ocr_calls))
        return board

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_extract_board_rows", fake_ocr_board)
    monkeypatch.setattr(
        BoardReaderTesseractOpenCv,
        "_estimate_board_rect",
        lambda self, image: _render_board_rect(),
    )
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_compute_cell_centers", _renderer_centers)

    bot = BotDeviceFake(board=board, valid_moves=move_paths, mode="light")

    observed = bot.get_board()

    assert observed == board
    assert len(ocr_calls) == 1


def test_fake_device_driver_plus_reader_classifies_word_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()
    move = _moves()[0]
    driver = DeviceDriverFake(board=board, valid_moves=[move], mode="light")
    reader = BoardReaderTesseractOpenCv(rows=8, cols=6)

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_extract_board_rows", lambda self, image: board)
    monkeypatch.setattr(
        BoardReaderTesseractOpenCv,
        "_estimate_board_rect",
        lambda self, image: _render_board_rect(),
    )
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_compute_cell_centers", _renderer_centers)

    before_png = driver.capture_screen()
    before_state = reader.extract_state(before_png)

    centers = {coord: pixel for pixel, coord in driver._coord_by_pixel.items()}
    driver.execute_path([centers[coord] for coord in move])

    after_png = driver.capture_screen()
    after_state = reader.extract_state(after_png)

    result = reader.classify_feedback(before_state, after_state, move)

    assert result == Highlight.WORD


def test_fake_device_driver_plus_reader_classifies_spangram_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()
    move = _moves()[0]
    driver = DeviceDriverFake(board=board, valid_moves=[move], spangram_indexes={0}, mode="dark")
    reader = BoardReaderTesseractOpenCv(rows=8, cols=6)

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_extract_board_rows", lambda self, image: board)
    monkeypatch.setattr(
        BoardReaderTesseractOpenCv,
        "_estimate_board_rect",
        lambda self, image: _render_board_rect(),
    )
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_compute_cell_centers", _renderer_centers)

    before_png = driver.capture_screen()
    before_state = reader.extract_state(before_png)

    centers = {coord: pixel for pixel, coord in driver._coord_by_pixel.items()}
    driver.execute_path([centers[coord] for coord in move])

    after_png = driver.capture_screen()
    after_state = reader.extract_state(after_png)

    result = reader.classify_feedback(before_state, after_state, move)

    assert result == Highlight.SPANGRAM


def test_fake_device_bot_apply_move_uses_reader_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()
    move = _moves()[0]

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_extract_board_rows", lambda self, image: board)
    monkeypatch.setattr(
        BoardReaderTesseractOpenCv,
        "_estimate_board_rect",
        lambda self, image: _render_board_rect(),
    )
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_compute_cell_centers", _renderer_centers)

    bot = BotDeviceFake(board=board, valid_moves=[move], mode="light")

    assert bot.apply_move(move) is True


def test_fake_device_bot_raises_when_initial_ocr_does_not_match_board(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()
    move = _moves()[0]
    mismatched = list(board)
    mismatched[0] = "xbcdef"

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_extract_board_rows", lambda self, image: mismatched)
    monkeypatch.setattr(
        BoardReaderTesseractOpenCv,
        "_estimate_board_rect",
        lambda self, image: _render_board_rect(),
    )
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_compute_cell_centers", _renderer_centers)

    with pytest.raises(ValueError, match="Initial OCR board does not match configured fake board"):
        BotDeviceFake(board=board, valid_moves=[move], mode="light")
