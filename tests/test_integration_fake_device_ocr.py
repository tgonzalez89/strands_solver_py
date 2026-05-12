import os
from importlib import import_module
from pathlib import Path

import pytest

from strands_solver.board_reader.board_reader import Highlight
from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCv
from strands_solver.bot.bot_device_fake import BotDeviceFake
from strands_solver.bot.bot_fake import BotFake
from strands_solver.device.device_driver_fake import DeviceDriverFake
from strands_solver.image_renderer.board_image_renderer import render_board_png
from strands_solver.solver.solver import Trie


def _ensure_tesserocr_ready() -> None:
    if "TESSDATA_PREFIX" not in os.environ:
        candidates = [
            "/usr/share/tesseract-ocr/5/tessdata",
            "/usr/share/tessdata",
            "/usr/local/share/tessdata",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                os.environ["TESSDATA_PREFIX"] = candidate
                break

    try:
        tesserocr = import_module("tesserocr")

        with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_CHAR, lang="eng"):
            pass
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"tesserocr runtime unavailable for integration tests: {error}")


def _board() -> list[str]:
    return [
        "acfghj",
        "jklmnr",
        "stuvwx",
        "yzacfh",
        "ghjklm",
        "nrstuv",
        "wxyzac",
        "cfghjk",
    ]


def _move() -> list[tuple[int, int]]:
    return [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_integration_fake_device_bot_apply_move_word_no_mocks() -> None:
    _ensure_tesserocr_ready()

    board = _board()
    move = _move()
    bot = BotDeviceFake(board=board, valid_moves=[move], mode="light")

    observed = bot.get_board()

    assert len(observed) == 8
    assert all(len(row) == 6 for row in observed)
    assert bot.apply_move(move) is True


def test_integration_fake_driver_reader_classifies_word_no_mocks() -> None:
    _ensure_tesserocr_ready()

    board = _board()
    move = _move()
    driver = DeviceDriverFake(board=board, valid_moves=[move], mode="light")
    reader = BoardReaderTesseractOpenCv(rows=8, cols=6)

    before_state = reader.extract_state(driver.capture_screen())

    pixel_path = reader.board_move_to_pixel_path(move)
    driver.execute_path(pixel_path)

    after_state = reader.extract_state(driver.capture_screen())
    result = reader.classify_feedback(before_state, after_state, move)

    assert result == Highlight.WORD


def test_integration_fake_driver_reader_classifies_spangram_no_mocks() -> None:
    _ensure_tesserocr_ready()

    board = _board()
    move = _move()
    driver = DeviceDriverFake(board=board, valid_moves=[move], spangram_indexes={0}, mode="dark")
    reader = BoardReaderTesseractOpenCv(rows=8, cols=6)

    before_state = reader.extract_state(driver.capture_screen())

    pixel_path = reader.board_move_to_pixel_path(move)
    driver.execute_path(pixel_path)

    after_state = reader.extract_state(driver.capture_screen())
    result = reader.classify_feedback(before_state, after_state, move)

    assert result == Highlight.SPANGRAM


def test_integration_fake_mode_matches_file_mode_hardcoded_no_mocks() -> None:
    _ensure_tesserocr_ready()

    board = [
        "acfhzz",
        "zzzzzz",
        "zzzzzz",
        "zzzzzz",
        "zzzzzz",
        "zzzzzz",
        "zzzzzz",
        "zzzzzz",
    ]
    moves = [[(0, 0), (0, 1), (0, 2), (0, 3)]]

    trie_for_file = Trie.build_from_words(["acfh"])
    trie_for_fake = Trie.build_from_words(["acfh"])

    file_bot = BotFake(board=board, valid_moves=moves)
    fake_bot = BotDeviceFake(board=board, valid_moves=moves, mode="light")

    file_result = file_bot.run(trie_for_file)
    fake_result = fake_bot.run(trie_for_fake)

    assert fake_result == file_result


@pytest.mark.parametrize(
    ("mode", "board"),
    [
        (
            "light",
            [
                "acfghj",
                "jklmnr",
                "stuvwx",
                "yzacfh",
                "ghjklm",
                "nrstuv",
                "wxyzac",
                "cfghjk",
            ],
        ),
        (
            "dark",
            [
                "rstuvw",
                "xyzacf",
                "fghjkl",
                "mnrstu",
                "vwxyza",
                "acfghj",
                "klmnrs",
                "tuvwxy",
            ],
        ),
        (
            "light",
            [
                "ghjklm",
                "nrstuv",
                "wxyzac",
                "cfghjk",
                "lmnrst",
                "uvwxyz",
                "acfghj",
                "jklmnr",
            ],
        ),
        (
            "dark",
            [
                "acfghj",
                "jklmnr",
                "stuvwx",
                "yzacfh",
                "ghjklm",
                "nrstuv",
                "wxyzac",
                "cfghjk",
            ],
        ),
    ],
)
def test_integration_rendered_board_roundtrip_ocr_matches(mode: str, board: list[str]) -> None:
    _ensure_tesserocr_ready()

    screenshot, _ = render_board_png(board, mode=mode)
    reader = BoardReaderTesseractOpenCv(rows=8, cols=6)

    state = reader.extract_state(screenshot)

    assert state.board == board


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_integration_rendered_highlighted_board_roundtrip_ocr_matches(mode: str) -> None:
    _ensure_tesserocr_ready()

    board = [
        "acfghj",
        "jklmnr",
        "stuvwx",
        "yzacfh",
        "ghjklm",
        "nrstuv",
        "wxyzac",
        "cfghjk",
    ]
    word_coords = {(0, 0), (0, 1), (1, 1), (2, 1)}
    spangram_coords = {(3, 3), (3, 4), (4, 4), (5, 4)}

    screenshot, _ = render_board_png(
        board,
        mode=mode,
        word_coords=word_coords,
        spangram_coords=spangram_coords,
    )
    reader = BoardReaderTesseractOpenCv(rows=8, cols=6)

    state = reader.extract_state(screenshot)

    selected = word_coords | spangram_coords
    for row_idx in range(8):
        for col_idx in range(6):
            observed = state.board[row_idx][col_idx]
            expected = board[row_idx][col_idx]
            if (row_idx, col_idx) in selected:
                assert observed.isalpha()
            else:
                assert observed == expected
