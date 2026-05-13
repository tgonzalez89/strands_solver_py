import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from strands_solver.board_reader.board_reader_tesseract_open_cv import TESSDATA_DIR
from strands_solver.bot.bot_device_fake import BotDeviceFake
from strands_solver.image_renderer.board_image_renderer import RenderConfig
from strands_solver.solver.solver import Trie
from strands_solver.util.util import BLOCKED_CELL, coords_to_word

if TYPE_CHECKING:
    import pytest


def _resolve_tessdata_path() -> str:
    candidates = [
        os.environ.get("TESSDATA_PREFIX"),
        TESSDATA_DIR,
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if (candidate_path / "eng.traineddata").exists():
            return str(candidate_path)

    msg = "No valid tessdata directory found (missing eng.traineddata)."
    raise RuntimeError(msg)


def _ensure_tesserocr_ready() -> None:
    try:
        tesserocr = import_module("tesserocr")
    except ModuleNotFoundError as error:
        msg = f"tesserocr is required for this integration test: {error}"
        raise RuntimeError(msg) from error

    tessdata_path = _resolve_tessdata_path()
    os.environ["TESSDATA_PREFIX"] = tessdata_path

    try:
        with tesserocr.PyTessBaseAPI(path=tessdata_path, psm=tesserocr.PSM.SINGLE_CHAR, lang="eng"):
            pass
    except Exception as error:  # pragma: no cover - environment dependent
        msg = f"tesserocr runtime unavailable for integration tests: {error}"
        raise RuntimeError(msg) from error


def test_integration_bot_device_fake_solves_full_board_with_verbose_output(capsys: pytest.CaptureFixture[str]) -> None:
    _ensure_tesserocr_ready()
    board = [
        "iaonud",
        "nrgtbg",
        "eabhte",
        "vicesa",
        "eshrel",
        "pneoda",
        "exaflb",
        "inpafe",
    ]
    valid_moves = [
        [(7, 0), (7, 1), (6, 0), (6, 1), (5, 0), (4, 0), (5, 1), (4, 1), (3, 1), (3, 0), (2, 0)],
        [(2, 2), (2, 1), (1, 1), (1, 2), (0, 1), (0, 0), (1, 0)],
        [(0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)],
        [(1, 4), (0, 4), (0, 5), (1, 5), (2, 5), (2, 4)],
        [(3, 4), (3, 5), (4, 5), (4, 4)],
        [(7, 3), (7, 4), (6, 3), (5, 3), (4, 3), (5, 4), (5, 5), (6, 5), (6, 4), (7, 5)],
    ]

    required_words = {coords_to_word(board, move) for move in valid_moves}
    extra_words = {"cheap", "gain", "expensive", "able", "afford", "budge", "arga", "sive", "pensive", "ford"}
    allowed_words = sorted(required_words | extra_words)

    bot = BotDeviceFake(
        board=board,
        valid_moves=valid_moves,
        spangram_indexes={2},
        mode="light",
        render_config=RenderConfig(),
    )
    trie = Trie.build_from_words(allowed_words)

    successful_moves = bot.run(trie, verbose=True)

    assert len(successful_moves) == len(valid_moves)
    assert bot.expected_move_count == len(valid_moves)

    solved_board = bot.get_board()
    assert len(solved_board) == 8
    assert all(len(row) == 6 for row in solved_board)
    assert all(char == BLOCKED_CELL for row in solved_board for char in row)

    printed = capsys.readouterr().out
    assert "[VERBOSE] Finding paths for board:" in printed
    assert "[VERBOSE] Trying candidate word" in printed
    assert "[VERBOSE] Move accepted." in printed
