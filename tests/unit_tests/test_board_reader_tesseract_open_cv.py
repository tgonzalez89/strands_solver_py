from typing import Any, cast

import numpy as np
import pytest

import strands_solver.board_reader.board_reader_tesseract_open_cv as board_reader_module
from strands_solver.board_reader.board_reader import Highlight


def test_reader_init_raises_when_extras_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(board_reader_module, "HAS_EXTRAS", False)

    with pytest.raises(NotImplementedError, match="Extra dependencies not found"):
        board_reader_module.BoardReaderTesseractOpenCV()


def test_patch_image_side_returns_even_length() -> None:
    side_default = board_reader_module.BoardReaderTesseractOpenCV._patch_image_side()
    side_smaller = board_reader_module.BoardReaderTesseractOpenCV._patch_image_side(0.5)

    assert side_default % 2 == 0
    assert side_smaller % 2 == 0
    assert side_smaller < side_default


def test_classify_cell_color_and_state_detect_word_color() -> None:
    word_rgb = board_reader_module.WORD_COLOR
    word_bgr = (word_rgb[2], word_rgb[1], word_rgb[0])
    cell_img = np.full((12, 12, 3), word_bgr, dtype=np.uint8)

    detected_color = board_reader_module.BoardReaderTesseractOpenCV._classify_cell_color(cell_img)
    detected_state = board_reader_module.BoardReaderTesseractOpenCV._classify_cell_state(cell_img)

    assert detected_color == board_reader_module.WORD_COLOR
    assert detected_state == Highlight.WORD


class _FakeTessApi:
    def __init__(self, text: str = "P") -> None:
        self._text = text
        self._psm = 7
        self._whitelist = "ABC"

    def GetPageSegMode(self) -> int:  # noqa: N802
        return self._psm

    def GetStringVariable(self, key: str) -> str:  # noqa: N802
        _ = key
        return self._whitelist

    def SetPageSegMode(self, value: int) -> None:  # noqa: N802
        self._psm = value

    def SetVariable(self, key: str, value: str) -> None:  # noqa: N802
        if key == "tessedit_char_whitelist":
            self._whitelist = value

    def SetImage(self, image: object) -> None:  # noqa: N802
        _ = image

    def GetUTF8Text(self) -> str:  # noqa: N802
        return self._text


def test_ocr_d_fallback_restores_psm_and_whitelist() -> None:
    api = _FakeTessApi(text="P")
    cell_img = np.zeros((8, 8), dtype=np.uint8)

    result = board_reader_module.BoardReaderTesseractOpenCV._ocr_d_cell_fallback(cast("Any", api), cell_img)

    assert result == "P"
    assert api.GetPageSegMode() == 7
    assert api.GetStringVariable("tessedit_char_whitelist") == "ABC"


def test_extract_board_rows_orchestrates_preprocess_and_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = board_reader_module.BoardReaderTesseractOpenCV(rows=8, cols=6)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    board_img = np.full((10, 10), 255, dtype=np.uint8)

    calls: dict[str, bool] = {"preprocess": False, "ocr": False, "imwrite": False}

    def fake_preprocess(img: object) -> np.ndarray:
        _ = img
        calls["preprocess"] = True
        return board_img

    def fake_ocr(img: object) -> list[str]:
        _ = img
        calls["ocr"] = True
        return ["ABCDEF"] * 8

    def fake_imwrite(path: str, img: object) -> bool:
        _ = img
        calls["imwrite"] = path == "debug_board.png"
        return True

    monkeypatch.setattr(reader, "_preprocess_board_image", fake_preprocess)
    monkeypatch.setattr(reader, "_ocr_board_rows", fake_ocr)
    cv2_any = cast("Any", board_reader_module).cv2
    monkeypatch.setattr(cv2_any, "imwrite", fake_imwrite)

    rows = reader._extract_board_rows(image)

    assert calls == {"preprocess": True, "ocr": True, "imwrite": True}
    assert rows == ["ABCDEF"] * 8
