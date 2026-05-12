from importlib import import_module
from typing import Any, Self, cast

import pytest

from strands_solver.board_reader.board_reader import BoardState, Highlight
from strands_solver.board_reader.board_reader_tesseract_open_cv import BoardReaderTesseractOpenCv

cv2 = cast("Any", import_module("cv2"))
np = cast("Any", import_module("numpy"))


def _state_grid(first: Highlight, second: Highlight) -> list[list[Highlight]]:
    grid = [[Highlight.NONE for _ in range(6)] for _ in range(8)]
    grid[0][0] = first
    grid[0][1] = second
    return grid


def test_tesseract_open_cv_reader_extract_state_raises_for_empty_screenshot() -> None:
    reader = BoardReaderTesseractOpenCv()

    with pytest.raises(ValueError, match="screenshot cannot be empty"):
        reader.extract_state(b"")


def test_tesseract_open_cv_reader_classify_feedback_detects_word_blue() -> None:
    reader = BoardReaderTesseractOpenCv()
    before = BoardState(
        board=["??????"] * 8,
        cell_states=_state_grid(Highlight.NONE, Highlight.NONE),
    )
    after = BoardState(
        board=["??????"] * 8,
        cell_states=_state_grid(Highlight.WORD, Highlight.WORD),
    )

    result = reader.classify_feedback(before, after, [(0, 0), (0, 1)])

    assert result == Highlight.WORD


def test_tesseract_open_cv_reader_classify_feedback_detects_spangram_yellow() -> None:
    reader = BoardReaderTesseractOpenCv()
    before = BoardState(
        board=["??????"] * 8,
        cell_states=_state_grid(Highlight.NONE, Highlight.NONE),
    )
    after = BoardState(
        board=["??????"] * 8,
        cell_states=_state_grid(Highlight.SPANGRAM, Highlight.SPANGRAM),
    )

    result = reader.classify_feedback(before, after, [(0, 0), (0, 1)])

    assert result == Highlight.SPANGRAM


def test_tesseract_open_cv_reader_classify_feedback_rejects_low_saturation_change() -> None:
    reader = BoardReaderTesseractOpenCv()
    before = BoardState(
        board=["??????"] * 8,
        cell_states=_state_grid(Highlight.WORD, Highlight.NONE),
    )
    after = BoardState(
        board=["??????"] * 8,
        cell_states=_state_grid(Highlight.WORD, Highlight.NONE),
    )

    result = reader.classify_feedback(before, after, [(0, 0), (0, 1)])

    assert result == Highlight.NONE


class _MockTessBaseAPI:
    """Minimal context-manager stub for tesserocr.PyTessBaseAPI."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.images: list[object] = []
        self.variables: dict[str, str] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def SetVariable(self, key: str, value: str) -> None:  # noqa: N802
        self.variables[key] = value

    def SetImage(self, img: object) -> None:  # noqa: N802
        self.images.append(img)

    def GetUTF8Text(self) -> str:  # noqa: N802
        return next(self._responses, "")

    def MeanTextConf(self) -> int:  # noqa: N802
        return 80


class _MockPSM:
    SINGLE_CHAR = 10


class _MockTesserocr:
    """Stub tesserocr module."""

    PSM = _MockPSM()

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.api_instance: _MockTessBaseAPI | None = None

    def PyTessBaseAPI(self, *, psm: object = None, lang: str = "eng") -> _MockTessBaseAPI:  # noqa: N802
        self.api_instance = _MockTessBaseAPI(self._responses)
        return self.api_instance


class _MockPilImage:
    """Stub PIL.Image module."""

    @staticmethod
    def fromarray(arr: object) -> object:
        return arr


def _make_numpy_image(rows: int, cols: int) -> object:
    """Create a small solid-colour BGR numpy image."""

    return np.full((rows, cols, 3), 128, dtype=np.uint8)


def test_extract_board_rows_assembles_rows_from_mocked_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """_extract_board_rows should turn per-cell OCR text into board rows."""
    letters = ["a", "b", "c", "d", "e", "f"] + [""] * 10
    mock_tesserocr = _MockTesserocr(letters)
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_ocr_module", staticmethod(lambda: mock_tesserocr))
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_pil", staticmethod(_MockPilImage))

    reader = BoardReaderTesseractOpenCv(rows=4, cols=4)
    image = _make_numpy_image(400, 400)
    board = reader._ocr_board(image, (0, 0, 400, 400))

    assert board[0] == "abcd"
    assert board[1] == "ef??"


def test_extract_board_rows_falls_back_to_placeholder_on_empty_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cells returning empty OCR text should use the placeholder character."""
    mock_tesserocr = _MockTesserocr([""] * 16 + ["b"] + ["c"] * 24)
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_ocr_module", staticmethod(lambda: mock_tesserocr))
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_pil", staticmethod(_MockPilImage))

    reader = BoardReaderTesseractOpenCv(rows=4, cols=4)
    image = _make_numpy_image(400, 400)
    board = reader._ocr_board(image, (0, 0, 400, 400))

    assert board[0][0] == "?"
    assert board[0][1] == "b"


def test_ocr_cell_returns_first_lowercased_character() -> None:
    """_ocr_cell should strip, lowercase, and return only the first char."""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    api = _MockTessBaseAPI(["A\n"])
    pil = _MockPilImage()

    result, confidence = BoardReaderTesseractOpenCv._ocr_cell(image, (10, 10, 90, 90), api, pil)

    assert result == "a"
    assert confidence >= 0


def test_ocr_cell_returns_placeholder_on_no_text() -> None:
    """_ocr_cell should return placeholder when OCR produces nothing."""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    api = _MockTessBaseAPI(["   "])
    pil = _MockPilImage()

    result, confidence = BoardReaderTesseractOpenCv._ocr_cell(image, (10, 10, 90, 90), api, pil)

    assert result == "?"
    assert confidence >= -1


def test_ocr_cell_returns_empty_string_for_zero_size_crop() -> None:
    """_ocr_cell should return empty string when the crop region is empty."""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    api = _MockTessBaseAPI(["x"])
    pil = _MockPilImage()

    result, confidence = BoardReaderTesseractOpenCv._ocr_cell(image, (50, 50, 50, 90), api, pil)

    assert result == ""
    assert confidence == -1


def test_extract_board_rows_sets_char_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ocr_board should configure the lowercase-only character whitelist."""
    mock_tesserocr = _MockTesserocr(["a"] * 16)
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_ocr_module", staticmethod(lambda: mock_tesserocr))
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_pil", staticmethod(_MockPilImage))

    reader = BoardReaderTesseractOpenCv(rows=4, cols=4)
    image = _make_numpy_image(400, 400)
    reader._ocr_board(image, (0, 0, 400, 400))

    assert mock_tesserocr.api_instance is not None
    assert mock_tesserocr.api_instance.variables.get("tessedit_char_whitelist") == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert mock_tesserocr.api_instance.variables.get("load_system_dawg") == "0"
    assert mock_tesserocr.api_instance.variables.get("load_freq_dawg") == "0"


def test_tesseract_open_cv_reader_init_raises_for_rows_below_minimum() -> None:
    with pytest.raises(ValueError, match="rows and cols must both be"):
        BoardReaderTesseractOpenCv(rows=1, cols=6)


def test_tesseract_open_cv_reader_init_raises_for_cols_below_minimum() -> None:
    with pytest.raises(ValueError, match="rows and cols must both be"):
        BoardReaderTesseractOpenCv(rows=8, cols=2)


def test_tesseract_open_cv_reader_init_rejects_placeholder_override() -> None:
    reader_ctor = cast("Any", BoardReaderTesseractOpenCv)
    with pytest.raises(TypeError, match="placeholder_char"):
        reader_ctor(placeholder_char="??")


class _FakeCV2:
    """Minimal cv2 stub that decodes images produced by _real_png()."""

    IMREAD_COLOR = 1
    COLOR_BGR2GRAY = 6
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 2

    def __init__(self, image_array: object) -> None:
        self._image = image_array

    def imdecode(self, buf: object, flags: int) -> object:
        return self._image

    def cvtColor(self, img: object, code: int) -> object:  # noqa: N802
        return np.mean(cast("Any", img), axis=2).astype(np.uint8)

    def GaussianBlur(self, img: object, ksize: object, sigma: float) -> object:  # noqa: N802
        return img

    def Canny(self, img: object, lo: float, hi: float) -> object:  # noqa: N802
        return np.zeros_like(cast("Any", img))

    def findContours(self, img: object, mode: int, method: int) -> tuple[list[object], None]:  # noqa: N802
        return [], None

    def boundingRect(self, contour: object) -> tuple[int, int, int, int]:  # noqa: N802
        return (0, 0, 10, 10)


def _real_png() -> tuple[bytes, object]:
    """Return a small solid-colour PNG and the numpy array it encodes."""
    arr = np.full((400, 300, 3), 128, dtype=np.uint8)

    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return bytes(buf), arr


def test_tesseract_open_cv_reader_extract_state_raises_for_invalid_image_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extract_state should raise ValueError when cv2 cannot decode the image."""
    fake_cv2 = _FakeCV2(None)
    fake_np = np

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_cv_modules", staticmethod(lambda: (fake_cv2, fake_np)))
    reader = BoardReaderTesseractOpenCv()

    with pytest.raises(ValueError, match="not a valid image payload"):
        reader.extract_state(b"not-a-real-image")


def test_tesseract_open_cv_reader_extract_state_returns_board_state_via_mocked_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extract_state should return a BoardState when given a valid PNG + mocked OCR."""
    _png_bytes, arr = _real_png()
    fake_cv2 = _FakeCV2(arr)

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_cv_modules", staticmethod(lambda: (fake_cv2, np)))
    monkeypatch.setattr(
        BoardReaderTesseractOpenCv,
        "_extract_board_rows",
        lambda self, image: ["abcdef"] * self._rows,
    )

    reader = BoardReaderTesseractOpenCv()
    state = reader.extract_state(b"fakepng")

    assert len(state.board) == reader._rows
    assert state.cell_states is not None


def test_tesseract_open_cv_reader_extract_state_reuses_cached_state_for_identical_screenshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _png_bytes, arr = _real_png()
    fake_cv2 = _FakeCV2(arr)

    ocr_calls = 0

    def fake_ocr(self: BoardReaderTesseractOpenCv, image: object, board_rect: tuple[int, int, int, int]) -> list[str]:
        nonlocal ocr_calls
        _ = image, board_rect
        ocr_calls += 1
        return ["abcdef"] * self._rows

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_cv_modules", staticmethod(lambda: (fake_cv2, np)))
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_ocr_board", fake_ocr)

    reader = BoardReaderTesseractOpenCv()
    first = reader.extract_state(b"same-shot")
    second = reader.extract_state(b"same-shot")

    assert first.board == second.board
    assert ocr_calls == 1


def test_tesseract_open_cv_reader_extract_state_skips_ocr_when_selection_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _png_bytes, arr = _real_png()
    fake_cv2 = _FakeCV2(arr)

    ocr_calls = 0

    def fake_ocr(self: BoardReaderTesseractOpenCv, image: object, board_rect: tuple[int, int, int, int]) -> list[str]:
        nonlocal ocr_calls
        _ = image, board_rect
        ocr_calls += 1
        return ["abcdef"] * self._rows

    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_load_cv_modules", staticmethod(lambda: (fake_cv2, np)))
    monkeypatch.setattr(BoardReaderTesseractOpenCv, "_ocr_board", fake_ocr)

    reader = BoardReaderTesseractOpenCv()
    first = reader.extract_state(b"first-frame")
    second = reader.extract_state(b"second-frame")

    assert first.board == second.board
    assert ocr_calls == 1


def test_tesseract_open_cv_reader_compute_cell_centers_covers_all_cells() -> None:
    """_compute_cell_centers should produce one entry per grid cell."""
    reader = BoardReaderTesseractOpenCv(rows=4, cols=6)
    centers = reader._compute_cell_centers((0, 0, 600, 400))

    assert len(centers) == 4
    assert all(len(row) == 6 for row in centers)
    assert isinstance(centers[0][0][0], int)
    assert isinstance(centers[3][5][1], int)


def test_tesseract_open_cv_reader_sample_cell_colors_returns_bgr_for_each_center() -> None:
    """_sample_cell_colors should return one BGR tuple per center coordinate."""
    image = np.full((200, 200, 3), 100, dtype=np.uint8)
    centers = [[(50, 50), (75, 75)], [(125, 125), (150, 150)]]
    colors = BoardReaderTesseractOpenCv._sample_cell_colors(image, centers)

    assert len(colors) == 2
    assert all(len(row) == 2 for row in colors)
    for row in colors:
        for bgr in row:
            assert len(bgr) == 3


def test_tesseract_open_cv_reader_estimate_board_rect_returns_fallback_for_blank_image() -> None:
    """_estimate_board_rect should return the fallback rect when no contours are found."""
    reader = BoardReaderTesseractOpenCv()
    blank = np.zeros((800, 600, 3), dtype=np.uint8)
    rect = reader._estimate_board_rect_cv(blank, cv2)

    assert rect[0] == int(600 * 0.13)
    assert rect[1] == int(800 * 0.275)
    assert rect[2] == int(600 * 0.74)
    assert rect[3] == int(800 * 0.53)


def test_selected_coords_from_states_returns_only_highlighted_cells() -> None:
    states: list[list[Highlight]] = [
        [Highlight.NONE, Highlight.WORD, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.SPANGRAM, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
    ]

    selected = BoardReaderTesseractOpenCv._selected_coords_from_states(states)

    assert selected == {(0, 1), (2, 2)}


def test_get_cell_state_rejects_neutral_gray() -> None:
    reader = BoardReaderTesseractOpenCv()

    assert reader._get_cell_state((128, 128, 128)) == Highlight.NONE
