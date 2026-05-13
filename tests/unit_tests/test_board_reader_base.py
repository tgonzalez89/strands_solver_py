import pytest

from strands_solver.board_reader.board_reader import BoardState, Highlight
from strands_solver.board_reader.board_reader_base import BoardReaderBase


class _DummyReader(BoardReaderBase):
    def __init__(
        self,
        *,
        rows: int = 4,
        cols: int = 4,
        board_rows: list[str] | None = None,
        cell_states: list[list[Highlight]] | None = None,
    ) -> None:
        super().__init__(rows=rows, cols=cols)
        self.decode_calls = 0
        self.center_calls = 0
        self.state_calls = 0
        self.board_calls = 0
        self.board_rows = board_rows or ["abcd", "efgh", "ijkl", "mnop"]
        self.cell_states = cell_states or [[Highlight.NONE for _ in range(cols)] for _ in range(rows)]

    def _decode_image(self, screenshot: bytes) -> object:
        self.decode_calls += 1
        return screenshot

    def _extract_cell_states(self, image: object) -> list[list[Highlight]]:
        _ = image
        self.state_calls += 1
        return [row.copy() for row in self.cell_states]

    def _extract_cell_centers(self, image: object) -> list[list[tuple[int, int]]]:
        _ = image
        self.center_calls += 1
        return [[(row * 10, col * 10) for col in range(self._cols)] for row in range(self._rows)]

    def _extract_board_rows(self, image: object) -> list[str]:
        _ = image
        self.board_calls += 1
        return self.board_rows.copy()


def test_board_reader_base_extract_state_caches_same_screenshot() -> None:
    reader = _DummyReader()
    screenshot = b"same"

    first = reader.extract_state(screenshot)
    second = reader.extract_state(screenshot)

    assert isinstance(first, BoardState)
    assert second is first
    assert reader.decode_calls == 1
    assert reader.center_calls == 1
    assert reader.state_calls == 1
    assert reader.board_calls == 1


def test_board_reader_base_reuses_previous_board_when_cell_states_unchanged() -> None:
    reader = _DummyReader()

    first = reader.extract_state(b"one")
    second = reader.extract_state(b"two")

    assert first.board == second.board
    assert reader.decode_calls == 2
    assert reader.board_calls == 1


def test_board_reader_base_reextracts_board_when_cell_states_change() -> None:
    reader = _DummyReader()

    first = reader.extract_state(b"one")
    reader.cell_states[0][0] = Highlight.WORD
    second = reader.extract_state(b"two")

    assert first.cell_states != second.cell_states
    assert reader.board_calls == 2


def test_board_reader_base_rejects_empty_screenshot() -> None:
    reader = _DummyReader()

    with pytest.raises(ValueError, match="screenshot cannot be empty"):
        reader.extract_state(b"")


def test_board_reader_base_classify_feedback_handles_all_outcomes() -> None:
    reader = _DummyReader()
    before_states = [[Highlight.NONE] * 4 for _ in range(4)]
    before = BoardState(board=["abcd"] * 4, cell_states=before_states)
    after_word_states = [[Highlight.NONE] * 4 for _ in range(4)]
    after_word_states[0][0] = Highlight.WORD
    after_word = BoardState(board=["abcd"] * 4, cell_states=after_word_states)

    after_spangram_states = [[Highlight.NONE] * 4 for _ in range(4)]
    after_spangram_states[0][1] = Highlight.SPANGRAM
    after_spangram = BoardState(board=["abcd"] * 4, cell_states=after_spangram_states)

    assert reader.classify_feedback(before, after_word, [(0, 0)]) == Highlight.WORD
    assert reader.classify_feedback(before, after_spangram, [(0, 1)]) == Highlight.SPANGRAM
    assert reader.classify_feedback(before, after_word, []) == Highlight.NONE


def test_board_reader_base_classify_feedback_rejects_preselected_move_cells() -> None:
    reader = _DummyReader()
    before_states = [[Highlight.NONE] * 4 for _ in range(4)]
    before_states[0][0] = Highlight.WORD
    before = BoardState(board=["abcd"] * 4, cell_states=before_states)
    after = BoardState(board=["abcd"] * 4, cell_states=[[Highlight.WORD] * 4 for _ in range(4)])

    assert reader.classify_feedback(before, after, [(0, 0)]) == Highlight.NONE


def test_board_reader_base_move_to_pixel_path_requires_geometry() -> None:
    reader = _DummyReader()

    with pytest.raises(ValueError, match="call extract_state first"):
        reader.board_move_to_pixel_path([(0, 0)])


def test_board_reader_base_move_to_pixel_path_uses_last_cell_centers() -> None:
    reader = _DummyReader()
    reader.extract_state(b"image")

    pixel_path = reader.board_move_to_pixel_path([(0, 0), (1, 1)])

    assert pixel_path == [(0, 0), (10, 10)]
