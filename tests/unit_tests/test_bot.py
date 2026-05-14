from typing import TYPE_CHECKING

import pytest

from strands_solver.board_reader.board_reader import BoardReader, BoardState, Highlight
from strands_solver.bot.bot_device import BotDevice
from strands_solver.bot.bot_fake import BotFake
from strands_solver.device.device_driver import DeviceDriver
from strands_solver.solver.solver import Trie

if TYPE_CHECKING:
    from pathlib import Path

    from strands_solver.util.util import BoardCoord, PixelCoord


class _FakeDriver(DeviceDriver):
    def __init__(self) -> None:
        self.screens = [b"screen", b"screen"]
        self.executed_paths: list[list[PixelCoord]] = []

    def capture_screen(self) -> bytes:
        return self.screens[0]

    def tap(self, coord: PixelCoord) -> None:
        _ = coord

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        self.executed_paths.append(pixel_path)


class _FakeReader(BoardReader):
    def __init__(self, states: list[list[str]], *, verify_result: bool = True) -> None:
        self._states = states
        self._state_idx = 0
        self._feedback = Highlight.WORD if verify_result else Highlight.NONE
        self._cell_centers: dict[BoardCoord, PixelCoord] = {
            (0, 0): (100, 200),
            (0, 1): (200, 200),
            (0, 2): (300, 200),
            (0, 3): (400, 200),
        }

    def extract_state(self, screenshot: bytes) -> BoardState:
        _ = screenshot
        return BoardState(self._states[self._state_idx])

    def classify_feedback(self, before: BoardState, after: BoardState, move: list[BoardCoord]) -> Highlight:
        _ = before, move
        if self._state_idx < len(self._states) - 1 and after.board == self._states[self._state_idx + 1]:
            self._state_idx += 1
        return self._feedback

    def board_move_to_pixel_path(self, move: list[BoardCoord]) -> list[PixelCoord]:
        return [self._cell_centers[coord] for coord in move]


class BotDeviceDouble(BotDevice):
    def __init__(self, states: list[list[str]], *, verify_result: bool = True) -> None:
        self._driver = _FakeDriver()
        self._reader = _FakeReader(states, verify_result=verify_result)
        super().__init__(driver=self._driver, reader=self._reader)


class LoggingBotFake(BotFake):
    def __init__(self, board: list[str], valid_moves: list[list[BoardCoord]]) -> None:
        super().__init__(board, valid_moves)
        self.attempted_moves: list[list[BoardCoord]] = []

    def apply_move(self, move: list[BoardCoord]) -> bool:
        self.attempted_moves.append(move.copy())
        return super().apply_move(move)


def test_run_applies_valid_moves_from_direct_api() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    test_path: list[BoardCoord] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    rate_path: list[BoardCoord] = [(2, 0), (2, 1), (2, 2), (2, 3)]
    trie = Trie.build_from_words(["test", "rate"])
    bot = BotFake(board, [test_path, rate_path])

    result = bot.run(trie)

    assert result == [("test", test_path), ("rate", rate_path)]
    assert bot.get_board() == [
        "####",
        "abcd",
        "####",
        "wxyz",
    ]


def test_run_loads_board_and_moves_from_files(tmp_path: Path) -> None:
    board_path = tmp_path / "board.txt"
    moves_path = tmp_path / "moves.txt"

    board_path.write_text("test\nabcd\nrate\nwxyz\n")
    moves_path.write_text("[(0,0),(0,1),(0,2),(0,3)]\n[(2,0),(2,1),(2,2),(2,3)]\n")

    trie = Trie.build_from_words(["test", "rate"])
    bot = BotFake(board_path, moves_path)

    result = bot.run(trie)

    assert result == [
        ("test", [(0, 0), (0, 1), (0, 2), (0, 3)]),
        ("rate", [(2, 0), (2, 1), (2, 2), (2, 3)]),
    ]


def test_run_returns_empty_when_no_move_matches() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    trie = Trie.build_from_words(["test"])
    bot = BotFake(board, [[(1, 0), (1, 1), (1, 2), (1, 3)]])

    assert bot.run(trie) == []


def test_apply_move_rejects_unknown_move_without_mutating_board() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    valid: list[list[BoardCoord]] = [[(0, 0), (0, 1), (0, 2), (0, 3)]]
    invalid: list[BoardCoord] = [(3, 0), (3, 1), (3, 2), (3, 3)]
    bot = BotFake(board, valid)

    assert bot.apply_move(invalid) is False
    assert bot.get_board() == board


def test_apply_move_cannot_apply_same_move_twice() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    valid_move: list[BoardCoord] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    bot = BotFake(board, [valid_move])

    assert bot.apply_move(valid_move) is True
    assert bot.apply_move(valid_move) is False


def test_run_returns_empty_when_trie_has_no_candidates() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    trie = Trie.build_from_words(["zzzz"])
    bot = BotFake(board, [[(0, 0), (0, 1), (0, 2), (0, 3)]])

    assert bot.run(trie) == []


def test_run_fallback_finds_move_without_dictionary_word() -> None:
    board = [
        "a###",
        "#b##",
        "##c#",
        "###d",
    ]
    abcd_path: list[BoardCoord] = [(0, 0), (1, 1), (2, 2), (3, 3)]
    trie = Trie.build_from_words(["zzzz"])
    bot = BotFake(board, [abcd_path])

    result = bot.run(trie)

    assert result == [("abcd", abcd_path)]
    assert bot.get_board() == ["####", "####", "####", "####"]


def test_run_fallback_runs_after_dictionary_exhausted_with_spaces_left() -> None:
    board = [
        "test",
        "a###",
        "#b##",
        "##cd",
    ]
    test_path: list[BoardCoord] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    abcd_path: list[BoardCoord] = [(1, 0), (2, 1), (3, 2), (3, 3)]
    trie = Trie.build_from_words(["test"])
    bot = BotFake(board, [test_path, abcd_path])

    result = bot.run(trie)

    assert result == [("test", test_path), ("abcd", abcd_path)]


def test_run_fallback_respects_diagonal_walls_from_accepted_moves() -> None:
    board = [
        "a###",
        "#be#",
        "xfc#",
        "ghxd",
    ]
    abcd_path: list[BoardCoord] = [(0, 0), (1, 1), (2, 2), (3, 3)]
    crossing_path: list[BoardCoord] = [(1, 2), (2, 1), (3, 0), (3, 1)]
    trie = Trie.build_from_words(["abcd"])
    bot = LoggingBotFake(board, [abcd_path])

    result = bot.run(trie)

    assert result == [("abcd", abcd_path)]
    assert crossing_path not in bot.attempted_moves


def test_bot_direct_input_validates_move_paths() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    invalid_moves: list[list[BoardCoord]] = [[(0, 0), (0, 1), (0, 1), (0, 3)]]

    with pytest.raises(ValueError, match="duplicate coord"):
        BotFake(board, invalid_moves)


def test_bot_direct_input_validates_board() -> None:
    invalid_board = ["ab", "cd", "ef", "gh"]
    moves: list[list[BoardCoord]] = [[(0, 0), (0, 1), (1, 0), (1, 1)]]

    with pytest.raises(ValueError, match="less than minimum word length"):
        BotFake(invalid_board, moves)


def test_device_bot_get_board_uses_extracted_state() -> None:
    bot = BotDeviceDouble(
        [
            ["test", "abcd", "rate", "wxyz"],
        ],
    )

    assert bot.get_board() == ["test", "abcd", "rate", "wxyz"]


def test_device_bot_apply_move_returns_true_when_verified() -> None:
    bot = BotDeviceDouble(
        [
            ["test", "abcd", "rate", "wxyz"],
            ["####", "abcd", "rate", "wxyz"],
        ],
        verify_result=True,
    )
    move: list[BoardCoord] = [(0, 0), (0, 1), (0, 2), (0, 3)]

    assert bot.apply_move(move) is True
    assert isinstance(bot._driver, _FakeDriver)
    assert bot._driver.executed_paths == [[(100, 200), (200, 200), (300, 200), (400, 200)]]


def test_device_bot_apply_move_returns_false_when_verification_fails() -> None:
    bot = BotDeviceDouble(
        [
            ["test", "abcd", "rate", "wxyz"],
            ["####", "abcd", "rate", "wxyz"],
        ],
        verify_result=False,
    )
    move: list[BoardCoord] = [(0, 0), (0, 1), (0, 2), (0, 3)]

    assert bot.apply_move(move) is False


def test_board_state_supports_optional_cell_states() -> None:
    states: list[list[Highlight]] = [
        [Highlight.WORD, Highlight.SPANGRAM, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
        [Highlight.NONE, Highlight.NONE, Highlight.NONE, Highlight.NONE],
    ]
    state = BoardState(
        board=["test", "abcd", "rate", "wxyz"],
        cell_states=states,
    )

    assert state.cell_states == states


def test_board_state_cell_states_optional() -> None:
    state = BoardState(board=["test", "abcd", "rate", "wxyz"])

    assert state.cell_states is None


def test_run_prioritizes_longer_words_first() -> None:
    board = [
        "tests",
        "abcde",
        "fghij",
        "klmno",
        "pqrst",
    ]
    test_path: list[BoardCoord] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    tests_path: list[BoardCoord] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    trie = Trie.build_from_words(["test", "tests"])
    bot = BotFake(board, [test_path, tests_path])

    result = bot.run(trie)

    assert result[0] == ("tests", tests_path)
