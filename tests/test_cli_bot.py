from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import strands_solver.cli.bot as cli

if TYPE_CHECKING:
    from strands_solver.solver.solver import Trie


class _TrieStub:
    def __init__(self, paths: list[list[tuple[int, int]]] | None = None) -> None:
        self._paths = paths or []

    def find_all_word_paths(self, board: list[str]) -> list[list[tuple[int, int]]]:
        _ = board
        return self._paths


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "allowed_words": Path("words.txt"),
        "driver": "file",
        "board": None,
        "valid_moves": None,
        "spangram_index": [],
        "fake_mode": "light",
        "verbose": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_run_appium_mode_rejects_board_and_moves() -> None:
    args = _args(driver="appium", board=Path("board.txt"), valid_moves=Path("moves.txt"))

    result = cli._run_appium_mode(args, cast("Trie", _TrieStub()))

    assert result == 2


def test_run_appium_mode_handles_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(driver="appium")

    class _Bot:
        def __init__(self, driver: object, reader: object) -> None:
            _ = driver, reader

        def run(self, trie: object, *, verbose: bool = False) -> list[tuple[str, list[tuple[int, int]]]]:
            _ = trie, verbose
            msg = "not configured"
            raise NotImplementedError(msg)

    monkeypatch.setattr(cli, "BotDevice", _Bot)

    result = cli._run_appium_mode(args, cast("Trie", _TrieStub()))

    assert result == 2


def test_run_fake_mode_requires_board_and_moves() -> None:
    args = _args(driver="fake", board=None, valid_moves=None)

    result = cli._run_fake_mode(args, cast("Trie", _TrieStub()))

    assert result == 2


def test_run_fake_mode_success(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(driver="fake", board=Path("board.txt"), valid_moves=Path("moves.txt"), spangram_index=[0])

    class _FakeBot:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs
            self.expected_move_count = 1

        def run(self, trie: object, *, verbose: bool = False) -> list[tuple[str, list[tuple[int, int]]]]:
            _ = trie, verbose
            return [("word", [(0, 0), (0, 1), (0, 2), (0, 3)])]

        def get_board(self) -> list[str]:
            return ["abcd", "efgh", "ijkl", "mnop"]

    monkeypatch.setattr(cli, "BotDeviceFake", _FakeBot)

    result = cli._run_fake_mode(args, cast("Trie", _TrieStub()))

    assert result == 0


def test_run_fake_mode_handles_initial_ocr_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(driver="fake", board=Path("board.txt"), valid_moves=Path("moves.txt"))

    class _FakeBot:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs
            msg = "mismatch details"
            raise cli.InitialOcrMismatchError(msg)

    monkeypatch.setattr(cli, "BotDeviceFake", _FakeBot)

    result = cli._run_fake_mode(args, cast("Trie", _TrieStub()))

    assert result == 2


def test_run_file_mode_requires_moves() -> None:
    args = _args(driver="file", board=Path("board.txt"), valid_moves=None)

    result = cli._run_file_mode(args, cast("Trie", _TrieStub()))

    assert result == 2


def test_run_file_mode_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(driver="file", board=Path("board.txt"), valid_moves=Path("moves.txt"))

    class _Bot:
        def __init__(self, board: Path, moves: Path) -> None:
            _ = board, moves

        def run(self, trie: object, *, verbose: bool = False) -> list[tuple[str, list[tuple[int, int]]]]:
            _ = trie, verbose
            return [("word", [(0, 0), (0, 1), (0, 2), (0, 3)])]

        def get_board(self) -> list[str]:
            return ["abcd", "efgh", "ijkl", "mnop"]

    monkeypatch.setattr(cli, "load_moves", lambda path: [[(0, 0), (0, 1), (0, 2), (0, 3)]])
    monkeypatch.setattr(cli, "BotFake", _Bot)

    result = cli._run_file_mode(args, cast("Trie", _TrieStub()))

    assert result == 0


def test_main_dispatches_to_file_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda argv=None: _args(driver="file", board=Path("board.txt"), valid_moves=Path("moves.txt")),
    )
    monkeypatch.setattr(cli, "load_allowed_words", lambda path: ["word"])
    monkeypatch.setattr(cli, "load_moves", lambda path: [[(0, 0), (0, 1), (0, 2), (0, 3)]])

    class _TrieFactory:
        @staticmethod
        def build_from_words(words: list[str]) -> _TrieStub:
            _ = words
            return _TrieStub(paths=[])

    monkeypatch.setattr(cli, "Trie", _TrieFactory)

    class _Bot:
        def __init__(self, board: Path, moves: list[list[tuple[int, int]]]) -> None:
            _ = board, moves

        def run(self, trie: object, *, verbose: bool = False) -> list[tuple[str, list[tuple[int, int]]]]:
            _ = trie, verbose
            return []

        def get_board(self) -> list[str]:
            return ["abcd", "efgh", "ijkl", "mnop"]

    monkeypatch.setattr(cli, "BotFake", _Bot)

    result = cli.main([])

    assert result == 0


def test_main_dispatches_to_appium_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "parse_args", lambda argv=None: _args(driver="appium"))
    monkeypatch.setattr(cli, "load_allowed_words", lambda path: ["word"])

    class _TrieFactory:
        @staticmethod
        def build_from_words(words: list[str]) -> _TrieStub:
            _ = words
            return _TrieStub()

    class _Bot:
        def __init__(self, driver: object, reader: object) -> None:
            _ = driver, reader

        def run(self, trie: object, *, verbose: bool = False) -> list[tuple[str, list[tuple[int, int]]]]:
            _ = trie, verbose
            return []

    monkeypatch.setattr(cli, "Trie", _TrieFactory)
    monkeypatch.setattr(cli, "BotDevice", _Bot)

    result = cli.main([])

    assert result == 0
