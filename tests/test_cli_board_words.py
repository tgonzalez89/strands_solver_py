from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import strands_solver.cli.board_words as cli

if TYPE_CHECKING:
    import pytest


class _TrieStub:
    def __init__(self, paths: list[list[tuple[int, int]]] | None = None) -> None:
        self._paths = paths or []

    def find_all_word_paths(self, board: list[str]) -> list[list[tuple[int, int]]]:
        _ = board
        return self._paths


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "allowed_words": Path("words.txt"),
        "board": Path("board.txt"),
    }
    values.update(overrides)
    return Namespace(**values)


def test_main_prints_all_board_words(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "parse_args", lambda argv=None: _args())
    monkeypatch.setattr(cli, "load_allowed_words", lambda path: ["abcd"])
    monkeypatch.setattr(cli, "load_board", lambda path: ["abcd", "efgh", "ijkl", "mnop"])

    class _TrieFactory:
        @staticmethod
        def build_from_words(words: list[str]) -> _TrieStub:
            _ = words
            return _TrieStub(paths=[[(0, 0), (0, 1), (0, 2), (0, 3)]])

    monkeypatch.setattr(cli, "Trie", _TrieFactory)

    result = cli.main([])

    captured = capsys.readouterr()
    assert result == 0
    assert "abcd: [(0, 0), (0, 1), (0, 2), (0, 3)]" in captured.out
