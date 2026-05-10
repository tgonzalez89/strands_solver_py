from pathlib import Path

import pytest

from strands_solver.bot import StrandsGameBotTest
from strands_solver.solver import Coord, Trie


def test_run_applies_valid_moves_from_direct_api() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    test_path: list[Coord] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    rate_path: list[Coord] = [(2, 0), (2, 1), (2, 2), (2, 3)]
    trie = Trie.build_from_words(["test", "rate"])
    bot = StrandsGameBotTest(board, [test_path, rate_path])

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
    bot = StrandsGameBotTest(board_path, moves_path)

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
    bot = StrandsGameBotTest(board, [[(1, 0), (1, 1), (1, 2), (1, 3)]])

    assert bot.run(trie) == []


def test_apply_move_rejects_unknown_move_without_mutating_board() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    valid: list[list[Coord]] = [[(0, 0), (0, 1), (0, 2), (0, 3)]]
    invalid: list[Coord] = [(3, 0), (3, 1), (3, 2), (3, 3)]
    bot = StrandsGameBotTest(board, valid)

    assert bot.apply_move(invalid) is False
    assert bot.get_board() == board


def test_apply_move_cannot_apply_same_move_twice() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    valid_move: list[Coord] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    bot = StrandsGameBotTest(board, [valid_move])

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
    bot = StrandsGameBotTest(board, [[(0, 0), (0, 1), (0, 2), (0, 3)]])

    assert bot.run(trie) == []


def test_bot_direct_input_validates_move_paths() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    invalid_moves: list[list[Coord]] = [[(0, 0), (0, 1), (0, 1), (0, 3)]]

    with pytest.raises(ValueError, match="duplicate coord"):
        StrandsGameBotTest(board, invalid_moves)


def test_bot_direct_input_validates_board() -> None:
    invalid_board = ["ab", "cd", "ef", "gh"]
    moves: list[list[Coord]] = [[(0, 0), (0, 1), (1, 0), (1, 1)]]

    with pytest.raises(ValueError, match="less than minimum word length"):
        StrandsGameBotTest(invalid_board, moves)
