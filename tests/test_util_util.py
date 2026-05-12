from pathlib import Path
from random import Random

import pytest

from strands_solver.util.util import (
    BLOCKED_CELL,
    coords_to_word,
    dump_board,
    generate_random_board,
    load_allowed_words,
    load_board,
    load_moves,
    print_board,
    validate_board,
    validate_move_paths,
)


def test_load_allowed_words_filters_by_min_length(tmp_path: Path) -> None:
    words_path = tmp_path / "allowed.txt"
    words_path.write_text("cat\nPear\nAPPLE\nMelOn\n")

    result = load_allowed_words(words_path)

    assert result == ["pear", "apple", "melon"]


def test_load_allowed_words_includes_exact_min_length(tmp_path: Path) -> None:
    words_path = tmp_path / "allowed.txt"
    words_path.write_text("pear\nplum\nberry\n")

    result = load_allowed_words(words_path)

    assert result == ["pear", "plum", "berry"]


def test_load_board_strips_line_whitespace(tmp_path: Path) -> None:
    board_path = tmp_path / "board.txt"
    board_path.write_text(" TEST \n AbCd\nRATE  \n WxYz \n")

    result = load_board(board_path)

    assert result == ["test", "abcd", "rate", "wxyz"]


def test_validate_board_rejects_too_few_rows() -> None:
    board = ["abcd", "efgh", "ijkl"]

    with pytest.raises(ValueError, match="less than minimum"):
        validate_board(board)


def test_validate_board_rejects_empty_board() -> None:
    with pytest.raises(ValueError, match="less than minimum"):
        validate_board([])


def test_validate_board_rejects_short_row() -> None:
    board = ["abcd", "efg", "ijkl", "mnop"]

    with pytest.raises(ValueError, match="less than minimum word length"):
        validate_board(board)


def test_validate_board_rejects_row_length_mismatch() -> None:
    board = ["abcd", "efgh", "ijklm", "mnop"]

    with pytest.raises(ValueError, match="different from previous row"):
        validate_board(board)


def test_validate_board_accepts_valid_board() -> None:
    board = ["abcd", "efgh", "ijkl", "mnop"]

    validate_board(board)


def test_validate_board_rejects_too_many_rows() -> None:
    board = ["abcd"] * 11

    with pytest.raises(ValueError, match="more than maximum"):
        validate_board(board)


def test_validate_board_rejects_too_many_cols() -> None:
    board = ["abcdefghijk"] * 4

    with pytest.raises(ValueError, match="greater than maximum"):
        validate_board(board)


def test_validate_board_rejects_invalid_characters() -> None:
    board = ["abc1", "efgh", "ijkl", "mnop"]

    with pytest.raises(ValueError, match="invalid characters"):
        validate_board(board)


def test_load_moves_ignores_blank_lines_and_parses_coords(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("\n[(0,0),(0,1)]\n\n[(1,1),(1,2)]\n")

    result = load_moves(moves_path)

    assert result == [[(0, 0), (0, 1)], [(1, 1), (1, 2)]]


def test_load_moves_returns_empty_for_empty_file(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("")

    assert load_moves(moves_path) == []


def test_load_moves_returns_empty_for_blank_only_lines(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("\n  \n\t\n")

    assert load_moves(moves_path) == []


def test_load_moves_rejects_invalid_python_syntax(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("(0,0) (0,1)\n")

    with pytest.raises(ValueError, match="not valid Python syntax"):
        load_moves(moves_path)


def test_load_moves_rejects_non_list_literal(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("{(0,0),(0,1)}\n")

    with pytest.raises(TypeError, match="must be a Python list/tuple"):
        load_moves(moves_path)


def test_load_moves_rejects_coord_with_wrong_tuple_size(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("[(0,0,0)]\n")

    with pytest.raises(TypeError, match="invalid coord"):
        load_moves(moves_path)


def test_load_moves_rejects_coord_that_is_not_tuple(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("[(0,0),(1,1)]\n")

    assert load_moves(moves_path) == [[(0, 0), (1, 1)]]


def test_load_moves_rejects_coord_with_non_int_values(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("[(0,'1')]\n")

    with pytest.raises(TypeError, match="invalid coord"):
        load_moves(moves_path)


def test_load_moves_accepts_tuple_move_and_list_coords(tmp_path: Path) -> None:
    moves_path = tmp_path / "moves.txt"
    moves_path.write_text("([0,0],(0,1),[0,2],(0,3))\n")

    result = load_moves(moves_path)

    assert result == [[(0, 0), (0, 1), (0, 2), (0, 3)]]


def test_validate_move_paths_rejects_out_of_bounds_coords() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    moves = [[(0, 0), (0, 1), (0, 2), (0, 4)]]

    with pytest.raises(ValueError, match="outside board bounds"):
        validate_move_paths(moves, board)


def test_validate_move_paths_rejects_out_of_bounds_row() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    moves = [[(4, 0)]]

    with pytest.raises(ValueError, match="outside board bounds"):
        validate_move_paths(moves, board)


def test_validate_move_paths_rejects_negative_row() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    moves = [[(-1, 0)]]

    with pytest.raises(ValueError, match="outside board bounds"):
        validate_move_paths(moves, board)


def test_validate_move_paths_rejects_negative_col() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    moves = [[(0, -1)]]

    with pytest.raises(ValueError, match="outside board bounds"):
        validate_move_paths(moves, board)


def test_validate_move_paths_rejects_duplicate_coords() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    moves = [[(0, 0), (0, 1), (0, 1), (0, 3)]]

    with pytest.raises(ValueError, match="duplicate coord"):
        validate_move_paths(moves, board)


def test_validate_move_paths_accepts_valid_paths() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    moves = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
    ]

    result = validate_move_paths(moves, board)
    assert result is None


def test_validate_move_paths_accepts_tuple_move_and_list_coords() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    moves = [[(0, 0), (0, 1), (0, 2), (0, 3)]]

    result = validate_move_paths(moves, board)
    assert result is None


def test_dump_board_writes_board_text(tmp_path: Path) -> None:
    board_path = tmp_path / "board.txt"

    dump_board(["abcd", "efgh"], board_path, separator=" ")

    assert board_path.read_text() == "a b c d\ne f g h\n"


def test_print_board_prints_board_text(capsys: pytest.CaptureFixture[str]) -> None:
    print_board(["abcd", "efgh"], separator=" ")

    captured = capsys.readouterr()
    assert captured.out == "a b c d\ne f g h\n"


def test_coords_to_word_returns_word_for_path() -> None:
    board = ["test", "abcd", "rate", "wxyz"]
    coords = [(0, 0), (0, 1), (0, 2), (0, 3)]

    assert coords_to_word(board, coords) == "test"


def test_coords_to_word_returns_empty_for_empty_path() -> None:
    board = ["test", "abcd", "rate", "wxyz"]

    assert coords_to_word(board, []) == ""


def test_generate_random_board_returns_requested_shape() -> None:
    board = generate_random_board(rows=4, cols=5, rng=Random(0))  # noqa: S311

    assert len(board) == 4
    assert all(len(row) == 5 for row in board)
    assert all(BLOCKED_CELL not in row for row in board)


def test_generate_random_board_can_include_blocked_cells() -> None:
    board = generate_random_board(rows=4, cols=4, include_blocked=True, rng=Random(1))  # noqa: S311

    assert len(board) == 4
    assert all(len(row) == 4 for row in board)
    assert all(all(char.islower() or char == BLOCKED_CELL for char in row) for row in board)


@pytest.mark.parametrize(("rows", "cols"), [(3, 4), (4, 3), (11, 4), (4, 11)])
def test_generate_random_board_rejects_invalid_dimensions(rows: int, cols: int) -> None:
    with pytest.raises(ValueError, match="between 4 and 10"):
        generate_random_board(rows=rows, cols=cols)
