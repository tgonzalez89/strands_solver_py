"""Input/output helpers and validation utilities for solver data files."""

import ast
from random import Random
from string import ascii_lowercase
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

type BoardCoord = tuple[int, int]
type PixelCoord = tuple[int, int]

MIN_WORD_LEN: Final = 4
MIN_BOARD_DIMENSION: Final = MIN_WORD_LEN
MAX_BOARD_DIMENSION: Final = 10
COORD_PARTS_COUNT: Final = 2
BLOCKED_CELL: Final = "#"
VALID_BOARD_CHARS: Final = frozenset(ascii_lowercase + BLOCKED_CELL)


def load_allowed_words(allowed_words_path: Path) -> list[str]:
    """Load and normalize allowed words from disk.

    Args:
        allowed_words_path: Path to file with one word per line.

    Returns:
        Lowercased words with length greater than or equal to `MIN_WORD_LEN`.
    """
    result: list[str] = []

    with allowed_words_path.open() as f:
        for line in f:
            word = line.strip().lower()
            if len(word) >= MIN_WORD_LEN:
                result.append(word)

    return result


def load_board(board_path: Path) -> list[str]:
    """Load and normalize a board from disk.

    Args:
        board_path: Path to board file, one row per line.

    Returns:
        Lowercased board rows with surrounding whitespace stripped.
    """
    with board_path.open(encoding="utf-8") as f:
        return [line.strip().lower().replace(" ", "") for line in f.readlines()]


def validate_board(board: list[str]) -> None:
    """Validate board shape and minimum dimensions.

    Args:
        board: Board rows to validate.

    Raises:
        ValueError: If row width or row count is below minimum constraints.
    """
    len_prev_row: int | None = None

    if len(board) < MIN_WORD_LEN:
        raise ValueError(f"Board has only {len(board)} rows, less than minimum ({MIN_WORD_LEN}).")
    if len(board) > MAX_BOARD_DIMENSION:
        raise ValueError(f"Board has {len(board)} rows, more than maximum ({MAX_BOARD_DIMENSION}).")

    for row_idx, row in enumerate(board):
        if len(row) < MIN_WORD_LEN:
            raise ValueError(
                f"Row number {row_idx + 1} has length ({len(row)}) less than minimum word length ({MIN_WORD_LEN})."
            )
        if len(row) > MAX_BOARD_DIMENSION:
            raise ValueError(
                f"Row number {row_idx + 1} has length ({len(row)}) greater than maximum ({MAX_BOARD_DIMENSION})."
            )
        invalid_chars = sorted({char for char in row if char not in VALID_BOARD_CHARS})
        if invalid_chars:
            raise ValueError(f"Row number {row_idx + 1} contains invalid characters: {''.join(invalid_chars)}.")

        if len_prev_row is None:
            len_prev_row = len(row)
            continue

        if len(row) != len_prev_row:
            raise ValueError(
                f"Row number {row_idx + 1} has length ({len(row)}) different from previous row ({len_prev_row})."
            )


def board_to_text(board: list[str], separator: str = "") -> str:
    """Serialize a board into text rows separated by newlines."""
    return "\n".join(separator.join(row) for row in board)


def print_board(board: list[str], separator: str = "") -> None:
    """Print a board to stdout."""
    print(board_to_text(board, separator=separator))


def dump_board(board: list[str], board_path: Path, separator: str = "") -> None:
    """Write a board to a file path."""
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board_path.write_text(board_to_text(board, separator=separator) + "\n", encoding="utf-8")


def load_moves(moves_path: Path) -> list[list[BoardCoord]]:
    """Load move coordinate paths from disk.

    Each non-empty line must be a Python literal representing a list/tuple of
    coordinate pairs.

    Args:
        moves_path: Path to move-path file.

    Returns:
        Parsed move paths as lists of `(row, col)` tuples.

    Raises:
        ValueError: If a line cannot be parsed as a Python literal.
        TypeError: If parsed structures do not match expected coord-pair shape.
    """
    result: list[list[BoardCoord]] = []

    with moves_path.open() as f:
        for line_num, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue

            try:
                raw_move = ast.literal_eval(row)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Line {line_num} is not valid Python syntax for a move path.") from exc

            if not isinstance(raw_move, list | tuple):
                raise TypeError(f"Line {line_num} must be a Python list/tuple of coordinate pairs.")

            move: list[BoardCoord] = []
            for coord in raw_move:
                if not isinstance(coord, list | tuple) or len(coord) != COORD_PARTS_COUNT:
                    raise TypeError(f"Line {line_num} has invalid coord {coord!r}; expected pair[int, int].")

                row_idx, col_idx = coord
                if not isinstance(row_idx, int) or not isinstance(col_idx, int):
                    raise TypeError(f"Line {line_num} has invalid coord {coord!r}; expected pair[int, int].")

                move.append((row_idx, col_idx))

            result.append(move)

    return result


def validate_move_paths(valid_moves: list[list[BoardCoord]], board: list[str]) -> None:
    """Validate move-path coordinates for uniqueness and bounds.

    Args:
        valid_moves: Candidate move paths to validate.
        board: Current board used for boundary checks.

    Raises:
        ValueError: If a path repeats coordinates or contains out-of-bounds cells.
    """
    for move_idx, move in enumerate(valid_moves):
        seen_coords: set[BoardCoord] = set()
        for coord in move:
            if coord in seen_coords:
                raise ValueError(f"Move {move_idx} contains duplicate coord {coord}.")
            if coord[0] < 0 or coord[0] >= len(board):
                raise ValueError(f"Move {move_idx} contains row {coord[0]} outside board bounds.")
            if coord[1] < 0 or coord[1] >= len(board[coord[0]]):
                raise ValueError(f"Move {move_idx} contains col {coord[1]} outside board bounds.")

            seen_coords.add(coord)


def generate_random_board(
    rows: int = 8, cols: int = 6, *, include_blocked: bool = False, rng: Random | None = None
) -> list[str]:
    """Generate a random board within the supported size limits."""
    if rows < MIN_BOARD_DIMENSION or rows > MAX_BOARD_DIMENSION:
        raise ValueError(f"rows must be between {MIN_BOARD_DIMENSION} and {MAX_BOARD_DIMENSION}")
    if cols < MIN_BOARD_DIMENSION or cols > MAX_BOARD_DIMENSION:
        raise ValueError(f"cols must be between {MIN_BOARD_DIMENSION} and {MAX_BOARD_DIMENSION}")

    generator = rng or Random()
    alphabet = ascii_lowercase + (BLOCKED_CELL if include_blocked else "")
    return ["".join(generator.choice(alphabet) for _ in range(cols)) for _ in range(rows)]


def coords_to_word(board: list[str], coords: list[BoardCoord]) -> str:
    """Convert a coordinate path into its board word."""
    return "".join(board[row_idx][col_idx] for row_idx, col_idx in coords)
