# Strands Solver

A Python tool for solving Strands word puzzle games. Given a board grid and allowed words, the solver finds valid word paths using depth-first search on a Trie data structure with adjacency constraints and island-size validation.

## Developer Setup

### Prerequisites

- **Python 3.12+** must be installed on your system
- **uv** package manager (install via https://docs.astral.sh/uv/getting-started/)

### Environment Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   cd strands_solver
   ```

2. Sync dependencies and create virtual environment:
   ```bash
   uv sync
   ```
   This command will:
   - Create a `.venv` virtual environment
   - Install all dependencies and dev tools (pytest, ruff, ty, pre-commit)
   - Install the project in editable mode

3. Verify the setup is correct:
   ```bash
   uv run python --version
   ```

### Running Tests

Run the full test suite:
```bash
uv run pytest
```

Run tests with coverage report:
```bash
uv run pytest --cov=src/strands_solver
```

Run specific test module:
```bash
uv run pytest tests/test_solver.py
uv run pytest tests/test_bot.py
uv run pytest tests/test_io_utils.py
```

Run a specific test:
```bash
uv run pytest tests/test_solver.py::test_neighbors_in_bounds
```

### Code Quality Checks

Run all quality checks:
```bash
uv run ty check          # Type checking
uv run ruff check .      # Linting
uv run ruff format --check .  # Format verification
```

Run pre-commit hooks:
```bash
uv run pre-commit run --all-files
```

### Running the CLI Application

The solver takes a board grid, a dictionary of allowed words, and a list of valid moves to verify.

**Required options:**
- `-w, --words FILE` — Path to allowed words file (one word per line)
- `-b, --board FILE` — Path to board grid file
- `-m, --moves FILE` — Path to valid moves file (Python literal format)

**Basic usage:**
```bash
uv run strands_solver -w <words_file> -b <board_file> -m <moves_file>
```

**Example with provided data:**
```bash
uv run strands_solver \
  -w data/allowed_words.txt \
  -b data/example1/board.txt \
  -m data/example1/valid_moves.txt
```

**Example output:**
```
budget: [(0, 1), (0, 2), (1, 2), (2, 2), (2, 3)]
bargain: [(1, 0), (0, 0), (1, 1), (2, 1), (2, 0), (0, 1), (1, 2)]
sale: [(2, 0), (1, 0), (0, 0), (0, 1)]
inexpensive: [(0, 1), (1, 2), (2, 2), (2, 3), (1, 3), (0, 2), (1, 1), (2, 1), (1, 0), (0, 0)]
affordable: [(0, 1), (1, 1), (1, 2), (0, 2), (2, 2), (1, 3), (0, 3), (1, 0), (0, 0)]
```

### Input File Formats

**Board file** (`board.txt`): Grid of characters, one row per line, all rows must be of same size:
```
hello
world
board
words
```

**Allowed words file** (`allowed_words.txt`): One word per line:
```
allowed
words
mold
house
hello
world
board
```

**Valid moves file** (`valid_moves.txt`): Coordinate paths as Python literals (lists or tuples of coordinate pairs):
```
[(0, 0), (0, 1), (0, 2), [0, 3], (0, 4)]
([1, 0], (1, 1), [1, 2], [1, 3], [1, 4])
```

### Development Notes

- All inputs are automatically normalized to **lowercase** on load
- Coordinates are **0-indexed** tuples: `(row, col)`
- Minimum word length is **4 characters**
- Adjacent cells include all 8 directions (horizontal, vertical, diagonal)
- The solver validates that matched paths don't leave isolated board regions smaller than 4 cells
- Type hints are enforced with `ty` (Python's strict type checker)
- Code is formatted and linted with `ruff`
