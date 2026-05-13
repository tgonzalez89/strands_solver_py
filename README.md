# Strands Solver

A Python tool for solving Strands word puzzle games. Given a board grid and allowed words, the solver finds valid word paths using depth-first search on a Trie data structure with adjacency constraints and island-size validation.

## Developer Setup

### Prerequisites

- **Python 3.14+** must be installed on your system
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

Run the full test suite without coverage gate:
```bash
uv run pytest --no-cov
```

Run tests with coverage report and fail-under enforcement:
```bash
uv run pytest --cov=src/strands_solver
```

Run specific test module:
```bash
uv run pytest tests/test_solver.py
uv run pytest tests/test_bot.py
uv run pytest tests/test_util_util.py
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

### Tools Scripts

Repository helper scripts now live in [tools/](tools/).

- [tools/generate_synthetic_example_data.py](tools/generate_synthetic_example_data.py):
   Regenerates synthetic OCR/cell-state datasets under [data/](data/) (`example_synth_*`).
- [tools/run_extract_example_data.py](tools/run_extract_example_data.py):
   Runs board-row OCR and cell-state extraction for all `data/example*` directories and writes logs to [.debug/](.debug/).

Run them from the repository root:

```bash
uv run python tools/generate_synthetic_example_data.py
uv run python tools/run_extract_example_data.py
```

The extraction tool writes OCR/cell-state logs and per-image debug boards under [.debug/](.debug/).

### Running the CLI Application

The CLI now selects mode with `--driver`:

- **File mode** (`--driver file`): reads board/moves from files
- **Appium mode** (`--driver appium`): runs real device flow
- **Fake mode** (`--driver fake`): runs OCR against generated screenshots

For file mode, there are two sub-modes:

- **Discovery mode** (no moves file): prints all possible words found on the board
- **Verification mode** (with moves file): applies expected moves and prints `matched=X/Y`

**Options:**
- `-w, --allowed-words FILE` - Path to allowed words file (one word per line)
- `--driver {file,appium,fake}` - Backend mode
- `-b, --board FILE` - Board file (required for `file` and `fake` driver)
- `-m, --valid-moves FILE` - Optional valid moves file (file mode only)
- `--spangram-index N` - 0-based spangram move index (repeatable, fake mode)
- `--fake-mode {light,dark}` - Render mode for fake screenshots (fake mode)

**File mode discovery:**
```bash
uv run strands_solver --driver file -w <words_file> -b <board_file>
```

**File mode verification:**
```bash
uv run strands_solver --driver file -w <words_file> -b <board_file> -m <moves_file>
```

**Appium mode:**
```bash
uv run strands_solver --driver appium -w <words_file>
```

**Fake OCR-test mode:**
```bash
uv run strands_solver \
   --driver fake \
   -w <words_file> \
   -b <board_file> \
   -m <moves_file> \
   --spangram-index 0 \
   --fake-mode light
```

### Tesseract data path (TESSDATA_PREFIX)

OCR-backed flows (`--driver fake` and `--driver appium`) require Tesseract language data.
If you see errors like `Failed to init API` or invalid tessdata path, set `TESSDATA_PREFIX` to the folder that contains `eng.traineddata`.

Common Linux paths:
- `/usr/share/tesseract-ocr/5/tessdata`
- `/usr/share/tessdata`
- `/usr/local/share/tessdata`

Check quickly:
```bash
ls "$TESSDATA_PREFIX"/eng.traineddata
```

Run fake mode with explicit tessdata path:
```bash
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
uv run strands_solver \
   --driver fake \
   -w data/allowed_words.txt \
   -b data/example1/board.txt \
   -m data/example1/valid_moves.txt \
   --spangram-index 2 \
   --fake-mode light
```

Example with verbose output:
```bash
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
uv run strands_solver \
   --driver fake \
   --allowed-words data/allowed_words.txt \
   --board data/example1/board.txt \
   --valid-moves data/example1/valid_moves.txt \
   --spangram-index 2 \
   --fake-mode light \
   --verbose
```

Note: appium mode requires a configured Appium session; otherwise the CLI reports `device_mode_not_ready`.

### Generating Board State Images

Use `strands_board_image` to render a Strands-like board image from a board file and optional moves file.

**Required arguments:**
- `--board`: board file path (for example `data/example1/board.txt`)
- `--output` / `-o`: output image path

**Optional arguments:**
- `--valid-moves`: moves file path (for example `data/example1/valid_moves.txt`)
- `--mode`: `dark` (default) or `light`
- `--spangram-index`: 0-based index of a move to draw as spangram (repeatable)

If a coordinate is not covered by `--valid-moves`, it is rendered with board background color.
In dark mode, unselected letters are rendered in white.

Install image dependencies if needed:
```bash
uv sync --extra device
```

Generate a dark-mode image:
```bash
uv run strands_board_image \
   --board data/example1/board.txt \
   --valid-moves data/example1/valid_moves.txt \
   --spangram-index 0 \
   --mode dark \
   --output data/example1_board_dark.png
```

Generate a light-mode image:
```bash
uv run strands_board_image \
   --board data/example1/board.txt \
   --valid-moves data/example1/valid_moves.txt \
   --spangram-index 0 \
   --mode light \
   --output data/example1_board_light.png
```

Useful render tuning options include `--width`, `--height`, `--board-width-ratio`, `--board-height-ratio`, `--board-center-y-ratio`, `--cell-radius-ratio`, and `--font-size-ratio`.

**Verification mode example with provided data:**
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

### Docstring Style (Google)

Use Google-style docstrings for all public modules, classes, and functions.

Template:

```python
def example(param: str) -> int:
   """Short imperative summary.

   Args:
      param: Description of the parameter.

   Returns:
      Description of the returned value.

   Raises:
      ValueError: When input is invalid.
   """
```

Guidelines used in this repository:
- Start with a one-line summary ending with a period.
- Add sections only when needed (`Args`, `Returns`, `Raises`).
- Keep argument names exactly matching the function signature.
- Prefer concise behavioral descriptions over implementation details.
