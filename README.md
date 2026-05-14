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

### Optional Dependencies (Device Features)

The `device` extras group (`appium-python-client`, `opencv-python`, `pillow`, `tesserocr`, …) is required for the ADB and Appium CLI commands that read the board from a real device screen. Install it with:

```bash
uv sync --all-extras
```

#### tesserocr on Windows

`tesserocr` has no official Windows wheel on PyPI. Before running `uv sync --all-extras` you must provide the wheel manually:

1. **Install Tesseract-OCR** on your machine.
   Follow the instructions on the [tesserocr PyPI page](https://pypi.org/project/tesserocr/).

2. **Download the matching wheel** for your Python version and architecture from:
   [https://github.com/simonflueckiger/tesserocr-windows_build/releases](https://github.com/simonflueckiger/tesserocr-windows_build/releases)

   Pick the file that matches your setup, e.g.:
   `tesserocr-2.10.0-cp314-cp314-win_amd64.whl` for Python 3.14 on 64-bit Windows.

3. **Place the wheel** in the `wheels/` directory at the repository root:
   ```
   strands_solver/
   └── wheels/
       └── tesserocr-2.10.0-cp314-cp314-win_amd64.whl
   ```

4. **Uncomment** the [tool.uv] and [tool.uv.sources] sections in the pyproject.toml

5. **Run `uv sync --all-extras`** — `uv` will pick up the wheel automatically from that directory.

> **Note:** The `wheels/` directory is git-ignored for `.whl` files, so wheels are never committed to the repository. Each developer must add their platform's wheel locally.

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
uv run pytest tests/unit_tests/test_solver.py
uv run pytest tests/unit_tests/test_bot.py
uv run pytest tests/unit_tests/test_util_util.py
```

Run a specific test:
```bash
uv run pytest tests/unit_tests/test_solver.py::test_neighbors_in_bounds
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

The bot CLI is split into focused commands:

- **File CLI**: `strands_solver_file`
- **Fake-device CLI**: `strands_solver_fake`
- **ADB-device CLI**: `strands_solver_adb`
- **Appium-device CLI**: `strands_solver_appium`
- **Word-discovery CLI**: `strands_words`

#### File CLI (`strands_solver_file`)

Required options:
- `-w, --allowed-words FILE` - Path to allowed words file
- `-b, --board FILE` - Board file
- `-m, --valid-moves FILE` - Valid moves file

Example:
```bash
uv run strands_solver_file -w <words_file> -b <board_file> -m <moves_file>
```

#### Fake-device CLI (`strands_solver_fake`)

Required options:
- `-w, --allowed-words FILE` - Path to allowed words file
- `-b, --board FILE` - Board file
- `-m, --valid-moves FILE` - Valid moves file

Optional fake options:
- `--spangram-index N` - 0-based spangram move index (repeatable)
- `--theme {light,dark}` - Render theme for fake screenshots
- `--tessdata-dir PATH` - Tesseract tessdata directory

Example:
```bash
uv run strands_solver_fake \
   -w <words_file> \
   -b <board_file> \
   -m <moves_file> \
   --spangram-index 0 \
   --theme light
```

#### Appium-device CLI (`strands_solver_appium`)

Required options:
- `-w, --allowed-words FILE` - Path to allowed words file

Optional Appium options:
- `--appium-url URL` - Appium server URL
- `--device-name SERIAL` - ADB device serial
- `--app-package PACKAGE` - Android package name
- `--app-activity ACTIVITY` - Android activity showing the board
- `--tessdata-dir PATH` - Tesseract tessdata directory

Example:
```bash
uv run strands_solver_appium -w <words_file>
```

#### ADB-device CLI (`strands_solver_adb`)

Required options:
- `-w, --allowed-words FILE` - Path to allowed words file

Optional ADB options:
- `--adb-path PATH` - ADB executable path/command (default: `adb`)
- `--adb-host HOST` - ADB server host (passed as `adb -H`)
- `--adb-port PORT` - ADB server port (passed as `adb -P`)
- `--device-serial SERIAL` - ADB device serial
- `--tap-delay-ms N` - Delay between taps in milliseconds
- `--adb-timeout-s N` - Timeout in seconds per ADB command
- `--tessdata-dir PATH` - Tesseract tessdata directory

Example:
```bash
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
uv run strands_solver_adb \
   -w data/allowed_words.txt \
   --device-serial RF8M12345AB \
   --verbose
```

WSL example (use Windows host adb server):
```bash
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
uv run strands_solver_adb \
   -w data/allowed_words.txt \
   --adb-host host.docker.internal \
   --adb-port 5037 \
   --device-serial RF8M12345AB \
   --verbose
```

#### Word-discovery CLI (`strands_words`)

Required options:
- `-w, --allowed-words FILE` - Path to allowed words file
- `-b, --board FILE` - Board file

Example:
```bash
uv run strands_words -w data/allowed_words.txt -b data/example1/board.txt
```

---

### Running on a Real Android Device

This section covers everything needed to control a physical Android phone via USB using `strands_solver_appium`.

#### Prerequisites

- An **Android phone** with a USB cable
- **Android Debug Bridge (ADB)** installed on your PC:
  ```bash
  # Debian/Ubuntu
  sudo apt install adb
  ```
- **Node.js 18+** for the Appium server:
  ```bash
  # Debian/Ubuntu (via NodeSource)
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt install -y nodejs
  ```
- The `device` extras of this package installed:
  ```bash
  uv sync --extra device
  ```

#### Step 1 – Enable USB Debugging on the phone

1. Open **Settings → About phone** and tap **Build number** seven times to unlock Developer Options.
2. Go to **Settings → Developer options** and enable **USB debugging**.
3. Plug the phone into the PC. Accept the "Allow USB debugging?" prompt on the phone.
4. Verify the connection:
   ```bash
   adb devices
   ```
   You should see one entry like `RF8M12345AB  device`. Note the serial — you'll use it as `--device-name`.

#### Step 2 – Install and start the Appium server

```bash
npm install -g appium
appium driver install uiautomator2
appium                              # listens on http://localhost:4723 by default
```

Leave this running in a separate terminal.

#### Step 3 – Find the NYT app's package name and activity

Open the NYT Games / Strands app on the phone (navigate to the Strands board screen), then run:

```bash
adb shell dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp'
```

The output will look something like:

```
mCurrentFocus=Window{... com.nytimes.android/com.nytimes.games.strands.StrandsActivity}
```

This gives you both values:
- `--app-package com.nytimes.android`
- `--app-activity com.nytimes.games.strands.StrandsActivity`

#### Step 4 – Run the bot

Combine everything into one command. Make sure the Strands puzzle board is visible on the phone before running.

```bash
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
uv run strands_solver_appium \
  -w data/allowed_words.txt \
  --device-name RF8M12345AB \
  --app-package com.nytimes.android \
  --app-activity com.nytimes.games.strands.StrandsActivity \
  --verbose
```

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--appium-url` | `http://localhost:4723` | Appium server URL |
| `--device-name` | *(auto-detected)* | ADB device serial from `adb devices` |
| `--app-package` | `com.nytimes.android` | Android package name |
| `--app-activity` | *(required)* | Android activity showing the board |

#### Troubleshooting

- **`adb devices` shows `unauthorized`** — unlock the phone and accept the USB debugging prompt.
- **Appium session fails to start** — confirm `uiautomator2` driver is installed (`appium driver list --installed`) and the phone is detected by ADB.
- **OCR reads the wrong letters** — the board must be fully visible with no dialogs or overlays. The `--verbose` flag prints each extracted board state for inspection.
- **`TESSDATA_PREFIX` errors** — see the [Tesseract data path](#tesseract-data-path-tessdata_prefix) section above.

---

**Fake OCR-test mode:**
```bash
uv run strands_solver_fake \
   -w <words_file> \
   -b <board_file> \
   -m <moves_file> \
   --spangram-index 0 \
   --theme light
```

### Tesseract data path (TESSDATA_PREFIX)

OCR-backed flows (`strands_solver_fake`, `strands_solver_adb`, and `strands_solver_appium`) require Tesseract language data.
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
uv run strands_solver_fake \
   -w data/allowed_words.txt \
   -b data/example1/board.txt \
   -m data/example1/valid_moves.txt \
   --spangram-index 2 \
   --theme light
```

Example with verbose output:
```bash
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
uv run strands_solver_fake \
   --allowed-words data/allowed_words.txt \
   --board data/example1/board.txt \
   --valid-moves data/example1/valid_moves.txt \
   --spangram-index 2 \
   --theme light \
   --verbose
```

Note: appium mode requires Appium server running locally and a real Android device connected via USB. ADB mode requires `adb` installed and a real Android device connected via USB.

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
