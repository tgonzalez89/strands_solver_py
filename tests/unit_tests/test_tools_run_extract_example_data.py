from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_extract_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "tools" / "run_extract_example_data.py"
    spec = importlib.util.spec_from_file_location("run_extract_example_data", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_circle_diameter_pass_and_fail() -> None:
    module = _load_extract_module()

    status_ok, message_ok = module.validate_circle_diameter(estimated=123, reference=120, tolerance_px=4)
    status_fail, message_fail = module.validate_circle_diameter(estimated=130, reference=120, tolerance_px=4)

    assert status_ok == "PASS"
    assert "tol=±4px" in message_ok
    assert status_fail == "FAIL"
    assert "diff=10px" in message_fail


def test_read_reference_circle_diameter(tmp_path: Path) -> None:
    module = _load_extract_module()

    example_dir = tmp_path / "example_synth_x"
    example_dir.mkdir()
    (example_dir / "circle_diameter.txt").write_text("118\n", encoding="utf-8")

    assert module.read_reference_circle_diameter(example_dir) == 118


def test_read_reference_circle_diameter_returns_none_when_missing(tmp_path: Path) -> None:
    module = _load_extract_module()

    example_dir = tmp_path / "example_synth_x"
    example_dir.mkdir()

    assert module.read_reference_circle_diameter(example_dir) is None
