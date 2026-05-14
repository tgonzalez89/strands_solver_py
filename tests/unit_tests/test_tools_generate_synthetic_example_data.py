from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_generator_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "tools" / "generate_synthetic_example_data.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_example_data", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_circle_diameter_file(tmp_path: Path) -> None:
    module = _load_generator_module()

    module.write_circle_diameter_file(tmp_path, 117)

    assert (tmp_path / "circle_diameter.txt").read_text(encoding="utf-8") == "117\n"
