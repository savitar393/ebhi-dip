"""Tests verifying that thin CLI runners in scripts/ are import-safe and overwrite-guarded."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from ebhi_dip.provenance import sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def test_all_scripts_import_cleanly_without_side_effects() -> None:
    split_sha_before = sha256_bytes(REPO_ROOT / "data/metadata/split_manifest.csv")
    script_paths = sorted(SCRIPTS_DIR.glob("*.py"))
    assert len(script_paths) == 17

    for script_path in script_paths:
        spec = importlib.util.spec_from_file_location(f"_cli_{script_path.stem}", script_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        assert hasattr(mod, "build_parser")

    assert sha256_bytes(REPO_ROOT / "data/metadata/split_manifest.csv") == split_sha_before


def test_scripts_refuse_overwrite_by_default_and_verify_split_from_outside_cwd(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("_cli_freeze", SCRIPTS_DIR / "freeze_split.py")
    assert spec is not None and spec.loader is not None
    freeze_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze_mod)

    with pytest.raises(FileExistsError, match="immutable"):
        freeze_mod.main(["--project-root", str(REPO_ROOT)])

    out = subprocess.check_output(
        [sys.executable, str(SCRIPTS_DIR / "freeze_split.py"), "--verify-only"],
        cwd=tmp_path,
        text=True,
    )
    assert "2226 samples" in out
    assert "353 groups" in out


def test_stage06_and_stage07_2x3_scripts_refuse_overwrite_by_default() -> None:
    for script_name in ("extract_features.py", "evaluate_svm_val.py"):
        spec = importlib.util.spec_from_file_location(f"_cli_{script_name[:-3]}", SCRIPTS_DIR / script_name)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            mod.main(["--project-root", str(REPO_ROOT)])
