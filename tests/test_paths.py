"""Tests for root discovery, frozen split raw-byte verification, and data-root overrides."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from ebhi_dip.paths import (
    FROZEN_SPLIT_REL,
    FROZEN_SPLIT_SHA_REL,
    find_project_root,
    resolve_data_root,
    resolve_manifest_path,
    verify_frozen_split,
)

EXPECTED_REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_SHA = "ceeba8e282f77cdb78a9675618b3dcb6729fbc2339e2d112085a11720fc9108d"


def _make_fake_project(root: Path, csv_bytes: bytes) -> tuple[Path, Path]:
    split_path = root / FROZEN_SPLIT_REL
    sha_path = root / FROZEN_SPLIT_SHA_REL
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_bytes(csv_bytes)
    digest = hashlib.sha256(csv_bytes).hexdigest()
    sha_path.write_text(f"{digest}  {split_path.name}\n", encoding="utf-8")
    return split_path, sha_path


def test_find_project_root_from_repo_and_nested_paths(tmp_path: Path) -> None:
    assert find_project_root(start=EXPECTED_REPO_ROOT) == EXPECTED_REPO_ROOT
    assert find_project_root(start=EXPECTED_REPO_ROOT / "notebooks") == EXPECTED_REPO_ROOT
    assert find_project_root(start=Path(__file__)) == EXPECTED_REPO_ROOT

    spaced = tmp_path / "folder with spaces" / "ebhi project"
    _make_fake_project(spaced, b"sample_id,split\r\ns1,train\r\n")
    nested = spaced / "notebooks" / "deep folder"
    nested.mkdir(parents=True)
    assert find_project_root(start=nested) == spaced.resolve()
    assert find_project_root(start=tmp_path, override=spaced) == spaced.resolve()


def test_find_project_root_errors_on_missing_or_invalid_override(tmp_path: Path) -> None:
    empty_dir = tmp_path / "no_markers_here"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Could not discover EBHI-DIP project root"):
        find_project_root(start=empty_dir)

    with pytest.raises(FileNotFoundError, match="Invalid project root override"):
        find_project_root(override=empty_dir)


def test_verify_frozen_split_real_and_newline_sensitivity(tmp_path: Path) -> None:
    assert verify_frozen_split(EXPECTED_REPO_ROOT) == EXPECTED_SPLIT_SHA

    crlf_bytes = b"sample_id,category,split\r\nNormal/A,Normal,train\r\n"
    split_path, _ = _make_fake_project(tmp_path, crlf_bytes)
    before_bytes = split_path.read_bytes()

    digest = verify_frozen_split(tmp_path)
    assert digest == hashlib.sha256(crlf_bytes).hexdigest()
    assert split_path.read_bytes() == before_bytes

    # Changing only CRLF -> LF must fail raw-byte verification without altering disk bytes.
    lf_bytes = crlf_bytes.replace(b"\r\n", b"\n")
    split_path.write_bytes(lf_bytes)
    with pytest.raises(ValueError, match="raw-byte checksum mismatch"):
        verify_frozen_split(tmp_path)
    assert split_path.read_bytes() == lf_bytes


def test_data_root_override_preserves_manifest_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    _make_fake_project(worktree, b"sample_id\n")

    external_checkout = tmp_path / "main checkout"
    img_rel = "data/raw/EBHI-SEG/Normal/image/GT1-1-400-001.png"
    target_img = external_checkout / img_rel
    target_img.parent.mkdir(parents=True)
    target_img.write_bytes(b"PNGDATA")

    assert resolve_data_root(worktree, data_root_override=external_checkout) == (
        external_checkout / "data/raw/EBHI-SEG"
    ).resolve()
    assert resolve_manifest_path(worktree, img_rel, data_root_override=external_checkout) == target_img.resolve()
    assert resolve_manifest_path(
        worktree,
        img_rel,
        data_root_override=external_checkout / "data/raw/EBHI-SEG",
    ) == target_img.resolve()

    monkeypatch.setenv("EBHI_DIP_DATA_ROOT", str(external_checkout))
    assert resolve_manifest_path(worktree, img_rel) == target_img.resolve()


def test_editable_import_outside_repository_root(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-c",
        f"import ebhi_dip; root = ebhi_dip.find_project_root(override={str(EXPECTED_REPO_ROOT)!r}); print(ebhi_dip.verify_frozen_split(root))",
    ]
    out = subprocess.check_output(cmd, cwd=tmp_path, text=True).strip()
    assert out == EXPECTED_SPLIT_SHA
