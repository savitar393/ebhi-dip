"""Tests for portable source-text hashing, legacy CRLF diagnosis, and provenance guards."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from ebhi_dip.provenance import (
    PROVENANCE_MIGRATION_REL,
    diagnose_legacy_source_hash,
    sha256_bytes,
    sha256_source_text,
    verify_helper_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED_KP_SHA = "42a0a3a5c84689f7f3f221d80e27a28ee3949e4d3e3a64c1c553fc1513503299"
RECORDED_PP_SHA = "b83e549480fec3f8373eb7d95db6a8194d53795d91611797f0125fc6c673f8ca"


def test_source_text_hash_invariance_and_edit_sensitivity(tmp_path: Path) -> None:
    lf_file = tmp_path / "mod_lf.py"
    crlf_file = tmp_path / "mod_crlf.py"
    cr_file = tmp_path / "mod_cr.py"
    bom_file = tmp_path / "mod_bom.py"

    lf_bytes = b"def f(x):\n    return x + 1\n"
    crlf_bytes = b"def f(x):\r\n    return x + 1\r\n"
    cr_bytes = b"def f(x):\r    return x + 1\r"
    bom_bytes = b"\xef\xbb\xbf" + crlf_bytes

    lf_file.write_bytes(lf_bytes)
    crlf_file.write_bytes(crlf_bytes)
    cr_file.write_bytes(cr_bytes)
    bom_file.write_bytes(bom_bytes)

    assert sha256_bytes(lf_file) != sha256_bytes(crlf_file)
    base_portable = sha256_source_text(lf_file)
    assert sha256_source_text(crlf_file) == base_portable
    assert sha256_source_text(cr_file) == base_portable
    assert sha256_source_text(bom_file) == base_portable

    # Sensitive to real code edit, whitespace edit, and final newline removal.
    code_edit = tmp_path / "edit_code.py"
    code_edit.write_bytes(b"def f(x):\n    return x + 2\n")
    assert sha256_source_text(code_edit) != base_portable

    space_edit = tmp_path / "edit_space.py"
    space_edit.write_bytes(b"def f(x):\n    return x + 1 \n")
    assert sha256_source_text(space_edit) != base_portable

    no_final_nl = tmp_path / "edit_nl.py"
    no_final_nl.write_bytes(b"def f(x):\n    return x + 1")
    assert sha256_source_text(no_final_nl) != base_portable


def test_frozen_split_rejected_from_source_normalization(tmp_path: Path) -> None:
    fake_split = tmp_path / "split_manifest.csv"
    fake_split.write_bytes(b"sample_id,split\r\n")
    with pytest.raises(ValueError, match="never be applied"):
        sha256_source_text(fake_split)
    with pytest.raises(ValueError, match="never be applied"):
        diagnose_legacy_source_hash(fake_split, "0" * 64)


def test_diagnose_legacy_crlf_lf_relationship(tmp_path: Path) -> None:
    sample_lf = b"import numpy as np\n\nx = 1\n"
    sample_crlf = sample_lf.replace(b"\n", b"\r\n")
    expected_crlf_sha = __import__("hashlib").sha256(sample_crlf).hexdigest()

    lf_copy = tmp_path / "helper.py"
    lf_copy.write_bytes(sample_lf)
    diag = diagnose_legacy_source_hash(lf_copy, expected_crlf_sha)
    assert diag["exact_match"] is False
    assert diag["lf_match"] is False
    assert diag["crlf_match"] is True


def test_verify_helper_provenance_enforces_declared_and_active_hashes(tmp_path: Path) -> None:
    res_kp = verify_helper_provenance(REPO_ROOT, "kmeans_prediction", RECORDED_KP_SHA)
    res_pp = verify_helper_provenance(REPO_ROOT, "postprocessing", RECORDED_PP_SHA)
    assert res_kp["verified"] is True
    assert res_pp["verified"] is True

    with pytest.raises(ValueError, match="Undeclared helper SHA-256"):
        verify_helper_provenance(REPO_ROOT, "kmeans_prediction", "f" * 64)

    # Set up isolated workspace copy to test tamper detection on active migrated module.
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "ebhi_dip").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / PROVENANCE_MIGRATION_REL, tmp_path / PROVENANCE_MIGRATION_REL)
    seg_target = tmp_path / "src" / "ebhi_dip" / "segmentation.py"
    shutil.copy2(REPO_ROOT / "src" / "ebhi_dip" / "segmentation.py", seg_target)

    # Convert segmentation.py to CRLF in temp workspace: portable check still succeeds!
    lf_bytes = seg_target.read_bytes().replace(b"\r\n", b"\n")
    seg_target.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    assert verify_helper_provenance(tmp_path, "kmeans_prediction", RECORDED_KP_SHA)["verified"] is True

    # Genuine code modification in segmentation.py must fail immediately!
    seg_target.write_bytes(lf_bytes + b"\n# unauthorized edit\n")
    with pytest.raises(ValueError, match="Active migrated module .* source mismatch"):
        verify_helper_provenance(tmp_path, "kmeans_prediction", RECORDED_KP_SHA)
