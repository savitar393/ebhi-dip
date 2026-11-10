from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ebhi_dip.classification import load_svm_artifact, predict_from_svm_artifact
from ebhi_dip.features import FEATURE_NAMES
from ebhi_dip.paths import RAW_DATASET_REL, find_project_root, has_raw_dataset


def _find_cli_executable(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "cpp" / "build" / "ebhi_cli.exe",
        repo_root / "cpp" / "build" / "ebhi_cli",
        repo_root / "cpp" / "build" / "Release" / "ebhi_cli.exe",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _cpp_env() -> dict[str, str]:
    env = dict(os.environ)
    ucrt_bin = r"C:\msys64\ucrt64\bin"
    if Path(ucrt_bin).exists():
        env["PATH"] = ucrt_bin + os.pathsep + env.get("PATH", "")
    return env


def test_cpp_sources_exist() -> None:
    root = find_project_root()
    cpp_root = root / "cpp"
    assert (cpp_root / "CMakeLists.txt").exists()
    assert (cpp_root / "include" / "ebhi_pipeline.hpp").exists()
    assert (cpp_root / "src" / "ebhi_pipeline.cpp").exists()
    assert (cpp_root / "src" / "main_cli.cpp").exists()
    assert (cpp_root / "src" / "main_gui.cpp").exists()


def test_cpp_cli_numerical_parity_against_python() -> None:
    root = find_project_root()
    cli_exe = _find_cli_executable(root)
    if cli_exe is None or not has_raw_dataset(root):
        pytest.skip("C++ CLI executable or raw dataset not available")

    raw_data_root = root / RAW_DATASET_REL
    for seg_method, feat_csv, svm_json in [
        ("k4_M0", "features_k4_M0.csv", "svm_k4_M0_selected.json"),
        ("otsu_M2", "features_otsu_M2.csv", "svm_otsu_M2_C.json"),
    ]:
        feat_df = pd.read_csv(root / "results" / "features" / feat_csv)
        test_rows = feat_df[feat_df["split"] == "test"].groupby("category", sort=True).head(1)
        assert len(test_rows) == 6

        svm_path = root / "models" / svm_json
        svm_art = load_svm_artifact(svm_path)

        for _, row in test_rows.iterrows():
            sample_id = str(row["sample_id"])
            category = str(row["category"])
            stem = sample_id.split("/")[-1]
            img_path = raw_data_root / category / "image" / f"{stem}.png"
            mask_path = raw_data_root / category / "label" / f"{stem}.png"

            proc = subprocess.run(
                [
                    str(cli_exe),
                    "--image",
                    str(img_path),
                    "--mask",
                    str(mask_path),
                    "--seg",
                    seg_method,
                    "--svm",
                    str(svm_path),
                    "--repo-root",
                    str(root),
                ],
                capture_output=True,
                text=True,
                env=_cpp_env(),
                check=True,
            )
            payload = json.loads(proc.stdout)

            # 1. Verify all 29 features match Python's canonical feature CSV
            cpp_feats = payload["features"]
            raw_vec = np.asarray([[float(row[col]) for col in FEATURE_NAMES]], dtype=np.float64)
            for col in FEATURE_NAMES:
                py_val = float(row[col])
                cpp_val = float(cpp_feats[col])
                assert np.isclose(cpp_val, py_val, rtol=1e-5, atol=1e-6), (
                    f"[{seg_method}] Feature mismatch on {sample_id} [{col}]: C++={cpp_val} vs Python={py_val}"
                )

            # 2. Verify SVM OVO decision values, votes, and predicted category match Python
            py_pred, py_dec, py_votes = predict_from_svm_artifact(raw_vec, svm_art)
            assert payload["predicted_category"] == str(py_pred[0])
            assert payload["ovo_votes"] == [int(v) for v in py_votes[0]]
            np.testing.assert_allclose(
                np.asarray(payload["ovo_decisions"], dtype=np.float64),
                py_dec[0],
                rtol=1e-5,
                atol=1e-6,
            )


def test_cpp_gui_offscreen_smoke(tmp_path: Path) -> None:
    root = find_project_root()
    gui_exe = root / "cpp" / "build" / "ebhi_gui.exe"
    if not gui_exe.exists() or not has_raw_dataset(root):
        pytest.skip("C++ GUI executable or raw dataset not available")

    sample_img = root / RAW_DATASET_REL / "Adenocarcinoma" / "image" / "GT2012165-1-400-001.png"
    screenshot_out = tmp_path / "gui_smoke.png"
    env = _cpp_env()
    env["QT_QPA_PLATFORM"] = "offscreen"

    subprocess.run(
        [
            str(gui_exe),
            "--repo-root",
            str(root),
            "--image",
            str(sample_img),
            "--seg",
            "k4_M0",
            "--screenshot",
            str(screenshot_out),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert screenshot_out.exists()
    assert screenshot_out.stat().st_size > 10_000

