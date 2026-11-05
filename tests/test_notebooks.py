"""Schema, execution-state, and discipline checks for migrated research notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = (
    "01_dataset_audit.ipynb",
    "02_split_audit.ipynb",
    "03_otsu_baseline.ipynb",
    "04_lab_kmeans.ipynb",
    "05_segmentation_comparison.ipynb",
    "05b_segmentation_case_analysis.ipynb",
)


@pytest.mark.parametrize("notebook_name", EXPECTED_NOTEBOOKS)
def test_notebook_schema_execution_and_figures(notebook_name: str) -> None:
    nb_path = NOTEBOOKS_DIR / notebook_name
    assert nb_path.is_file(), f"Missing notebook: {nb_path}"

    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)

    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert code_cells, f"{notebook_name} has no code cells"

    expected_exec = 1
    png_count = 0
    joined_source = "\n".join(cell.source for cell in code_cells)
    assert "EXPORT_FIGURES = False" in joined_source

    for cell in code_cells:
        assert cell.execution_count == expected_exec, (
            f"{notebook_name} cell {cell.get('id')} has execution_count={cell.execution_count}, "
            f"expected sequential {expected_exec}"
        )
        expected_exec += 1
        for out in cell.outputs:
            assert out.get("output_type") != "error", (
                f"{notebook_name} contains error output: {out.get('ename')}: {out.get('evalue')}"
            )
            if "image/png" in out.get("data", {}):
                png_count += 1

    assert png_count >= 1, f"{notebook_name} has no rendered PNG figures"
