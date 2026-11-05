# EBHI-DIP — struktur dan migrasi notebook

Checkpoint review: `8192e3e09384b05fa760104436f3a2e226380ac3`, 2026-11-05. Arsitektur telah disetujui. Migrasikan pekerjaan yang sudah tersedia beserta analisis kasus dan histogram dalam satu task; urutan di bawah adalah dependency order, bukan approval gates.

## Penilaian proyek

Repository berada pada checkpoint segmentation experiments: dataset audit, grouping, frozen split, Otsu, global Lab K-means, morphology, dan validation scores sudah tersedia. Ada 22 scripts di root; path handling dan metrics masih berulang. Raw images dan sampled-pixel cache memang Git-ignored.

Frozen split cocok dengan checksum: 1.535 train, 372 val, 319 test, dengan 353 canonical filename groups. Groups ini belum verified patient IDs. Dua validation CSVs menyediakan seluruh hasil per citra untuk perbandingan.

Raw k4/M0 adalah kandidat image-dependent terbaik: equal-category mean Dice 0.7761 dan IoU 0.6427. All-foreground control lebih tinggi, 0.7767/0.6477. Temuan ini harus terlihat dalam narasi; manfaat morphological descriptors bagi classification belum diuji.

Pada checkpoint belum ada committed notebook, README, dependency specification, feature extraction, classifier, C++ app, atau `segmentation_selection.json`. Periksa actual checkout sebelum membuat file; jangan menimpa pekerjaan lokal yang lebih baru.

## Struktur hybrid

Notebooks menyajikan pertanyaan, analisis, tabel, figures, dan interpretasi. Python package menyimpan perhitungan reusable. Thin CLI runners mendukung batch work. Aplikasi C++ nantinya memakai exported artifacts dan parity contract yang eksplisit.

Jangan membuat satu notebook per script atau satu notebook besar berisi semua tahap. Kelompokkan berdasarkan tujuan penelitian berikut.

| Path | Fungsi |
|---|---|
| `README.md` | Tujuan, setup, lokasi data, urutan notebook, status dan keterbatasan. |
| `pyproject.toml` | Editable package dan direct dependencies; notebook/dev extras. |
| `requirements-lock.txt` | Resolved environment beserta Python/platform context; pisahkan baseline historis dari environment migrasi. |
| `notebooks/` | Numbered research notebooks dengan concise Markdown dan useful saved outputs. |
| `src/ebhi_dip/` | Numerical core, data/path validation, artifact loading, metrics, plotting. |
| `scripts/` | Thin guarded CLI entry points; import tidak menjalankan batch jobs. |
| `configs/` | Parameter dan recorded decisions dengan provenance. |
| `data/raw/` | Original dataset/archive, lokal dan Git-ignored. |
| `data/metadata/` | Manifests, audit records, grouping, frozen split dan checksum; paths tetap. |
| `data/processed/` | Regenerable caches, Git-ignored. |
| `models/` | Learned JSON artifacts sekarang; classifier/scaler exports kelak. |
| `results/` | Historical score CSVs dan run metadata; paths/bytes dipertahankan. |
| `reports/figures/` | Figure exports dari plotting code. |
| `docs/` | Methodology, provenance migration, research decisions. |
| `tests/` | Numerical contracts, cohort integrity, paired ranking, provenance, dan parity. |
| `cpp/` | Future desktop application; tidak diimplementasikan dalam task ini. |

Package cukup memiliki modul yang benar-benar diperlukan: `paths.py`, `dataset.py`, `metrics.py`, `provenance.py`, `segmentation.py`, `analysis.py`, dan `plotting.py`. Pecah modul bila kompleksitas nyata membutuhkannya. Jangan membuat feature/classification framework kosong untuk pekerjaan masa depan.

Gunakan editable install melalui `.venv`. Hindari repeated `sys.path` hacks atau `%run` complete legacy scripts sebagai notebook architecture.

## Peta notebook

| Notebook | Source material | Isi |
|---|---|---|
| `01_dataset_audit.ipynb` | `inspect_dataset`, `build_manifest`, `audit_files`, `audit_masks`, `audit_duplicates`, `inspect_pair` | Pairing, decoding, mask encoding, missing masks, duplicates, dan training-image overlays. |
| `02_split_audit.ipynb` | `inspect_groups`, `create_split`, `freeze_split`, frozen metadata | Grouping, overlap checks, distribusi category/image/group; default memverifikasi split yang sudah ada. |
| `03_otsu_baseline.ipynb` | `inspect_otsu`, `evaluate_otsu_train`, `evaluate_otsu_val` | Per-image threshold, fixed train-selected polarity, validation baseline, grayscale histogram. |
| `04_lab_kmeans.ipynb` | `inspect_lab`, sampling/verification scripts, `fit_lab_kmeans`, `map_kmeans_foreground`, `inspect_kmeans` | Physical Lab, sampling provenance, saved centers/mapping, contoh inference, histogram Lab. |
| `05_segmentation_comparison.ipynb` | Morphology evaluation scripts, saved CSV/JSON, supplied seed | 13 validation methods termasuk control; category means, morphology, coverage, provenance. |
| `05b_segmentation_case_analysis.ipynb` | Paired saved scores dan original validation images | Worst/best/advantage cases, both-poor cases, gallery, error overlays, paired scatter dan histogram. |
| `06_feature_extraction.ipynb` | Future research | Lab statistics, GLCM, predicted-region descriptors. Belum diimplementasikan. |
| `07_classification.ipynb` | Future research | A/B/C features, train-fitted scaling, validation-selected SVM. Belum diimplementasikan. |
| `08_final_evaluation.ipynb` | Future research | Held-out evaluation setelah keputusan final. Belum diimplementasikan. |

Scope sekarang adalah 01–05 dan 05b. Jangan membuat notebook 06–08 seolah-olah eksperimennya sudah selesai.

## Konvensi notebook dan bahasa

- Mulai dengan pertanyaan, input artifacts, dan batas analisis.
- Tempatkan imports, parameters, environment info, dan root discovery di awal.
- Validate inputs sebelum menghitung. Setiap notebook dapat dijalankan dari fresh kernel tanpa state notebook lain.
- Gunakan bahasa Indonesia yang ringkas; pertahankan English technical terms yang lazim serta original IDs/schema.
- Jelaskan tujuan sebelum operasi penting, lalu satu interpretasi singkat dekat tabel/figure terkait. Jangan menarasikan setiap baris kode.
- Pisahkan observasi dari dugaan penyebab dan keterbatasan. Jangan memaksakan kesimpulan sesuai harapan.
- Default membaca existing artifacts. Fitting, regenerate split, dan heavy writes opt-in; tidak dieksekusi dalam migrasi.
- Simpan useful tables/plots, bukan full pixel arrays atau log panjang. Figure export berasal dari plotting code.
- Jika inputs opsional tidak tersedia, bagian lain tetap berguna dan missing checks dinyatakan eksplisit.
- Gunakan `.venv` kernel di VS Code. Notebook format sendiri tidak mengharuskan penggantian scientific environment.

## Shared numerical core

Ekstrak image loading, RGB→Lab, reference thresholding, fixed Otsu/K-means inference, morphology, Dice/IoU, dan equal-category aggregation. Pindahkan callers sesudah parity terbukti. Runners harus memiliki `if __name__ == '__main__':` dan memanggil package, bukan menduplikasi scientific implementation.

Reference tidak boleh menjadi input untuk memilih prediksi per citra. Pertahankan seluruh kontrak numerical dari `CODEX_HANDOFF.md`, termasuk cluster mapping, float64 assignment, connectivity, morphology order/kernel/border, dan empty-mask convention.

## Reproducibility yang harus diselesaikan

**Paths.** Dua puluh scripts memakai `Path(__file__).resolve().parent`. Setelah pindah ke `scripts/`, asumsi ini salah. Gunakan satu root discovery dan explicit data-root override untuk ignored inputs di luar worktree. Manifest paths tetap relative terhadap project root.

**Source hashes.** Committed LF helpers tidak cocok dengan recorded raw-byte hashes, tetapi CRLF conversion cocok tepat. Pertahankan fingerprints lama. Buat versioned portable source convention dan explicit migration record, disertai numerical parity; jangan disable guards atau silently replace hashes.

**Frozen split.** Hash raw bytes tetap authoritative. Source-text normalization tidak boleh diterapkan ke frozen CSV; jangan regenerate, resave, atau renormalize file ini.

**Git attributes.** Pertahankan rule `data/metadata/split_manifest.csv -text`. Beberapa rule lain menunjuk path yang tidak ada; perbaiki secara sengaja, tanpa mass renormalization historical artifacts.

**Environment.** Catat runtime aktual dan limitations. Jangan mengubah run records lama agar menyatakan environment baru. Jika package/API berbeda menyebabkan mismatch, diagnosis dahulu; jangan mengubah scores.

## Urutan implementasi

1. Baseline hashes, package/environment, paths, metrics, artifact loaders.
2. Portable provenance dan explicit adapter untuk historical CRLF guards.
3. Shared segmentation functions dan parity; lalu thin runners.
4. Notebook 01–05, termasuk localization seed dengan gaya Indonesia + technical English yang alami.
5. Notebook 05b sesuai `VISUAL_ANALYSIS_REQUIREMENTS.md`: paired ranking, gallery, overlays, image/score histograms, optional widgets.
6. Fresh-kernel execution, meaningful tests, visual QA, historical-byte verification, README, dan laporan akhir.

Lanjutkan seluruh urutan tanpa intermediate approval gates. Feature extraction, SVM, final test, dan C++ adalah roadmap berikutnya. Detail tasks ada pada plan full migration yang direferensikan brief.
