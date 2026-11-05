# Implementation plan — notebook migration dan visual case analysis

Arsitektur dan scope telah disetujui. Selesaikan seluruh tasks berikut dalam satu task Codex, tanpa meminta review setelah setiap tahap. Urutan mengikuti dependencies; ini menggantikan plan foundation lama.

**Tujuan:** notebook 01–05 dan 05b yang reproducible, shared Python core, paired image comparisons, histogram, serta dokumentasi ringkas. Gunakan bahasa Indonesia dengan English technical terms yang alami; jangan memaksakan terjemahan atau menulis paragraf panjang.

**Referensi:** `codex_handoff/CODEX_HANDOFF.md`, `NOTEBOOK_MIGRATION.md`, dan `VISUAL_ANALYSIS_REQUIREMENTS.md`. Preserve scientific checkpoint, historical bytes, actual local changes, serta train/val/test boundaries.

## Task 1 — baseline, package, paths, environment

**Files:** `pyproject.toml`, `src/ebhi_dip/__init__.py`, `paths.py`, `tests/test_paths.py`, README/setup, environment record.

**Interfaces:**

- `find_project_root(start=None, override=None) -> Path`: valid explicit override atau ancestor discovery dari start/current directory, memakai frozen metadata/checksum markers. Missing/invalid root menghasilkan useful error.
- `verify_frozen_split(project_root) -> str`: verifikasi exact CSV bytes terhadap sidecar, kembalikan digest atau raise. Tidak menulis inputs.
- Explicit raw-data-root override bila dataset berada di checkout utama tetapi kode di worktree; manifest strings tidak diubah. Reads boleh menuju dataset eksternal, writes ke working checkout.

- [ ] Periksa git status, HEAD, environment, local files, dan seed; simpan baseline byte hashes dari historical artifacts sebelum mengedit.
- [ ] Buat editable package dengan notebook/dev extras. Dokumentasikan real command `python -m pip install -e ".[notebooks,dev]"` dan `.venv` kernel.
- [ ] Catat resolved environment dan Python/platform. Jangan mengganti historical run metadata atau mencantumkan unverified versions sebagai tested.
- [ ] Uji root discovery dari root/nested notebooks, path dengan spaces, valid override, dan missing-root failure.
- [ ] Uji temporary frozen CSV fixture: matching bytes lolos; perubahan newline saja gagal. Pastikan verifier tidak memodifikasi input.
- [ ] Pastikan editable import bekerja dari luar repository root.

## Task 2 — metrics dan harmonized validation results

**Files:** `metrics.py`, `analysis.py` atau artifact loader kecil, relevant tests.

**Interfaces:**

- `overlap_scores(predicted, reference) -> (dice, iou)`: equal-shape Boolean masks, shape mismatch error; both empty `(1,1)`.
- `category_macro_scores(results, expected_categories=None) -> DataFrame`: columns `method/category/dice/iou`; output indexed by method, `mean_category_dice` dan `mean_category_iou`. Hitung mean per kategori dahulu.
- `load_validation_results(project_root) -> DataFrame`: harmonize wide Otsu dan long K-means schemas; validate sample IDs/categories/groups/splits dan reconcile shared control. Output 13 methods × 372 images = 4.836 rows pada checkpoint.

- [ ] Uji identical/disjoint masks, both empty, one empty, shape mismatch; fixture `P=[[1,1],[0,0]], R=[[1,0],[1,0]]` → Dice .5, IoU 1/3.
- [ ] Uji unequal category sizes: A dua scores 1, B satu score 0 → primary mean .5, bukan 2/3.
- [ ] Reject nonfinite/out-of-range metrics, missing expected categories, duplicate/missing cohort IDs, dan category/group mismatch.
- [ ] Pertahankan full precision; rounding hanya untuk display/established selection rule.
- [ ] Reconcile all-foreground per-image scores dan analytic relation `Dice=2r/(1+r)`, `IoU=r` dari recorded reference fraction.

## Task 3 — portable provenance dan explicit legacy adapter

**Files:** `provenance.py`, tests, concise methodology/migration record; review `.gitattributes` secara sengaja.

**Interfaces:**

- `sha256_bytes(path)`: exact bytes.
- `sha256_source_text(path)`: scheme `utf8-lf-v1`; UTF-8 dengan optional BOM → normalize CRLF/CR ke LF → UTF-8 tanpa BOM. Pertahankan whitespace lain dan ada/tidaknya final newline.
- `diagnose_legacy_source_hash(path, expected_sha)`: Boolean `exact_match/lf_match/crlf_match`; diagnosis bukan generic integrity bypass.

- [ ] Uji raw-byte newline sensitivity, canonical LF/CRLF/BOM equivalence, serta sensitivity terhadap real code edit, whitespace edit, dan final-newline edit.
- [ ] Verifikasi historical helper hashes di brief dan exact CRLF relationship terhadap snapshot/current legacy source.
- [ ] Buat explicit versioned migration adapter/record yang menghubungkan verified legacy implementation, refactored implementation, artifacts, dan parity evidence.
- [ ] Pertahankan historical expected hashes/run records. Penerimaan newline equivalence harus declared, restricted pada verified legacy sources, dan reject genuine source mismatch.
- [ ] Migrasikan callers yang memakai helper-byte guards; jangan membiarkan portable hash hanya sebagai diagnostic sementara image inference tetap broken.
- [ ] Jangan retrain untuk mengatasi pemindahan file. Jangan menonaktifkan integrity checks atau menerapkan source normalization ke frozen CSV.
- [ ] Pertahankan `data/metadata/split_manifest.csv -text`; tangani invalid attribute paths tanpa broad renormalization historical artifacts.

## Task 4 — shared segmentation dan thin runners

**Files:** `dataset.py`, `segmentation.py`, small helpers bila perlu, migrated `scripts/`, relevant parity tests.

**Interfaces:**

- `load_rgb(path) -> uint8 RGB array`.
- `load_reference_mask(path, threshold=128) -> bool array`; sediakan akses raw L array terpisah untuk encoding histogram.
- `predict_mask(image_rgb, method_id, project_root) -> bool H×W`: fixed Otsu/M0–M2, saved k2/k3/k4 M0–M2, dan all-foreground control. Tidak menerima reference, tidak fitting.
- RGB→Lab helper mengikuti float conversion di brief. Preserve exact numerical behavior sebelum merapikan kode.

- [ ] Ekstrak pure functions tanpa meng-import top-level batch runners yang menjalankan pekerjaan saat import.
- [ ] Preserve polarity, nearest-center float64/tie rule, cluster mapping, area/connectivity, kernel/order/iterations/border.
- [ ] Uji controlled fixtures untuk nearest-center ties, mask threshold 127/128, connectivity, area=15/16, dan morphology border behavior.
- [ ] Bandingkan old/new masks pada representative train images; selected validation cases kemudian harus mereproduksi saved scores.
- [ ] Migrasikan paths/callers dan guarded CLI entry points. Jangan memindahkan scripts dahulu lalu membiarkan root assumptions salah.
- [ ] Retire/move legacy runners hanya setelah equivalence terbukti; hindari scientific implementations yang diduplikasi dan tetap aktif.
- [ ] Jangan menjalankan ulang training, sampling besar, split creation, atau historical CSV overwrite dalam proses migrasi.

## Task 5 — notebook 01–05

**Files:** notebook map pada `NOTEBOOK_MIGRATION.md`; plotting helpers secukupnya; README notebook index.

- [ ] Konsolidasikan existing audit/split/Otsu/Lab/K-means/morphology work menurut map. Default membaca existing artifacts; optional raw-image checks diberi guard yang jelas.
- [ ] 01: pairing/missing masks, decoding, duplicate evidence, grayscale mask audit, dan representative training overlays. Jangan menyamakan semua intermediate gray pixels dengan corruption.
- [ ] 02: frozen checksum, grouping/capitalization rationale, distributions, no-group/no-recorded-fingerprint overlap. Jangan menyebut groups sebagai verified patients.
- [ ] 03: fixed train-selected polarity, per-image threshold, Otsu visual example dan grayscale histogram, saved train/val scores.
- [ ] 04: physical Lab channels/histograms, sampled-pixel provenance, saved centers dan annotation-based mapping, example fixed inference. Jangan refit by default.
- [ ] 05: gunakan seed tanpa menimpa local edits. Ganti duplicated utilities dengan package imports; pertahankan reconciliation, plots, provenance, negative control finding.
- [ ] Localize seed sesuai gaya bahasa terbaru: bahasa Indonesia ringkas dengan natural English terms. IDs/schema tetap; tidak perlu menerjemahkan semua kata.
- [ ] Setiap notebook punya parameters, input validation, useful tables/figures dan concise interpretation; fresh-kernel independent, tanpa hidden cross-notebook state.
- [ ] Jangan membuat future notebooks 06–08 seolah eksperimennya sudah selesai. Jangan menyimpan fabricated outputs saat inputs tidak ada.

## Task 6 — paired ranking dan case builder

**Files:** `analysis.py` atau `cases.py`, ranking/error-metric tests.

**Interfaces:**

- `pair_scores(results, method_a, method_b, category=None) -> DataFrame`: one-to-one validation join, paired metrics dan signed deltas.
- `rank_cases(paired, mode, n=3) -> DataFrame`: semua modes/ordering/tolerance dari visual requirements, stable sample-ID tie-break.
- `SegmentationCase`: sample/category/group, selection reasons, RGB, raw L reference, binary reference, prediction A/B, method IDs, recomputed metrics, computed Otsu threshold.
- `build_case(project_root, row, method_a, method_b, selection_reasons, ...)`: resolve actual raw data, fixed full-image inference, recompute metrics, compare saved Dice/IoU `atol=1e-10`, rtol=0. Explicit data-root override didukung bila diperlukan.

- [ ] Uji ranking fixture, missing/duplicate pair failures, group/category consistency, signed advantage filtering, tie tolerance, dan deduplication dengan multiple selection reasons.
- [ ] Uji TP/FP/FN/TN definitions serta precision/recall denominator nol; jangan mengubah historical overlap-score convention.
- [ ] Recompute validation counts: raw pair 96/276/0; processed pair 122/250/0. Jika newer legitimate artifacts berbeda, diagnosis dan laporkan, bukan hardcode output.
- [ ] Pastikan reference hanya dipakai setelah prediksi. Tidak ada per-image ground-truth optimization.
- [ ] Recomputed selected-case metrics harus cocok CSV; mismatch menghentikan klaim parity dan memicu diagnosis, bukan overwriting scores.

## Task 7 — gallery, histogram, dan notebook 05b

**Files:** `plotting.py`, `notebooks/05b_segmentation_case_analysis.ipynb`, meaningful histogram checks.

**Interfaces:**

- `plot_segmentation_case(case) -> Figure`: six-panel original/reference/predictions/error overlays plus readable metadata/legend.
- `plot_image_histograms(case, bins=64, lab_edges=None, density=False) -> Figure`: RGB, grayscale+Otsu threshold, raw reference+128, L*, a*, b*. Integer channels memakai complete 0–255 bins; physical Lab shared finite ranges.
- `plot_score_histograms(paired, bins=20) -> Figure`: paired Dice distribution dan signed delta distribution; bins konsisten.

- [ ] Implementasi detail mengikuti `VISUAL_ANALYSIS_REQUIREMENTS.md`, termasuk static defaults maksimum empat distinct cases, paired scatter, counts per kategori, dan metrics tables.
- [ ] Uji histogram counts mencakup seluruh piksel; signed Lab, constant arrays, shared bins, dan empty regions bila region histograms disediakan.
- [ ] Bedakan Otsu threshold pada image dari fixed reference threshold 128. Jangan merepresentasikan K-means sebagai single-intensity threshold.
- [ ] Optional widgets boleh ditambahkan dengan synchronized selection; static output dan parameter-cell fallback tetap berguna tanpa widget renderer.
- [ ] Jika raw data tidak ada, CSV ranking/score plots tetap jalan; gallery/pixel histogram menjelaskan missing inputs dan tidak berpura-pura telah executed.
- [ ] Export opt-in ke `reports/figures/`; render dari actual numerical arrays. Bahasa dan labels alami, concise, readable.

## Task 8 — verifikasi, visual QA, dan penyelesaian

- [ ] Jalankan meaningful tests: `python -m pytest -q`. Jangan berhenti setelah foundation tests saja.
- [ ] Validate notebook schema dengan `nbformat`; execute 01–05 dan 05b dari fresh kernel sesuai default artifact-reading mode. Jangan menggunakan wildcard yang tanpa sengaja menjalankan future final-test notebooks milik pengguna.
- [ ] Jalankan normal Jupyter execution bila environment mendukung, misalnya `python -m jupyter nbconvert --execute --to notebook --inplace notebooks/05b_segmentation_case_analysis.ipynb`; lakukan juga untuk lima notebook lain yang dimigrasikan.
- [ ] Jika kernel startup blocked, jelaskan alasan, alternate execution bila dilakukan, dan checks yang belum dijalankan. Schema-only pass tidak cukup untuk claim successful notebook execution.
- [ ] Verifikasi 13 methods, 372 validation IDs, 4.836 rows dan rounded macro checkpoint. Verifikasi paired counts tanpa mengubah scores.
- [ ] Periksa gallery, error legends, histogram threshold lines, signed axes, static fallback, font sizes, dan clipping pada normal viewing size.
- [ ] Hash historical artifacts lagi; frozen split, model JSONs, score CSVs, dan historical run metadata harus tetap identik. Bedakan source migration yang disengaja dari protected scientific inputs.
- [ ] README menjelaskan setup, dataset/cache placement, notebook order, case controls, export, current negative finding, dan batas grouping/provenance.
- [ ] Laporan akhir ringkas: changed files, commands/checks aktual, notebook execution dan visual QA, missing inputs/limitations, cara membuka hasil, serta roadmap berikutnya.
- [ ] Tidak push/merge/deploy, tidak new tuning/training, dan tidak held-out test performance evaluation.

Selesai berarti seluruh scope migrasi dan analisis visual telah ditangani, dengan keterbatasan yang dilaporkan jujur bila environment/data menghalangi verifikasi. Feature extraction, SVM, dan C++ tetap task berikutnya.
