# EBHI-DIP — handoff untuk Codex

## Mulai di sini

Pengguna menyetujui struktur notebook dan meminta **seluruh migrasi pekerjaan yang sudah ada**, termasuk analisis per citra dan histogram, diselesaikan dalam satu task. Tidak perlu berhenti meminta persetujuan pada setiap tahap. Brief ini menggantikan instruksi lama “Phase 1 only” dan “stop at this checkpoint”.

- Repository: https://github.com/savitar393/ebhi-dip
- Checkout utama: `D:\ebhi-dip`, Windows, VS Code, `.venv`.
- Checkpoint yang ditinjau: `8192e3e09384b05fa760104436f3a2e226380ac3`, ditinjau 2026-11-05.
- Periksa checkout aktual dan perubahan lokal. Checkpoint adalah referensi, bukan perintah reset. Pertahankan pekerjaan pengguna dan hasil baru yang sah.

Urutan baca:

1. `codex_handoff/CODEX_HANDOFF.md`.
2. `codex_handoff/VISUAL_ANALYSIS_REQUIREMENTS.md`.
3. `codex_handoff/NOTEBOOK_MIGRATION.md`.
4. `codex_handoff/docs/superpowers/plans/2026-11-05-notebook-migration-and-case-analysis.md`.
5. Kode, schema, konfigurasi, dan artifacts pada checkout aktual.

Jika instruksi sebelumnya bertentangan, brief ini dan tambahan pengguna terbaru berlaku. Jangan mengubah data agar cocok dengan checkpoint bila pekerjaan lokal telah berubah secara sah; jelaskan perbedaannya.

## Hasil yang diminta

1. Package kecil `src/ebhi_dip/` untuk numerical core, paths, metrics, provenance, dan reusable plotting.
2. Notebook `01`–`05` yang mengonsolidasikan pekerjaan yang sudah ada.
3. `05b_segmentation_case_analysis.ipynb`: ranking citra, paired comparisons, gallery, error overlays, histogram citra dan skor.
4. Thin batch runners, environment specification, README, dan checks yang memadai.
5. Laporan akhir ringkas: perubahan, verifikasi aktual, keterbatasan, dan cara menjalankan.

Python mendukung eksperimen dan analisis; aplikasi akhirnya tetap **C++ desktop GUI**. GUI framework belum dipilih. Feature extraction, SVM, GUI, eksperimen baru, dan final test evaluation adalah pekerjaan berikutnya, bukan bagian migrasi ini.

## Bahasa dan keterbacaan — preferensi terbaru pengguna

- Gunakan bahasa Indonesia untuk penjelasan, interpretasi, README, dan laporan. Pertahankan istilah yang lazim dalam bahasa Inggris: *ground truth*, *foreground*, *background*, *baseline*, *threshold*, *split*, *overlay*, *pipeline*, Dice, IoU, Otsu, K-means, dan Lab.
- Jangan memaksakan terjemahan. Label alami seperti “Histogram grayscale”, “Jumlah piksel”, “Selisih Dice”, dan “Foreground coverage” diperbolehkan.
- Pertahankan code identifiers, nama file, metadata schema, sample IDs, method IDs, serta category labels asli. Tidak perlu menerjemahkan setiap kata atau UI/library term.
- Buat paragraf pendek. Gunakan tabel atau daftar untuk informasi sejajar; hindari pengulangan dan penjelasan textbook yang tidak membantu analisis.
- Tiap bagian cukup menjawab: **apa yang diperiksa, mengapa, hasilnya, dan maknanya**. Bedakan observasi, dugaan penyebab, dan keterbatasan.
- Code cells harus mudah dibaca. Gunakan fungsi jelas untuk logika berulang; hindari framework atau abstraksi berlebihan.
- Tidak perlu tutorial langkah demi langkah selama pengerjaan. Hasil akhir tetap explainable dan reviewable.

## Kondisi awal repository

Checkpoint memiliki 22 Python scripts di root, termasuk helpers `kmeans_prediction.py` dan `postprocessing.py`. Dua puluh scripts mengasumsikan root berada di `Path(__file__).resolve().parent`. Beberapa runner melakukan I/O/fitting saat di-import; `overlap_scores` terduplikasi di enam scripts.

Pekerjaan tersedia: dataset audit, filename grouping, frozen split, train-only Otsu polarity selection, global Lab K-means fitting dan foreground mapping, morphology variants, serta validation results.

Pada checkpoint belum ada notebook, README, dependency specification, feature extraction, classifier, aplikasi C++, atau committed `configs/segmentation_selection.json`. Periksa versi lokal sebelum membuat atau mengganti file.

## Scientific checkpoint yang harus dipertahankan

| Komponen | Kontrak |
|---|---|
| Cohort | 2.228 original images; 2.226 paired images/masks; 224×224, RGB images dan L masks. |
| Unpaired | Low-grade IN: `GT2012149-2-400-001.png` dan `GTGT2012149-2-400-001.png`; pixel duplicates, dikeluarkan dari paired cohort. |
| Reference | `mask >= 128`; raw masks tetap. Semua 35 full-foreground masks tetap dipakai. |
| Grouping | Token filename pertama sebelum `-`, uppercase lintas kategori: 353 groups. Ini **belum verified patient IDs**. |
| Frozen split | Train 1.535 / val 372 / test 319; groups 247 / 53 / 53. |
| Split file | `data/metadata/split_manifest.csv`; raw-byte checksum authoritative. |
| Split SHA-256 | `ceeba8e282f77cdb78a9675618b3dcb6729fbc2339e2d112085a11720fc9108d` |
| Otsu | Threshold per citra; global foreground polarity **dark**, dipilih pada train, tersimpan di `configs/otsu_raw.json`. |
| Lab | RGB uint8 → float32 / 255 → `cv2.COLOR_RGB2LAB`; tanpa scaling tambahan. |
| K-means | Global MiniBatchKMeans k=2,3,4; 1.000 posisi berbeda per train image, seed 42; total 1.535.000 Lab vectors. |
| Mapping | Foreground jika training-reference foreground rate >=0.5, memakai posisi sampling train yang sama. Mapping menggunakan training annotations. |
| Foreground clusters | k2 `[1]`; k3 `[0,2]`; k4 `[0,1,2]`. |
| Assignment | Saved centers; squared Euclidean distance float64; exact tie memilih cluster ID terendah. |
| M0 / M1 / M2 | M0 raw; M1 component area >=16, connectivity 8; M2 = M1 lalu opening dan closing, 3×3 cross, satu iteration masing-masing, `BORDER_REPLICATE`. |
| Primary metric | Mean image score per kategori, lalu equal-weight mean atas enam kategori. Both-empty Dice=IoU=1. |

Jangan regenerate, pandas-resave, atau normalize line endings frozen CSV maupun mengganti checksum-nya. Pertahankan original masks, learned JSON artifacts, historical metadata, dan result CSVs.

Validation checkpoint, dibulatkan untuk tampilan:

| Comparator | Dice | IoU |
|---|---:|---:|
| Otsu M0 | 0.7232 | 0.5788 |
| Otsu M2 | 0.7288 | 0.5886 |
| K-means k4/M0 | 0.7761 | 0.6427 |
| K-means k4/M1 | 0.7760 | 0.6426 |
| All-foreground control | 0.7767 | 0.6477 |

Raw k4/M0 adalah kandidat image-dependent terbaik pada checkpoint. Control masih memiliki mean lebih tinggi; pertahankan temuan tersebut. Belum ada bukti significance atau manfaat klasifikasi dari morphological descriptors. Jangan menyebut prediksi sebagai validated nuclei/gland/cancer segmentation.

Selection rule yang sudah digunakan: Dice empat desimal, lalu IoU empat desimal, lalu operasi lebih sedikit. Gunakan full precision untuk perhitungan dan artifacts.

## Analisis per citra dan histogram

Tujuannya melihat variasi performa antar citra. Untuk setiap citra terpilih, tampilkan **kedua metode pada citra yang sama**. Worst Otsu dan worst K-means boleh berbeda IDs; masing-masing tetap harus menampilkan kedua prediksi.

Default: validation cohort, `otsu_M0` vs `k4_M0`; secondary pair `otsu_M2` vs `k4_M1`. Ranking mencakup worst, best, keuntungan terbesar masing-masing, ties, dan kasus kedua metode lemah.

Checkpoint per-image Dice dari CSV:

| Pair | Otsu lebih tinggi | K-means lebih tinggi | Tie |
|---|---:|---:|---:|
| M0 vs k4/M0 | 96 | 276 | 0 |
| M2 vs k4/M1 | 122 | 250 | 0 |

Counts ini memakai toleransi tie `1e-12`, bukan primary equal-category mean. Ikuti detail gallery, histogram, overlays, dan acceptance criteria pada `VISUAL_ANALYSIS_REQUIREMENTS.md`.

Jangan menetapkan kesimpulan bahwa satu metode selalu unggul atau keduanya pasti gagal. Jelaskan pola yang didukung scores dan visualisasi. Tandai dugaan biologis atau penyebab kegagalan sebagai interpretasi.

## Provenance dan line endings

Frozen split dan mapped model JSON hashes cocok dengan snapshot. Recorded helper hashes:

- `kmeans_prediction.py`: `42a0a3a5c84689f7f3f221d80e27a28ee3949e4d3e3a64c1c553fc1513503299`.
- `postprocessing.py`: `b83e549480fec3f8373eb7d95db6a8194d53795d91611797f0125fc6c673f8ca`.

Committed helpers memakai LF; konversi bytes ke CRLF mereproduksi kedua recorded hashes secara tepat. Selesaikan melalui explicit versioned provenance adapter dan numerical parity checks. Pertahankan historical fingerprints; jangan mengganti expected hashes diam-diam, menonaktifkan guards, atau retrain karena file dipindah.

Scheme source-text baru boleh menormalkan newline/BOM secara eksplisit. Scheme ini tidak berlaku untuk frozen CSV. Perubahan kode aktual tetap harus terdeteksi. Dokumentasikan lineage historical artifacts ke implementasi baru.

Rule frozen split di `.gitattributes` benar. Beberapa rule lain menunjuk `data/configs/otsu_raw.csv` dan `data/results/...`, bukan path aktual. Tinjau tanpa broad renormalization historical artifacts.

## Environment dan notebook seed

Environment pengguna: Python 3.14.0, NumPy 2.5.4, Pillow 13.0.0, Matplotlib 3.11.2, opencv-python 4.14.0.94 (`cv2` 4.14.0), scikit-learn 1.9.1, SciPy 1.18.1. Periksa environment aktual; jangan mengubah historical runtime records.

Raw dataset `data/raw/EBHI-SEG/` dan `data/processed/kmeans_train_pixels.npz` Git-ignored. Worktree/clone tidak otomatis memilikinya. Bila diperlukan, gunakan explicit data-root configuration ke dataset yang sudah ada; writes tetap di working checkout.

`notebook_seed/05_segmentation_comparison.ipynb` membaca saved validation CSVs. Gabungkan dengan hati-hati bila notebook lokal sudah ada. Seed masih berbahasa Inggris; sesuaikan narasi dengan preferensi bahasa dan keterbacaan di atas, tanpa mengubah scientific meaning.

Seed telah schema-validated dan 11 code cells dijalankan berurutan menggunakan fresh IPython InProcessKernel; empat plots diperiksa visual. Separate Jupyter kernel terhalang socket binding pada environment persiapan. Jalankan normal kernel di target environment bila tersedia; laporkan metode eksekusi aktual.

Jika raw images tidak tersedia, analisis CSV tetap dapat dikerjakan. Gallery, pixel histograms, dan raw-image inference parity harus dinyatakan belum dijalankan; jangan membuat fake images atau outputs.

## Batas dan definisi selesai

- Selesaikan notebook 01–05 dan 05b, bukan hanya package foundation.
- Default notebook membaca artifacts. Fitting, split creation, dan heavy writes harus opt-in; jangan dijalankan dalam migrasi ini.
- Predictor menerima citra dan fixed artifacts, tanpa memilih cluster/polarity berdasarkan reference citra tersebut.
- Jangan evaluasi held-out test performance; metadata split boleh diperiksa untuk integrity.
- Migrasikan callers sebelum memindahkan/meretir legacy runners. Pertahankan Git history dan artifact lineage.
- Jangan push, merge, deploy, retrain, tune parameters baru, atau implementasikan SVM/GUI.
- Verifikasi historical bytes tetap dan numerical parity. Laporkan commands yang benar-benar dijalankan serta checks yang terhalang.
- Laporan akhir ringkas menyebut status execution, gallery, histogram, widgets bila ada, dan visual QA. Schema validation saja bukan successful execution.

Tidak perlu meminta ulang persetujuan arsitektur. Pengguna meninjau hasil konkret setelah seluruh scope yang disetujui selesai.
