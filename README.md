# EBHI-DIP — Segmentasi Klasik, Klasifikasi SVM, & Aplikasi Desktop C++/Qt6 Histopatologi EBHI-SEG

Repositori ini berisi *pipeline* pengolahan citra digital klasik (*Otsu thresholding*, *clustering* ruang warna fisik CIE Lab `MiniBatchKMeans`, dan pascapemrosesan morfologi), ekstraksi 29 fitur (*CIE Lab*, tekstur kuantisasi 16-level *GLCM*, dan deskriptor morfologi komponen terhubung), klasifikasi *Support Vector Machine* (SVM) 6 kelas (*One-vs-One*), evaluasi akhir satu kali (*single-shot*) pada subset *held-out* `test`, serta **aplikasi desktop GUI C++17 (Qt6 + OpenCV)** pada dataset histopatologi kolorektal **EBHI-SEG**.

---

## Ringkasan Kohort & Kontrak Ilmiah

Kohort eksperimen menggunakan *frozen split* otoritatif (`data/metadata/split_manifest.csv`, SHA-256 `ceeba8e282f77cdb78a9675618b3dcb6729fbc2339e2d112085a11720fc9108d`):
- **2.228** citra asli (`224×224`), **2.226** pasangan citra–*mask* valid lintas 6 kategori diagnostik (`Adenocarcinoma`, `High-grade IN`, `Low-grade IN`, `Normal`, `Polyp`, `Serrated adenoma`). Dua citra `Low-grade IN` tanpa *mask* (`GT2012149-2-400-001.png` dan `GTGT2012149-2-400-001.png`) terbukti duplikat piksel dan dikeluarkan dari kohort berpasangan.
- Pembagian berbasis **353** *canonical uppercase filename groups*: **1.535 `train`** (`247` grup) / **372 `val`** (`53` grup) / **319 `test`** (`53` grup), tanpa tumpang tindih grup maupun sidik jari piksel.
- Referensi biner ditetapkan sebagai `mask >= 128` tanpa memodifikasi *raw masks* (seluruh 35 *full-foreground masks* tetap dipertahankan).

---

## Ringkasan Hasil Validasi (`val`) & Evaluasi Akhir (`test`)

### 1. Evaluasi Segmentasi (`372 val` vs `319 test`, *equal-weight mean* lintas 6 kategori)

| Komparator | Method ID | Val Mean Category Dice | Val Mean Category IoU | Test Mean Category Dice (95% CI) | Test Mean Category IoU (95% CI) |
|---|---|---:|---:|---:|---:|
| Raw Otsu (`dark` polarity) | `otsu_M0` | 0.7232 | 0.5788 | 0.7098 `[0.6976, 0.7267]` | 0.5653 `[0.5517, 0.5855]` |
| Processed Otsu (`M2`) | `otsu_M2` | 0.7288 | 0.5886 | 0.7094 `[0.6975, 0.7285]` | 0.5666 `[0.5535, 0.5887]` |
| **Primary Lab K-means (`k=4, M0`)** | **`k4_M0`** | **0.7761** | **0.6427** | **0.7687 `[0.7575, 0.7859]`** | **0.6351 `[0.6208, 0.6597]`** |
| Processed Lab K-means (`k=4, M1`) | `k4_M1` | 0.7760 | 0.6426 | 0.7685 `[0.7573, 0.7858]` | 0.6349 `[0.6205, 0.6595]` |
| All-foreground control | `all_foreground` | 0.7767 | 0.6477 | 0.7587 `[0.7478, 0.7758]` | 0.6274 `[0.6143, 0.6499]` |

- Pada subset `val`, **`k4_M0`** merupakan kandidat *image-dependent* terbaik (`0.7761` Dice) dan dikunci di `configs/segmentation_selection.json`.
- Pada subset *held-out* `test` (`319` citra, `53` grup), **`k4_M0` (`0.7687` Dice) melampaui kontrol konstan `all_foreground` (`0.7587` Dice)** dengan selisih **`+0.0099` Dice (`95% CI [+0.0014, +0.0266]`)**, mengungguli `otsu_M0` pada **245/319 citra (`+0.0588` Dice)**, dan mengungguli `otsu_M2` pada **213/319 citra (`+0.0593` Dice)**.

### 2. Klasifikasi SVM 6 Kelas

Studi klasifikasi dipisahkan secara eksplisit antara protokol **konfigurasi 2×3 final berdisiplin kebocoran `v3` (`2x3-strict`, `train` + `val`)** dan **rekam historis `v1` + Tahap 08 (`test`)**:

#### 2a. Protokol Final `v3` (`2x3-strict`, *Fold-Aware Upstream CV* & *Validation-Blind* Grid Search)

Artefak di `results/classification_2x3_strict/` dan `models/cv_2x3_strict/` (dengan `v2` *fixed-upstream CV* disimpan terpisah di `results/classification_2x3/` dan `models/cv_2x3/`). Pada `v3`, seluruh komponen *upstream* berparameter (`otsu` *polarity check*, `MiniBatchKMeans(k=4)` + *supervised cluster mapping* `rate >= 0.50`, serta `StandardScaler`) dipasang ulang **di dalam tiap fold latih *5-fold StratifiedGroupKFold*** (`1.535` citra, `247` grup), grid `32` kandidat per model (**480 baris total**) bersifat ***validation-blind*** (`0` kolom `val_*`), model terpilih (`cv_selection_key_for_svm`) dievaluasi **sekali** pada `val` (`372` citra, `53` grup), dan ketidakpastian diestimasi dengan **bootstrap berpasangan terblok grup (`53 val groups`, `B = 2.000`, `seed = 42`)**. Subset *held-out* `test` (`319` citra) **sama sekali tidak diakses**.

| Track & Konfigurasi | Feature Set | Model Terpilih CV (`cv_selected`) | Strict CV Macro F1 | Val Macro F1 (95% Bootstrap CI) | Val Bal Acc | Val Accuracy |
|---|---|---|---:|---:|---:|---:|
| `shared_appearance` (Baseline) | `A` (10 fitur) | `rbf, none, C=10, g=scale` | 0.3346 | 0.3140 `[0.2304, 0.4166]` | 0.3207 | 49.73% |
| **`otsu_M0` (Top Overall)** | **`C` (29 fitur)** | **`linear, none, C=3`** | **0.4642** | **0.4058 `[0.3316, 0.5025]`** | **0.4016** | **56.72%** |
| `otsu_M1` | `C` (29 fitur) | `rbf, balanced, C=3, g=0.01` | 0.4481 | 0.3470 `[0.2669, 0.4439]` | 0.4171 | 45.16% |
| `otsu_M2` | `C` (29 fitur) | `rbf, balanced, C=3, g=0.01` | 0.4622 | 0.3462 `[0.2666, 0.4345]` | 0.4240 | 44.62% |
| `k4_M0` | `C` (29 fitur) | `rbf, balanced, C=3, g=scale` | 0.3707 | 0.3386 `[0.2624, 0.4277]` | 0.4018 | 44.62% |
| `k4_M1` | `C` (29 fitur) | `rbf, balanced, C=10, g=0.01` | 0.3643 | 0.2951 `[0.2174, 0.3813]` | 0.3489 | 38.98% |
| `k4_M2` | `C` (29 fitur) | `rbf, balanced, C=3, g=scale` | 0.3766 | 0.3055 `[0.2410, 0.3829]` | 0.3587 | 41.13% |
| `otsu_M0` (Best Set B) | `B` (19 fitur) | `rbf, balanced, C=10, g=0.01` | 0.4126 | 0.3280 `[0.2712, 0.4113]` | 0.3781 | 43.55% |
| `otsu_M1` / `otsu_M2` | `B` (19 fitur) | `rbf, balanced, C=10, g=0.01` | 0.3789 / 0.4077 | 0.3033 / 0.3055 | 0.3576 / 0.3393 | 38.17% / 41.13% |
| `k4_M0` / `k4_M1` / `k4_M2` | `B` (19 fitur) | `rbf, balanced` (`C=10/10/3`) | 0.2744 / 0.2797 / 0.2633 | 0.2490 / 0.2311 / 0.2072 | 0.2790 / 0.2829 / 0.2173 | 33.33% / 32.80% / 29.57% |
| Sensitivitas `k4_M0` (`A_seg` / `C_seg`) | `10` / `29` FG | `linear, bal, C=1` / `rbf, bal, C=10, 0.01` | 0.3709 / 0.3897 | 0.2575 / 0.2572 | 0.3177 / 0.2593 | 30.91% / 35.75% |

- **Tiga kontras berpasangan dengan *95% bootstrap CI* di atas nol (`B = 2.000`, `82.3%` undian memuat keenam kelas; `prop>0` = proporsi replikasi bootstrap dengan selisih > 0, bukan probabilitas posterior Bayesian):**
  - `otsu_M0_C - shared_A = +0.0919 [+0.0190, +0.1788], prop>0 = 0.9890` (morfologi mentah Otsu menambah informasi di atas warna/GLCM saja).
  - `otsu_M0_B - k4_M0_B = +0.0790 [+0.0316, +0.1417], prop>0 = 0.9995` (morfologi murni Otsu jauh melampaui `k4`).
  - `otsu_M0_C - otsu_M0_B = +0.0778 [+0.0060, +0.1483], prop>0 = 0.9840` (menggabungkan *appearance* A dan morfologi Otsu B nyata melampaui Set B saja).
- **Analisis sensitivitas *conditional all-6-class bootstrap* (`2.000` undian diterima dari `2.432` penarikan; *acceptance rate* `82.24%`, `432` penarikan ditolak karena kehilangan kelas minoritas, `seed = 42`):** mewajibkan keenam kelas hadir mempersempit lebar *95% CI* Macro-F1 lintas seluruh 15 model (rasio lebar `0.714–0.940`; mis. `otsu_M0_C` `[0.3272, 0.4726]` vs `[0.3316, 0.5026]`) karena memangkas bias ekor atas saat `Serrated adenoma`/`Normal` absen, namun **tidak mengubah kesimpulan substantif**: ketiga kontras utama tetap mengecualikan nol (`[+0.0238, +0.1748]` `prop>0 = 0.9965`; `[+0.0340, +0.1385]` `prop>0 = 1.0000`; `[+0.0038, +0.1405]` `prop>0 = 0.9805`), seluruh 15 kontras Balanced Accuracy/Accuracy tidak berubah tanda, dan hanya satu kontras batas `otsu_M0_C - otsu_M1_C` bergeser tipis melintasi nol (`[-0.0015, +0.1231]` $\rightarrow$ `[+0.0003, +0.1214]`).
- **Antar-konfigurasi Set C:** `otsu_M0_C` unggul atas `k4_M0_C` sebesar `+0.0672 [-0.0060, +0.1532], prop>0 = 0.9640` (`Val Acc +12.10 pp [+4.15, +21.09 pp]`) dan atas `otsu_M2_C` sebesar `+0.0596 [-0.0091, +0.1330], prop>0 = 0.9525` (`Val Acc +12.10 pp [+4.32, +20.00 pp]`), sementara pembatasan `k4_M0_C_seg` tertinggal dari unmasked `k4_M0_C` (`-0.0815 [-0.1296, -0.0300]`).
- **Dampak penghapusan kebocoran *upstream* (`v2` $\rightarrow$ `v3`):** `otsu` tetap invarian pada semua 5 fold (polaritas `"dark"` margin Dice $\ge +0.2841$; paritas fitur `<= 2.22e-16`), sehingga seluruh 6 model Otsu dan `Shared Set A` identik antara `v2` dan `v3`. Sebaliknya, *refitting* `k4` per fold mengubah *hyperparameter* pada **6/7 model `k4`** dan skor CV pada **7/7 model `k4`** (`k4_M0_B` CV `0.2937 -> 0.2744`; `k4_M0_C_seg` Val `0.2942 -> 0.2572`).
- **Hubungan deskriptif Dice segmentasi vs klasifikasi ($n=6$):** berkorelasi **negatif** (`Set B`: Pearson deskriptif `r = -0.9799` CV / `-0.9478` Val; `Set C`: `r = -0.9876` CV / `-0.7706` Val), karena `k4` menutupi `85%–87%` area gambar sehingga melebur kompleks epitel–stroma, sedangkan Otsu (`53%–54%`) mempertahankan struktur komponen epitel/nukleus yang informatif.

#### 2b. Rekam Historis Evaluasi Akhir `v1` / Tahap 08 (`372 val` vs `319 test`, Dipertahankan Utuh)

Tabel historis berikut mencatat artefak `v1` (`results/classification/`, `models/svm_*.json`) serta evaluasi *single-shot held-out test* Tahap 08 (`results/test/`). Karena protokol `v1` memakai penyetelan berbasis `val` pada `k4_M0_C` sebelum perluasan `2x3` (`v2`/`v3`), hasil `test` di bawah ini **milik `v1` saja dan tidak dievaluasi ulang pada `v2`/`v3`**:

| Track Model (`v1`) | Artifact JSON | Mask & Himpunan Fitur | Kernel (`C`, `gamma`, `weight`) | Val Macro F1 | Test Macro F1 (95% CI) | Test Balanced Acc | Test Accuracy |
|---|---|---|---|---:|---:|---:|---:|
| Ablasi Set A (Appearance) | `models/svm_k4_M0_A.json` | Tanpa mask (`Set A`: 10 fitur Lab+GLCM) | RBF (`1.0`, `scale`, `balanced`) | 0.3420 | 0.3313 `[0.2253, 0.4111]` | 0.3654 | 36.99% |
| Ablasi Set B (Morphology) | `models/svm_k4_M0_B.json` | `k4_M0` (`Set B`: 19 fitur morfologi) | RBF (`3.0`, `0.1`, `none`) | 0.2608 | 0.2094 `[0.1659, 0.2519]` | 0.2216 | 39.81% |
| **Primary Locked Track (`v1`)** | **`models/svm_k4_M0_selected.json`** | **`k4_M0` (`Set C`: 29 fitur gabungan)** | **RBF (`10.0`, `0.01`, `none`)** | **0.3625** | **0.3507 `[0.2545, 0.4171]`** | **0.3313** | **48.28%** |
| Segmented-Appearance (`Set C_seg`) | `models/svm_k4_M0_C_seg.json` | `k4_M0` (`Set C_seg`: 29 fitur fokus FG) | RBF (`10.0`, `scale`, `none`) | 0.3182 | 0.3297 `[0.2607, 0.3847]` | 0.3264 | 47.96% |
| Comparator Otsu Raw (`Set C`) | `models/svm_otsu_M0_C.json` | `otsu_M0` (`Set C`: 29 fitur gabungan) | RBF (`10.0`, `0.01`, `none`) | 0.3976 | 0.4075 `[0.3111, 0.4860]` | 0.3917 | 54.86% |
| Best Classifier Comparator (`v1`) | `models/svm_otsu_M2_C.json` | `otsu_M2` (`Set C`: 29 fitur gabungan) | RBF (`10.0`, `0.01`, `none`) | 0.4587 | 0.4117 `[0.3435, 0.4616]` | 0.4044 | 59.25% |

---

## Setup Environment Python (`.venv`)

Gunakan *virtual environment* `.venv` pada repositori ini dan pasang paket `ebhi_dip` secara *editable* beserta dependensi *notebook* dan pengujian:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks,dev]"
```

Di **VS Code**, buka file `.ipynb` pada folder `notebooks/`, pilih kernel **`.venv` (`Python 3.14.0`)**, lalu gunakan **Restart -> Run All**. Daftar versi paket teruji tercatat pada `requirements-lock.txt`.

---

## Penempatan Data dan Konfigurasi *Data-Root*

File dataset mentah dan *cache* piksel berukuran besar diabaikan oleh Git (`.gitignore`):
- Dataset mentah: `data/raw/EBHI-SEG/<Kategori>/{image,label}/*.png`
- Cache sampel piksel latih (opsional): `data/processed/kmeans_train_pixels.npz`

Apabila menjalankan kode atau *notebook* dari *worktree* terpisah sementara dataset berada di *checkout* utama (`D:\ebhi-dip`), arahkan pembacaan citra mentah tanpa mengubah path relatif di dalam *manifest*:
- Melalui environment variable: `$env:EBHI_DIP_DATA_ROOT = "D:\ebhi-dip"`
- Melalui parameter di sel awal *notebook*: `DATA_ROOT_OVERRIDE = Path(r"D:\ebhi-dip")`
- Melalui argumen CLI pada `scripts/`: `--data-root "D:\ebhi-dip"`

Jika citra mentah tidak tersedia, seluruh analisis berbasis CSV/JSON tetap berjalan dan bagian visualisasi piksel/galeri menampilkan status eksplisit tanpa menghasilkan gambar sintetis.

---

## Urutan dan Peta Notebook (`01`–`07` Rutin, plus `08` Rekam Historis `v1`)

| Notebook | Fokus pertanyaan & isi utama |
|---|---|
| `notebooks/01_dataset_audit.ipynb` | Kelengkapan 2.226 pasangan citra–*mask*, bukti duplikasi 2 file `missing_mask`, audit nilai abu-abu menengah & 35 *full-foreground masks*, serta *overlay* citra `train`. |
| `notebooks/02_split_audit.ipynb` | Verifikasi *raw-byte* `split_manifest.csv`, normalisasi kapitalisasi 353 `group_id`, bukti nol tumpang tindih antar `train`/`val`/`test`, dan distribusi kategori. |
| `notebooks/03_otsu_baseline.ipynb` | Pemilihan polaritas `dark` pada `train`, rekonsiliasi hasil `val` (`M0`, `M1`, `M2`), contoh visual *threshold* per citra, dan histogram *grayscale*. |
| `notebooks/04_lab_kmeans.ipynb` | Konversi fisik CIE Lab (`float32`), provenance *sampling* `1.535.000` piksel `train`, pusat *cluster* `k=2,3,4` & pemetaan `rate >= 0.5`, serta contoh inferensi penuh. |
| `notebooks/05_segmentation_comparison.ipynb` | Rekonsiliasi 13 metode pada 372 citra `val` (`4.836` baris), perbandingan *equal-category mean* terhadap kontrol `all_foreground`, *heatmap* kategori, dan *coverage*. |
| `notebooks/05b_segmentation_case_analysis.ipynb` | Ranking kasus (`worst`, `best`, `advantage`, `both_poor`, `ties`), *paired scatter*, histogram skor/delta, galeri 2×3 dengan *error overlay* (`TP`/`FP`/`FN`), dan histogram intensitas/Lab 6-panel. |
| `notebooks/06_feature_extraction.ipynb` | Ekstraksi fitur 2×3 lengkap (`otsu_M0..k4_M2`): `Set A` bersama (10 fitur CIE Lab + GLCM), `Set B` morfologi (19 fitur per konfigurasi), `Set C` (29 fitur), sensitivitas `A_seg`/`C_seg`, batas ketergantungan *upstream* per-citra vs *train-fit*, dan uji paritas `< 1e-12`. |
| `notebooks/07_svm_classification.ipynb` | Klasifikasi SVM 2×3 berdisiplin kebocoran `v3` (`2x3-strict`): audit *fold-aware* Otsu & `k4`, grid 32 kandidat *validation-blind*, bootstrap berpasangan `53` grup `val`, asosiasi deskriptif Dice vs Macro-F1 ($n=6$), kontras protokol `v1`/`v2`/`v3`, dan paritas 15 model `models/cv_2x3_strict/*.json`. |
| `notebooks/08_final_test_evaluation.ipynb` | Rekam historis evaluasi satu kali (*single-shot*) milik **`v1`** pada 319 citra `test` (`53` grup) dengan *group-blocked bootstrap 95% CI* (`2.000` *resamples*, `seed=42`); tidak dijalankan ulang oleh `v2`/`v3`. |

---

## Aplikasi Desktop GUI C++ & CLI (`cpp/`)

Implementasi C++17 berada di direktori `cpp/` dan menggunakan **OpenCV** (`core`, `imgproc`, `imgcodecs`) serta **Qt6** (`Core`, `Gui`, `Widgets`) tanpa ketergantungan pada runtime Python. Seluruh rumus konversi warna CIE Lab, *Otsu thresholding*, *K-Means assignment*, morfologi `M0`/`M1`/`M2`, ekstraksi 29 fitur (termasuk *4-connected Benkrid & Crookes perimeter* dan *exact integer convex-hull pixel area* untuk `solidity`), serta inferensi SVM *One-vs-One* memiliki paritas numerik langsung terhadap paket Python (`tests/test_cpp_parity.py`).

### 1. Cara Kompilasi (`CMake` + `Ninja` pada MSYS2 UCRT64 / Windows)

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;C:\Program Files\CMake\bin;" + $env:PATH
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j 8
```

Target kompilasi yang dihasilkan di `cpp/build/`:
- `libebhi_core.a` — Pustaka inti C++ (`cpp/include/ebhi_pipeline.hpp`, `cpp/src/ebhi_pipeline.cpp`).
- `ebhi_cli.exe` — Antarmuka baris perintah (CLI) untuk inferensi citra tunggal, ekspor *mask*/*overlay*, dan verifikasi paritas JSON (`cpp/src/main_cli.cpp`).
- `ebhi_gui.exe` — Aplikasi desktop GUI interaktif berbasis Qt6 Widgets (`cpp/src/main_gui.cpp`).

### 2. Menjalankan Aplikasi Desktop GUI (`ebhi_gui.exe`)

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
.\cpp\build\ebhi_gui.exe --repo-root "D:\ebhi-dip"
```

Fitur utama `ebhi_gui.exe`:
- Pemilih cepat sampel citra dari ke-6 kategori histopatologi maupun dialog buka file `.png` bebas (`Pilih Citra RGB...` & deteksi otomatis *ground-truth mask* pada folder `label/`).
- Pemilihan metode segmentasi (`k4_M0`, `otsu_M2`, `k4_M1`, `k4_M2`, `otsu_M0`, `otsu_M1`, `k3_M0`, `k2_M0`) dan artefak model SVM (`svm_k4_M0_selected.json`, `svm_otsu_M2_C.json`, `svm_k4_M0_A.json`, `svm_k4_M0_B.json`, dll.).
- Empat panel visual berdampingan: **(1) Citra Asli RGB**, **(2) Prediksi Mask Segmentasi**, **(3) Ground Truth Reference Mask**, dan **(4) Error Overlay (`TP` Hijau / `FP` Oranye / `FN` Magenta)**.
- Panel distribusi suara *One-vs-One* (`OVO` `/15` pasangan pengklasifikasi), metrik `Dice`/`IoU`/`Precision`/`Recall`, serta tabel lengkap **29 fitur** (nilai mentah dan *standardized z-score*).

### 3. Menjalankan Inferensi CLI (`ebhi_cli.exe`)

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
.\cpp\build\ebhi_cli.exe `
  --image "D:\ebhi-dip\data\raw\EBHI-SEG\Adenocarcinoma\image\GT2012165-1-400-001.png" `
  --mask "D:\ebhi-dip\data\raw\EBHI-SEG\Adenocarcinoma\label\GT2012165-1-400-001.png" `
  --seg k4_M0 `
  --svm "D:\ebhi-dip\models\svm_k4_M0_selected.json" `
  --repo-root "D:\ebhi-dip"
```

---

## Struktur Direktori

- `src/ebhi_dip/` — Paket inti Python: `paths.py`, `dataset.py`, `metrics.py`, `provenance.py`, `segmentation.py`, `analysis.py`, `plotting.py`, `features.py`, `classification.py`, `strict_cv.py`, `evaluation.py`.
- `cpp/` — Aplikasi C++17 (`CMakeLists.txt`, `include/ebhi_pipeline.hpp`, `src/ebhi_pipeline.cpp`, `src/main_cli.cpp`, `src/main_gui.cpp`).
- `notebooks/` — Notebook riset terurut (`01`–`08`, termasuk `05b`).
- `scripts/` — *Thin CLI runners* dengan penjaga `--overwrite` (`extract_features.py`, `evaluate_svm_val.py`, `evaluate_final_test.py`, dll.).
- `configs/` — Konfigurasi eksperimen (`otsu_raw.json`, `segmentation_selection.json`, `provenance_migration.json`).
- `data/metadata/` — Manifest, audit, grup nama file, serta `split_manifest.csv` + `.sha256` yang dibekukan.
- `models/` — Artefak JSON pusat *cluster* Lab K-means (`kmeans_lab_k*_mapped.json`), model SVM historis `v1` (`svm_*.json`), model `v2` (`cv_2x3/`), dan model kanonik `v3` (`cv_2x3_strict/`).
- `results/` — Skor validasi segmentasi, tabel fitur 2×3 (`results/features/`), hasil SVM historis `v1` (`results/classification/`), `v2` (`results/classification_2x3/`), `v3` final (`results/classification_2x3_strict/`, termasuk analisis sensitivitas *conditional all-6 bootstrap*), dan hasil *test* historis `v1` (`results/test/`).
- `reports/` — Laporan analisis ilmiah final (`reports/laporan_analisis_ilmiah_ebhi_dip.md`) dan seluruh figur ilmiah terkurasi (`reports/figures/`).
- `docs/` — Dokumentasi garis keturunan provenance (`docs/provenance_migration.md`).
- `tests/` — Pengujian kontrak numerik, integritas kohort, provenance, ekstraksi fitur 2×3, klasifikasi SVM `v1`/`v2`/`v3` (*fold-aware* + *val-blind* + *conditional bootstrap*), evaluasi historis `test`, serta paritas C++ vs Python.

---

## Verifikasi dan Pengujian

Jalankan seluruh *unit & parity tests* (termasuk pengujian paritas CLI dan GUI C++ terhadap Python):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Jalankan evaluasi klasifikasi 2×3 protokol ketat (`v3`, memakai *fold cache* di `data/processed/strict_cv_2x3/` bila tersedia):

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_svm_val.py --protocol 2x3-strict --workers 8
```

### Verifikasi *Notebook* Rutin (`01`–`07`, termasuk `05b`)

Eksekusi ulang *Run All* rutin **hanya mencakup notebook `01`–`07`** (`08` dikecualikan agar subset *held-out* `test` tetap terisolasi):

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/01_dataset_audit.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/02_split_audit.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/03_otsu_baseline.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/04_lab_kmeans.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/05_segmentation_comparison.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/05b_segmentation_case_analysis.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/06_feature_extraction.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/07_svm_classification.ipynb
```

### Reproduksi Terpisah: Evaluasi Akhir Historis `v1` (`notebooks/08_final_test_evaluation.ipynb`)

`notebooks/08_final_test_evaluation.ipynb` beserta artefak pada `results/test/` merupakan **rekam evaluasi satu kali historis milik protokol `v1`** (`results/classification/`, `models/svm_*.json`) yang **tidak** berlaku untuk model `v2`/`v3` dan **dikeluarkan dari alur `Run All` rutin**. Hanya jalankan secara terpisah apabila reproduksi historis `v1` memang diperlukan secara ilmiah:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/08_final_test_evaluation.ipynb
```

---

## Batas Metodologis dan Provenance

1. **Batas pengelompokan nama file:** `group_id` diturunkan dari prefix nama file sebelum `-` (*uppercase*) dan merupakan **kunci pengelompokan sementara**, bukan ID pasien klinis yang terverifikasi.
2. **Makna biologis segmentasi warna/intensitas:** baik Otsu maupun Lab K-means memisahkan kecerahan/warna pewarnaan H&E secara global; *mask* yang dihasilkan adalah representasi kompartemen warna jaringan dan **bukan** segmentasi nukleus, kelenjar, atau area kanker yang tervalidasi secara klinis.
3. **Integritas frozen split:** `data/metadata/split_manifest.csv` diverifikasi menggunakan *raw bytes* secara ketat; normalisasi *line-ending* dilarang pada file tersebut.
4. **Lineage kode sumber (`utf8-lf-v1`):** perbedaan *hash* `CRLF` vs `LF` pada *helper* historis ditangani secara eksplisit melalui `configs/provenance_migration.json` dan `docs/provenance_migration.md`.
5. **Disiplin subset `test` & batasan validasi 2×3:** evaluasi historis `test` (`319` citra, Tahap 08) hanya dilakukan pada artefak `v1`. Pada protokol **`v2` dan `v3` (`2x3-strict`)**, `test` sama sekali tidak disentuh (`0` citra). Korelasi Dice vs Macro-F1 lintas $n=6$ konfigurasi bersifat deskriptif saja, dan evaluasi `val` dibatasi oleh jumlah grup kelas minoritas yang sangat kecil (`2` grup `Serrated adenoma`, `3` grup `Normal`; `82.3%` penarikan unconditional bootstrap memuat keenam kelas, dan sensitivitas *conditional all-6-class* mencatat *acceptance rate* `82.24%` dengan `432` penarikan ditolak dari `2.432` total).
