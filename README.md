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

### 2. Klasifikasi SVM 6 Kelas (`372 val` vs `319 test`)

| Track Model | Artifact JSON | Mask & Himpunan Fitur | Kernel (`C`, `gamma`, `weight`) | Val Macro F1 | Test Macro F1 (95% CI) | Test Balanced Acc | Test Accuracy |
|---|---|---|---|---:|---:|---:|---:|
| Ablasi Set A (Appearance) | `models/svm_k4_M0_A.json` | Tanpa mask (`Set A`: 10 fitur Lab+GLCM) | RBF (`3.0`, `0.03`, `balanced`) | 0.4272 | 0.3313 `[0.2587, 0.3883]` | 0.3401 | 0.4138 |
| Ablasi Set B (Morphology) | `models/svm_k4_M0_B.json` | `k4_M0` (`Set B`: 19 fitur morfologi) | RBF (`0.3`, `scale`, `none`) | 0.2508 | 0.2094 `[0.1538, 0.2535]` | 0.2371 | 0.3699 |
| **Primary Locked Track** | **`models/svm_k4_M0_selected.json`** | **`k4_M0` (`Set C`: 29 fitur gabungan)** | **RBF (`10.0`, `scale`, `none`)** | **0.4357** | **0.3507 `[0.2545, 0.4171]`** | **0.3457** | **0.4828** |
| Segmented-Appearance (`Set C_seg`) | `models/svm_k4_M0_C_seg.json` | `k4_M0` (`Set C_seg`: 29 fitur fokus FG) | RBF (`3.0`, `scale`, `none`) | 0.4063 | 0.3563 `[0.2752, 0.4167]` | 0.3568 | 0.4671 |
| Comparator Otsu Raw (`Set C`) | `models/svm_otsu_M0_C.json` | `otsu_M0` (`Set C`: 29 fitur gabungan) | RBF (`10.0`, `scale`, `none`) | 0.4252 | 0.4090 `[0.3435, 0.4596]` | 0.4024 | 0.5831 |
| **Best Classifier Comparator** | **`models/svm_otsu_M2_C.json`** | **`otsu_M2` (`Set C`: 29 fitur gabungan)** | **RBF (`3.0`, `scale`, `none`)** | **0.4601** | **0.4117 `[0.3435, 0.4616]`** | **0.4044** | **0.5925** |

- **Ablasi fitur pada `k4_M0` konsisten di `val` maupun `test`:** penggabungan fitur warna/tekstur dan morfologi (`Set C`: `0.3507` Macro F1, `48.28%` akurasi) melampaui `Set A` (`0.3313` Macro F1, `41.38%` akurasi) maupun `Set B` (`0.2094` Macro F1, `36.99%` akurasi).
- **Trade-off antara kesesuaian *ground-truth* area jaringan vs daya pisah klasifikasi:** meskipun `k4_M0` unggul signifikan dalam segmentasi (`0.7687` vs `0.7094` Dice), *mask* `k4_M0` cenderung menyatukan komponen jaringan (`pred_fg_fraction` tinggi), sedangkan `otsu_M2` menghasilkan komponen epitel gelap yang lebih terpisah sehingga deskriptor morfologi `Set B` pada `otsu_M2` memberi daya pisah kategori lebih tinggi (`test_macro_f1 = 0.4117`, `test_accuracy = 59.25%`).

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

## Urutan dan Peta Notebook (`01`–`08`)

| Notebook | Fokus pertanyaan & isi utama |
|---|---|
| `notebooks/01_dataset_audit.ipynb` | Kelengkapan 2.226 pasangan citra–*mask*, bukti duplikasi 2 file `missing_mask`, audit nilai abu-abu menengah & 35 *full-foreground masks*, serta *overlay* citra `train`. |
| `notebooks/02_split_audit.ipynb` | Verifikasi *raw-byte* `split_manifest.csv`, normalisasi kapitalisasi 353 `group_id`, bukti nol tumpang tindih antar `train`/`val`/`test`, dan distribusi kategori. |
| `notebooks/03_otsu_baseline.ipynb` | Pemilihan polaritas `dark` pada `train`, rekonsiliasi hasil `val` (`M0`, `M1`, `M2`), contoh visual *threshold* per citra, dan histogram *grayscale*. |
| `notebooks/04_lab_kmeans.ipynb` | Konversi fisik CIE Lab (`float32`), provenance *sampling* `1.535.000` piksel `train`, pusat *cluster* `k=2,3,4` & pemetaan `rate >= 0.5`, serta contoh inferensi penuh. |
| `notebooks/05_segmentation_comparison.ipynb` | Rekonsiliasi 13 metode pada 372 citra `val` (`4.836` baris), perbandingan *equal-category mean* terhadap kontrol `all_foreground`, *heatmap* kategori, dan *coverage*. |
| `notebooks/05b_segmentation_case_analysis.ipynb` | Ranking kasus (`worst`, `best`, `advantage`, `both_poor`, `ties`), *paired scatter*, histogram skor/delta, galeri 2×3 dengan *error overlay* (`TP`/`FP`/`FN`), dan histogram intensitas/Lab 6-panel. |
| `notebooks/06_feature_extraction.ipynb` | Ekstraksi 29 fitur (`Set A`: 6 statistik CIE Lab + 4 tekstur *GLCM* 16-level; `Set B`: 4 cakupan *mask* + 15 statistik komponen `area_rel`, `perimeter_rel`, `circularity`, `eccentricity`, `solidity`) beserta varian fokus *foreground* (`Set A_seg`/`C_seg`). |
| `notebooks/07_svm_classification.ipynb` | Seleksi hiperparameter SVM *One-vs-One* (`StandardScaler` *train-only*, *5-fold GroupKFold* + evaluasi `val`), ablasi `Set A`/`B`/`C`, sensitivitas segmentasi (`k4_M0`, `otsu_M0`, `otsu_M2`, `k4_M1`), *confusion matrix*, dan ekspor artefak JSON portabel untuk C++. |
| `notebooks/08_final_test_evaluation.ipynb` | Evaluasi akhir satu kali (*single-shot*) pada 319 citra `test` (`53` grup) dengan *group-blocked bootstrap 95% CI* (`2.000` *resamples*, `seed=42`) untuk segmentasi dan klasifikasi SVM beserta galeri kasus `test`. |

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

- `src/ebhi_dip/` — Paket inti Python: `paths.py`, `dataset.py`, `metrics.py`, `provenance.py`, `segmentation.py`, `analysis.py`, `plotting.py`, `features.py`, `classification.py`, `evaluation.py`.
- `cpp/` — Aplikasi C++17 (`CMakeLists.txt`, `include/ebhi_pipeline.hpp`, `src/ebhi_pipeline.cpp`, `src/main_cli.cpp`, `src/main_gui.cpp`).
- `notebooks/` — Notebook riset terurut (`01`–`08`, termasuk `05b`).
- `scripts/` — *Thin CLI runners* dengan penjaga `--overwrite` (`extract_features.py`, `evaluate_svm_val.py`, `evaluate_final_test.py`, dll.).
- `configs/` — Konfigurasi eksperimen (`otsu_raw.json`, `segmentation_selection.json`, `provenance_migration.json`).
- `data/metadata/` — Manifest, audit, grup nama file, serta `split_manifest.csv` + `.sha256` yang dibekukan.
- `models/` — Artefak JSON pusat *cluster* Lab K-means (`kmeans_lab_k*_mapped.json`) dan model SVM portabel (`svm_*.json`).
- `results/` — Skor validasi segmentasi, tabel fitur (`results/features/`), hasil validasi SVM (`results/classification/`), dan hasil evaluasi akhir *held-out* `test` (`results/test/`).
- `docs/` — Dokumentasi garis keturunan provenance (`docs/provenance_migration.md`).
- `tests/` — Pengujian kontrak numerik, integritas kohort, provenance, ekstraksi fitur, klasifikasi SVM, evaluasi `test`, serta paritas C++ vs Python (`59 passed`).

---

## Verifikasi dan Pengujian

Jalankan seluruh *unit & parity tests* (termasuk pengujian paritas CLI dan GUI C++ terhadap Python):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Jalankan ulang seluruh *notebook* (`01`–`08`) dari *fresh kernel*:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/01_dataset_audit.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/02_split_audit.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/03_otsu_baseline.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/04_lab_kmeans.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/05_segmentation_comparison.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/05b_segmentation_case_analysis.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/06_feature_extraction.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/07_svm_classification.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/08_final_test_evaluation.ipynb
```

---

## Batas Metodologis dan Provenance

1. **Batas pengelompokan nama file:** `group_id` diturunkan dari prefix nama file sebelum `-` (*uppercase*) dan merupakan **kunci pengelompokan sementara**, bukan ID pasien klinis yang terverifikasi.
2. **Makna biologis segmentasi warna/intensitas:** baik Otsu maupun Lab K-means memisahkan kecerahan/warna pewarnaan H&E secara global; *mask* yang dihasilkan adalah representasi kompartemen warna jaringan dan **bukan** segmentasi nukleus, kelenjar, atau area kanker yang tervalidasi secara klinis.
3. **Integritas frozen split:** `data/metadata/split_manifest.csv` diverifikasi menggunakan *raw bytes* secara ketat; normalisasi *line-ending* dilarang pada file tersebut.
4. **Lineage kode sumber (`utf8-lf-v1`):** perbedaan *hash* `CRLF` vs `LF` pada *helper* historis ditangani secara eksplisit melalui `configs/provenance_migration.json` dan `docs/provenance_migration.md`.
5. **Disiplin subset `test`:** subset `test` (`319` citra) hanya dievaluasi satu kali setelah seluruh parameter segmentasi (`configs/segmentation_selection.json`) dan model SVM (`models/svm_*.json`) dikunci pada `train`/`val`.
