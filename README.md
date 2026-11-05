# EBHI-DIP — Segmentasi Klasik & Analisis Kasus Histopatologi EBHI-SEG

Repositori ini berisi *pipeline* pengolahan citra digital klasik (*Otsu thresholding*, *clustering* ruang warna fisik CIE Lab `MiniBatchKMeans`, dan pascapemrosesan morfologi) beserta analisis kasus per citra pada dataset histopatologi kolorektal **EBHI-SEG**.

Lingkungan Python dan Jupyter digunakan untuk audit data, eksperimen terukur, validasi paritas numerik, dan analisis visual. Aplikasi akhir proyek tetap ditargetkan berupa **C++ desktop GUI** (kerangka GUI belum dipilih; ekstraksi fitur, klasifikasi SVM, evaluasi akhir `test`, dan implementasi C++ merupakan tahap berikutnya).

---

## Ringkasan status & temuan validasi saat ini

Kohort eksperimen menggunakan *frozen split* otoritatif (`data/metadata/split_manifest.csv`, SHA-256 `ceeba8e282f77cdb78a9675618b3dcb6729fbc2339e2d112085a11720fc9108d`):
- **2.228** citra asli (`224×224`), **2.226** pasangan citra–*mask* valid lintas 6 kategori diagnostik (`2` citra `Low-grade IN` tanpa *mask* dikeluarkan dan terbukti merupakan duplikat piksel).
- Pembagian berbasis **353** *canonical uppercase filename groups*: **1.535 `train`** (`247` grup) / **372 `val`** (`53` grup) / **319 `test`** (`53` grup), tanpa tumpang tindih grup maupun sidik jari piksel.
- Referensi biner ditetapkan sebagai `mask >= 128` tanpa memodifikasi *raw masks* (seluruh 35 *full-foreground masks* tetap dipertahankan).

Ringkasan metrik utama validasi (*mean image score* di dalam tiap kategori, lalu *equal-weight mean* lintas enam kategori):

| Komparator | Method ID | Mean Category Dice | Mean Category IoU |
|---|---|---:|---:|
| Raw Otsu (`dark` polarity) | `otsu_M0` | 0.7232 | 0.5788 |
| Processed Otsu | `otsu_M2` | 0.7288 | 0.5886 |
| Raw Lab K-means (`k=4`) | `k4_M0` | 0.7761 | 0.6427 |
| Processed Lab K-means (`k=4`) | `k4_M1` | 0.7760 | 0.6426 |
| All-foreground control | `all_foreground` | 0.7767 | 0.6477 |

- **`k4_M0`** adalah kandidat *image-dependent* terbaik pada checkpoint saat ini, tetapi kontrol konstan **`all_foreground`** masih memiliki rerata sedikit lebih tinggi (`+0.0006` Dice) karena banyak anotasi referensi mencakup area jaringan yang sangat lebar.
- Pada perbandingan per citra (`372` citra `val`, toleransi *tie* `1e-12`), `k4_M0` mengungguli `otsu_M0` pada **276 citra**, sedangkan `otsu_M0` mengungguli `k4_M0` pada **96 citra** (`0` *tie*). Pada pasangan terproses (`otsu_M2` vs `k4_M1`), perbandingannya adalah **122** vs **250** citra (`0` *tie*).
- Prediksi memisahkan kecerahan/warna pewarnaan H&E secara global dan **bukan** segmentasi nukleus, kelenjar, atau area kanker yang tervalidasi secara klinis.

---

## Setup environment (`.venv`)

Gunakan *virtual environment* `.venv` pada repositori ini dan pasang paket `ebhi_dip` secara *editable* beserta dependensi *notebook* dan pengujian:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks,dev]"
```

Di **VS Code**, buka file `.ipynb` pada folder `notebooks/`, pilih kernel **`.venv` (`Python 3.14.0`)**, lalu gunakan **Restart -> Run All**. Daftar versi paket teruji tercatat pada `requirements-lock.txt` (memisahkan *baseline* ilmiah historis dari paket analisis/pengujian tambahan).

---

## Penempatan data dan konfigurasi *data-root*

File dataset mentah dan *cache* piksel berukuran besar sengaja diabaikan oleh Git (`.gitignore`):
- Dataset mentah: `data/raw/EBHI-SEG/<Kategori>/{image,label}/*.png`
- Cache sampel piksel latih (opsional): `data/processed/kmeans_train_pixels.npz`

Apabila menjalankan kode atau *notebook* dari *worktree* terpisah sementara dataset berada di *checkout* utama (`D:\ebhi-dip`), arahkan pembacaan citra mentah tanpa mengubah path relatif di dalam *manifest*:
- Melalui environment variable: `$env:EBHI_DIP_DATA_ROOT = "D:\ebhi-dip"`
- Melalui parameter di sel awal *notebook*: `DATA_ROOT_OVERRIDE = Path(r"D:\ebhi-dip")`
- Melalui argumen CLI pada `scripts/`: `--data-root "D:\ebhi-dip"`

Jika citra mentah tidak tersedia, seluruh analisis berbasis CSV/JSON tetap berjalan dan bagian visualisasi piksel/galeri menampilkan status eksplisit tanpa menghasilkan gambar sintetis.

---

## Urutan dan peta notebook

| Notebook | Fokus pertanyaan & isi utama |
|---|---|
| `notebooks/01_dataset_audit.ipynb` | Kelengkapan 2.226 pasangan citra–*mask*, bukti duplikasi 2 file `missing_mask`, audit nilai abu-abu menengah & 35 *full-foreground masks*, serta *overlay* citra `train`. |
| `notebooks/02_split_audit.ipynb` | Verifikasi *raw-byte* `split_manifest.csv`, normalisasi kapitalisasi 353 `group_id`, bukti nol tumpang tindih antar `train`/`val`/`test`, dan distribusi kategori. |
| `notebooks/03_otsu_baseline.ipynb` | Pemilihan polaritas `dark` pada `train`, rekonsiliasi hasil `val` (`M0`, `M1`, `M2`), contoh visual *threshold* per citra, dan histogram *grayscale*. |
| `notebooks/04_lab_kmeans.ipynb` | Konversi fisik CIE Lab (`float32`), provenance *sampling* `1.535.000` piksel `train`, pusat *cluster* `k=2,3,4` & pemetaan `rate >= 0.5`, serta contoh inferensi penuh. |
| `notebooks/05_segmentation_comparison.ipynb` | Rekonsiliasi 13 metode pada 372 citra `val` (`4.836` baris), perbandingan *equal-category mean* terhadap kontrol `all_foreground`, *heatmap* kategori, dan *coverage*. |
| `notebooks/05b_segmentation_case_analysis.ipynb` | Ranking kasus (`worst`, `best`, `advantage`, `both_poor`, `ties`), *paired scatter*, histogram skor/delta, galeri 2×3 dengan *error overlay* (`TP`/`FP`/`FN`), dan histogram intensitas/Lab 6-panel. |

**Tahap riset berikutnya (belum diimplementasikan dalam migrasi ini):**
`06_feature_extraction.ipynb` -> `07_classification.ipynb` (SVM) -> `08_final_evaluation.ipynb` (*held-out* `test`) -> integrasi `cpp/`.

---

## Kontrol analisis kasus & ekspor figure (`05b`)

Pada sel parameter awal `notebooks/05b_segmentation_case_analysis.ipynb`:
- `METHOD_A` & `METHOD_B`: pasangan metode yang dibandingkan (*default*: `"otsu_M0"` vs `"k4_M0"`).
- `CATEGORY_FILTER`: `None` untuk seluruh kohort `val` atau nama kategori tertentu (mis. `"Polyp"`).
- `RANKING_MODE`: `"default_four"` (maksimal 4 citra unik gabungan `worst_a`, `worst_b`, `a_advantage`, `b_advantage`) atau salah satu dari `worst_a`, `worst_b`, `a_advantage`, `b_advantage`, `both_poor`, `best_a`, `best_b`, `ties`.
- `SAMPLE_ID_OVERRIDE`: isi `sample_id` validasi spesifik untuk memeriksa satu citra tertentu.
- `HISTOGRAM_DENSITY`: `False` (`Jumlah piksel`) atau `True` (`Kerapatan`, integral luas = 1).
- `EXPORT_FIGURES`: ubah ke `True` untuk mengekspor PNG beresolusi tinggi ke `reports/figures/`.
- Bila `ipywidgets` aktif di editor, panel interaktif di bagian akhir *notebook* menyinkronkan seluruh tabel, metrik, galeri, dan histogram.

---

## Struktur direktori

- `src/ebhi_dip/` — Paket inti: `paths.py`, `dataset.py`, `metrics.py`, `provenance.py`, `segmentation.py`, `analysis.py`, `plotting.py`.
- `notebooks/` — Notebook riset terurut (`01`–`05` dan `05b`).
- `scripts/` — *Thin CLI runners* dengan penjaga `--overwrite` (aman di-*import* tanpa menjalankan proses *batch*).
- `configs/` — Konfigurasi eksperimen (`otsu_raw.json`) dan catatan migrasi provenance (`provenance_migration.json`).
- `data/metadata/` — Manifest, audit, grup nama file, serta `split_manifest.csv` + `.sha256` yang dibekukan.
- `models/` — Artefak JSON pusat *cluster* dan pemetaan *foreground* Lab K-means.
- `results/` — Skor historis CSV dan metadata *run* validasi.
- `reports/figures/` — Tujuan ekspor *figure* opsional dari kode visualisasi.
- `docs/` — Dokumentasi garis keturunan provenance (`docs/provenance_migration.md`).
- `tests/` — Pengujian kontrak numerik, integritas kohort, provenance, ranking, dan histogram.
- `cpp/` — Ruang kerja aplikasi desktop C++ tahap selanjutnya.

---

## Verifikasi dan pengujian

Jalankan *unit tests*:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Jalankan ulang keenam *notebook* secara eksplisit dari *fresh kernel* (tanpa *wildcard*):

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_dataset_audit.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/02_split_audit.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/03_otsu_baseline.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/04_lab_kmeans.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/05_segmentation_comparison.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/05b_segmentation_case_analysis.ipynb
```

---

## Batas metodologis dan provenance

1. **Batas pengelompokan nama file:** `group_id` diturunkan dari prefix nama file sebelum `-` (*uppercase*) dan merupakan **kunci pengelompokan sementara**, bukan ID pasien klinis yang terverifikasi.
2. **Integritas frozen split:** `data/metadata/split_manifest.csv` diverifikasi menggunakan *raw bytes* secara ketat; normalisasi *line-ending* dilarang pada file tersebut.
3. **Lineage kode sumber (`utf8-lf-v1`):** perbedaan *hash* `CRLF` vs `LF` pada *helper* historis ditangani secara eksplisit melalui `configs/provenance_migration.json` dan `docs/provenance_migration.md` tanpa menonaktifkan pemeriksaan integritas maupun mengubah artefak historis.
4. **Disiplin subset `test`:** subset `test` (`319` citra) dijaga ketat untuk evaluasi akhir tunggal setelah seluruh keputusan fitur dan klasifikasi dikunci.
