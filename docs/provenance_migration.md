# Provenance Migration & Line-Ending Lineage

Dokumen ini mencatat hubungan eksplisit antara *helper* historis pada checkpoint `8192e3e09384b05fa760104436f3a2e226380ac3`, implementasi terpusat pada `src/ebhi_dip/segmentation.py`, dan artefak eksperimen yang dipertahankan tanpa perubahan *bytes*.

## Latar masalah

Artefak `models/kmeans_lab_k{2,3,4}_mapped.json` dan `results/kmeans_morph_val_run.json` menyimpan `SHA-256` *raw bytes* dari `kmeans_prediction.py` dan `postprocessing.py` saat dijalankan pada *working tree* Windows (`CRLF`). Pada *checkout* dengan *line endings* `LF`, *hash* *raw bytes* berbeda walaupun isi kode identik secara karakter.

Artefak historis **tidak ditulis ulang** dan *guard* integritas **tidak dinonaktifkan**. Sebagai gantinya, proyek menggunakan konvensi *source-text* portabel `utf8-lf-v1` dan catatan migrasi eksplisit `configs/provenance_migration.json`.

## Aturan hashing

| Jenis file | Fungsi | Aturan |
|---|---|---|
| `data/metadata/split_manifest.csv` | `verify_frozen_split` / `sha256_bytes` | **Raw bytes authoritative** (`ceeba8e2...`). Normalisasi teks dilarang keras. |
| Artefak model & JSON historis | `sha256_bytes` | **Raw bytes** tetap cocok dengan metadata *run* historis. |
| Kode sumber Python (`utf8-lf-v1`) | `sha256_source_text` | Decode UTF-8 ketat, hapus UTF-8 BOM awal bila ada, normalisasi `CRLF`/`CR` menjadi `LF`, pertahankan spasi dan *final newline*. |

## Peta lineage helper ke package

| Peran | File historis | SHA-256 historis (`CRLF`) | SHA-256 `utf8-lf-v1` historis | Target migrasi | SHA-256 `utf8-lf-v1` modul baru |
|---|---|---|---|---|---|
| Prediksi *cluster* Lab | `kmeans_prediction.py` | `42a0a3a5...c1513503299` | `d66c08b0...fa87ffaf9b3272c` | `ebhi_dip.segmentation.nearest_clusters` | `128cb8c9...3f75bdb70a510a7` |
| Varian morfologi M0–M2 | `postprocessing.py` | `b83e5494...11797f0125fc6c673f8ca` | `767318f2...7642d3f2233cd4f` | `ebhi_dip.segmentation.morphology_variants` | `128cb8c9...3f75bdb70a510a7` |

## Bukti paritas dan batas keamanan

- `verify_helper_provenance` hanya menerima *digest* historis yang dideklarasikan eksplisit pada `configs/provenance_migration.json`.
- Perubahan kode nyata, perubahan spasi, atau perubahan *final newline* pada `src/ebhi_dip/segmentation.py` langsung menggagalkan pemeriksaan `utf8-lf-v1`.
- Paritas numerik diverifikasi melalui *unit tests* pada *fixture* terkontrol (*exact tie*, area komponen 15 vs 16, *connectivity* 8, perilaku batas `BORDER_REPLICATE`), perbandingan *mask* identik pada citra `train`, serta kecocokan skor `Dice`/`IoU` terhadap CSV validasi pada `atol=1e-10`.
