# Analisis per citra, gallery, dan histogram

## Tujuan dan bahasa

Analisis harus menunjukkan pada citra mana masing-masing metode bekerja lebih baik dan seperti apa kesalahannya. Gunakan saved validation scores untuk memilih kasus, lalu tampilkan prediksi kedua metode pada citra yang sama.

Narasi utama menggunakan bahasa Indonesia yang ringkas; istilah teknis yang alami tetap bahasa Inggris. Pertahankan original IDs, category labels, dan method IDs. Gunakan paragraf pendek, tabel metrics, serta caption yang menjelaskan temuan tanpa mengulang semua angka.

## Cohort dan paired comparison

- Gunakan 372 validation images; tidak melakukan test performance evaluation atau tuning baru.
- Default method A = `otsu_M0`, B = `k4_M0`.
- Secondary pair: A = `otsu_M2`, B = `k4_M1`. Metode lain yang sudah dievaluasi boleh dipilih.
- Join scores pada `sample_id` secara one-to-one; verifikasi `category`, `group_id`, dan cohort sama. Duplicate/missing pairs harus menghasilkan error yang jelas, bukan silently dropped.
- Definisikan `delta_dice = dice_a - dice_b`; positif berarti A lebih tinggi. Tie tolerance `1e-12`, rtol=0.
- Tampilkan counts A lebih tinggi / B lebih tinggi / tie, keseluruhan dan per kategori. Jelaskan bahwa ini counts per citra, berbeda dari equal-category mean.
- Sediakan paired scatter Dice A vs Dice B dengan identity line, sumbu sama pada [0,1].

## Ranking kasus

| Mode | Aturan |
|---|---|
| `worst_a` | Dice A ascending. |
| `worst_b` | Dice B ascending. |
| `a_advantage` | Delta positif di atas tie tolerance, terbesar dahulu. |
| `b_advantage` | Delta negatif di bawah minus tie tolerance, terkecil dahulu. |
| `both_poor` | `max(dice_a, dice_b)` ascending; kedua metode harus rendah untuk muncul di awal. |
| `best_a` / `best_b` | Dice metode terkait descending. |
| `ties` | Absolute delta <= tie tolerance. |

Gunakan `sample_id` ascending sebagai deterministic tie-break. Jika suatu kelompok kosong, nyatakan kosong; jangan memasukkan kasus dengan arah selisih berlawanan. Deduplikasi kasus yang masuk beberapa ranking, tetapi simpan semua alasan pemilihannya.

Ranking table berisi sample ID, kategori, group ID, Dice/IoU kedua metode, delta, dan alasan pemilihan. Bedakan “worst pada metode A” dari “A lebih buruk daripada B”.

Default saved output cukup **maksimal empat citra berbeda**: worst A, worst B, largest A advantage, largest B advantage. Ranking lain tersedia melalui tabel/parameter; hindari notebook berisi ratusan gallery.

## Gallery untuk setiap citra terpilih

Gunakan layout 2×3 dengan enam panel:

1. Original RGB.
2. Binary reference (`mask >= 128`).
3. Prediksi metode A.
4. Prediksi metode B.
5. Error overlay metode A pada original RGB.
6. Error overlay metode B pada original RGB.

Semua panel memakai citra, ukuran, dan orientation yang sama. Tampilkan sample ID, kategori, group, kedua metode, dan alasan pemilihan. Gunakan nearest-neighbor display untuk binary masks dan layout cukup besar agar struktur serta teks terbaca.

Error overlay memakai definisi berikut:

| Komponen | Definisi | Tampilan |
|---|---|---|
| TP | `prediction & reference` | Warna konsisten, misalnya hijau. |
| FP | `prediction & ~reference` | Warna konsisten, misalnya oranye. |
| FN | `~prediction & reference` | Warna konsisten, misalnya magenta. |
| TN | `~prediction & ~reference` | Transparan agar original tetap terlihat. |

Sertakan textual legend TP/FP/FN; jangan mengandalkan warna saja. Sertakan tabel Dice, IoU, pixel precision, pixel recall, dan foreground coverage. Precision/recall dengan denominator nol harus ditandai undefined/NaN, bukan diberi nilai sempurna diam-diam.

## Prediksi dan validasi nilai

- Predictor hanya menerima original RGB dan fixed saved artifacts. Reference digunakan sesudah prediksi untuk evaluasi/overlay.
- Jangan memilih polarity, cluster mapping, atau konfigurasi berdasarkan ground truth citra yang sedang ditampilkan.
- Inference menggunakan seluruh 50.176 piksel per citra. Sampling 1.000 piksel per train image hanya untuk fitting historis.
- Recomputed Dice/IoU pada selected cases harus cocok dengan saved CSV pada `atol=1e-10`, rtol=0.
- Jika mismatch, diagnosis inputs, library behavior, dan provenance. Jangan menimpa historical scores untuk menyembunyikannya.
- Raw grayscale reference tetap tersedia untuk audit encoding; binary reference adalah array terpisah.

## Histogram citra — enam panel per kasus

| Panel | Isi dan aturan |
|---|---|
| RGB | R, G, B dari original uint8; domain 0–255; count per channel. |
| Grayscale | Grayscale yang benar-benar dipakai Otsu; computed Otsu threshold per citra sebagai garis; polarity dark diberi label. |
| Reference encoding | Raw grayscale reference mask; garis threshold 128. Jelaskan bahwa ini bukan Otsu image threshold. |
| L* | Physical float Lab L*, umumnya 0–100; bukan uint8-scaled L. |
| a* | Signed physical a*; pertahankan nilai negatif. |
| b* | Signed physical b*; pertahankan nilai negatif. |

Gunakan exact grayscale conversion dan RGB→Lab yang dipakai pipeline. K-means memakai ketiga Lab coordinates, bukan single grayscale threshold; jangan menggambar satu garis threshold seolah-olah mewakili batas K-means.

Untuk grayscale/RGB/reference, gunakan bins yang mencakup seluruh integer 0–255. Lab menggunakan shared edges saat membandingkan kasus/regions, mencakup semua finite values. Jangan menganggap a*/b* selalu berada dalam batas universal ±128. Tangani constant channel dengan rentang bins yang valid.

Default histogram memakai counts: label y “Jumlah piksel”. Jika menyediakan density mode, label “Kerapatan” dan jelaskan integral histogram bernilai satu; jangan menyebut density sebagai persentase.

Region-conditioned histograms boleh ditambahkan untuk foreground/background menurut reference dan kedua predictions. Gunakan bins sama, tampilkan jumlah piksel setiap region, dan tangani empty region secara eksplisit. Analisis tersebut bersifat post-hoc, bukan bagian inference.

## Histogram performa

- Overlay histogram Dice A dan B dengan bins sama pada [0,1]; y = “Jumlah citra”.
- Histogram delta Dice dengan garis nol dan arah selisih yang jelas.
- Gunakan domain delta yang mencakup nilai positif dan negatif aktual; jangan memotong ekor distribusi.
- Jelaskan jumlah observasi dan filter kategori aktif. Histogram ini berbeda dari pixel-intensity histogram.

## Controls, fallback, dan export

`ipywidgets` opsional: method A/B, category, ranking mode, jumlah kasus, dan sample ID. Semua panel, metrics, dan histogram harus merespons pilihan yang sama. Pertahankan static defaults yang berguna dan parameter-cell fallback agar notebook tetap dapat dibaca tanpa widget renderer.

Ekspor figures ke `reports/figures/` melalui parameter opt-in. Gunakan Matplotlib dari actual arrays; jangan menggunakan image generation, screenshots grafik lama, atau synthetic images sebagai pengganti data penelitian.

Jika raw data tidak tersedia, ranking/tables/score plots berbasis CSV tetap berjalan. Tampilkan pesan jelas pada bagian gallery dan pixel histogram; laporkan bahwa actual-image checks belum dijalankan. Jangan mengisi bagian tersebut dengan fabricated outputs.

## Acceptance checks

1. Fixture ranking: s1 A=.1/B=.8, s2 A=.9/B=.2, s3 A=.4/B=.4. Worst A=s1; worst B=s2; A advantage=s2; B advantage=s1; both poor=s3; tie=s3.
2. Missing/duplicate paired IDs menghasilkan error. Alasan pemilihan tetap lengkap setelah deduplication.
3. `P=[[1,1],[0,0]]`, `R=[[1,0],[1,0]]` menghasilkan TP=FP=FN=TN=1. Precision/recall undefined cases ditangani eksplisit.
4. Counts histogram mencakup seluruh piksel setiap channel; uji signed Lab values, constant channel, dan empty region. Jangan sekadar menguji shape output plot.
5. Raw pair checkpoint: A lebih tinggi 96, B 276, ties 0. Processed pair: A 122, B 250, ties 0. Verifikasi dari CSV, bukan hardcode outputs.
6. Actual-image selected cases memenuhi parity Dice/IoU dengan CSV; overlays dan histogram diperiksa visual pada ukuran normal.
7. Narasi/captions ringkas, istilah teknis alami, label terbaca, dan setiap klaim didukung data. Jangan mengklaim actual-image verification bila raw inputs tidak tersedia.
