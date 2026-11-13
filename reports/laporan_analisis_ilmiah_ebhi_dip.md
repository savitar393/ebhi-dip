# Laporan Analisis Ilmiah Proyek EBHI-DIP: Ekstraksi Fitur 2×3, Klasifikasi SVM Strict (*v3*), dan Evaluasi Sensitivitas Bootstrap

Laporan ini merangkum temuan kanonik proyek **EBHI-DIP** pada *colorectal histopathology* (`EBHI-SEG`, `224×224 px`, 6 kelas: `Adenocarcinoma`, `High-grade IN`, `Low-grade IN`, `Normal`, `Polyp`, `Serrated adenoma`) dengan protokol akhir **`stage07_2x3_strict_fold_aware_val_blind_cv_v3` (`v3`)**. Seluruh analisis dilakukan tanpa menyentuh subset `test` (`0` citra diakses untuk `v2`/`v3`), dan memisahkan secara ketat rekam historis `v1` / Tahap 08 dari evaluasi kanonik `v3`.

---

## 1. Metodologi Ekstraksi Fitur

Dataset (`split_manifest.csv`, SHA-256 `ceeba8e2...`) terbagi secara *group-disjoint* berbasis `group_id` menjadi **1.535 citra `train` (`247` grup)**, **372 citra `val` (`53` grup)**, dan **319 citra `test` (`53` grup terisolasi)**:

- `train` (`1.535 / 247` grup): `Adenocarcinoma` (`540 / 75`), `High-grade IN` (`131 / 28`), `Low-grade IN` (`444 / 101`), `Normal` (`54 / 11`), `Polyp` (`324 / 81`), `Serrated adenoma` (`42 / 10`).
- `val` (`372 / 53` grup): `Adenocarcinoma` (`131 / 15`), `High-grade IN` (`20 / 5`), `Low-grade IN` (`107 / 21`), `Normal` (`10 / 3`), `Polyp` (`96 / 21`), `Serrated adenoma` (`8 / 2`).

Berdasarkan [`results/features/feature_manifest_2x3.json`](../results/features/feature_manifest_2x3.json), setiap citra pada desain faktorial **2×3** (`{otsu, k4} × {M0, M1, M2}`) diekstrak menjadi tiga himpunan fitur kanonik:

1. **Set A (`appearance.csv`, 10 fitur bersama lintas metode):** independen terhadap mask segmentasi. Terdiri atas **6 statistik CIE Lab** fisik (`float32 / 255 -> cv2.COLOR_RGB2LAB`; rerata dan simpangan baku populasi pada kanal `L*`, `a*`, `b*` seluruh piksel) serta **4 deskriptor tekstur GLCM** (*contrast*, *homogeneity*, *energy*, *correlation*) dari citra grayscale terkuantisasi tetap `gray // 16` (`16` level), jarak `{1, 2}`, dan sudut `{0°, 45°, 90°, 135°}` (simetris, ternormalisasi, dirata-ratakan atas 8 pasangan). Pada `train+val` (`1.907` citra), daya pisah univariat Kruskal-Wallis $\eta^2$ tertinggi pada Set A dimiliki oleh `lab_b_std` (`0.1379`), `lab_a_mean` (`0.1117`), `lab_b_mean` (`0.1101`), `lab_l_mean` (`0.1060`), dan `glcm_homogeneity` (`0.0535`).
2. **Set B (19 fitur morfologi per konfigurasi `otsu_M0` s.d. `k4_M2`):** terdiri atas **4 deskriptor mask global** (`mask_foreground_fraction`, `mask_component_count`, `mask_edge_foreground_fraction`, `mask_is_empty`) dan **15 statistik komponen 8-connected** (*mean*, *population std*, *median* dari luas relatif terhadap luas citra `region_area_rel`, perimeter 4-connected *Benkrid–Crookes* relatif terhadap $\sqrt{HW}$ `region_perimeter_rel`, `region_circularity` = $4\pi A / P^2$ dengan kebijakan `0` bila $P=0$, `region_eccentricity`, dan `region_solidity` menggunakan luas convex hull piksel eksak). Tidak ditemukan satu pun mask kosong pada keenam konfigurasi (`empty_masks_by_method = 0`).
3. **Set C (29 fitur gabungan):** konkatenasi deterministik Set A + Set B. Selain itu, dipertahankan sensitivitas sekunder **`A_seg` (10 fitur)** dan **`C_seg` (29 fitur)** pada `k4_M0` yang menghitung `Set A` hanya di atas piksel foreground prediksi.

Uji paritas menunjukkan selisih maksimum `<= 1.72e-15` terhadap tabel fitur historis yang dipertahankan. Distribusi fitur dan struktur korelasi divisualisasikan pada [`06_feature_sensitivity_2x3.png`](figures/06_feature_sensitivity_2x3.png), [`06_morphology_feature_histograms_2x3.png`](figures/06_morphology_feature_histograms_2x3.png), [`06_feature_distributions_by_category.png`](figures/06_feature_distributions_by_category.png), dan [`06_feature_correlation_heatmap.png`](figures/06_feature_correlation_heatmap.png).

---

## 2. Protokol Klasifikasi SVM dan Audit Kebocoran Data (*v1*, *v2*, dan *v3*)

Untuk menjaga akuntabilitas historis, repositori memisahkan keempat generasi evaluasi berikut:

| Generasi / Artefak | Cakupan Eksperimen | Aturan Seleksi Hyperparameter | Fitting Upstream & Scaler dalam CV | Status Subset `test` (`319` citra) |
|---|---|---|---|---|
| **Historis `v1`** (`results/classification/`, `models/svm_*.json`) | `k4_M0` (`A`, `B`, `C`, `A_seg`, `C_seg`) + 4 komparator | Memakai metrik `val` sebagai *tie-breaker* utama (`val_macro_f1` 4 desimal) | Upstream `k4` fit penuh `train`; scaler internal CV | Hanya diuji oleh Tahap 08 |
| **Tahap 08 (`test` Historis `v1`)** (`notebooks/08_final_test_evaluation.ipynb`, `results/test/`) | **Khusus model `v1`** (`12` model + `13` segmentasi) | Membekukan artefak `v1` lalu mengevaluasi 1 kali pada `test` | Mengacu `models/svm_*.json` (`v1`) | Dievaluasi **1×** untuk `v1` saja; **dikeluarkan dari `Run All`** |
| **Ekspansi `v2`** (`results/classification_2x3/`, `models/cv_2x3/`) | `2×3` lengkap + `Shared A` + `A_seg`/`C_seg` (`15` model × `32` kandidat) | **Buta `val` (`cv_selection_key_for_svm`)**: ketat 5-fold `StratifiedGroupKFold` | `StandardScaler` per fold, tetapi mask `k4`/`otsu` masih fit penuh `train` | **`0` citra diakses** |
| **Kanonik `v3` (`2x3-strict`)** (`results/classification_2x3_strict/`, `models/cv_2x3_strict/`) | Grid sama (`15 × 32 = 480` baris) + sensitivitas bootstrap conditional | **CV-only ketat** (`select_top_cv_model_row`, nol pembacaan `val` saat seleksi) | **Refit penuh upstream + scaler di tiap fold `train`** | **`0` citra diakses** |

Audit kebocoran hulu (`fold_upstream_audit.json`) membuktikan perbedaan mekanisme dua keluarga segmentasi saat *refit* dilakukan secara ketat di dalam setiap fold (`1.224–1.232` citra `train` fold vs `303–311` citra `val` fold, 100% terpisah menurut `group_id` dan `sample_id`):

- **Invariansi Otsu per fold:** pencarian polaritas berbasis `mean_category_dice` memilih `"dark"` secara seragam pada **5/5 fold** dengan margin konsisten (`0.2841`, `0.2952`, `0.2937`, `0.2964`, `0.2882`), menghasilkan perbedaan absolut fitur maksimum `<= 2.22e-16` dibanding tabel kanonik. Oleh sebab itu seluruh **6 model `otsu`** dan `Shared Set A` **identik** antara `v2` dan `v3`.
- **Sensitivitas K-Means `k=4` + pemetaan cluster terawasi per fold:** penyamplingan ulang + re-clustering + aturan `rate >= 0.5` menghasilkan pergeseran pusat cluster (`mean center L2 = 1.4842–8.4629`), rotasi indeks cluster FG terurut (`[0,2,3]`, `[0,1,3]`, `[1,2,3]`), serta kesepakatan piksel fold-val `0.9701–0.9958` (`IoU = 0.9601–0.9949`). Menghapus kebocoran kecil ini mengubah *hyperparameter* terpilih pada **6 dari 7 model `k4`** dan mengubah skor CV pada **7 dari 7 model `k4`** (mis. `k4_M0_B` CV turun `0.2937 -> 0.2744`; `k4_M1_C` `linear` berubah menjadi `rbf balanced C=10, g=0.01`; `k4_M0_C_seg` Val turun `0.2942 -> 0.2572`), seperti tampak pada [`07_2x3_strict_vs_fixed_upstream.png`](figures/07_2x3_strict_vs_fixed_upstream.png).
- **Koreksi pemilihan laporan akhir (`v3`):** fungsi [`select_top_cv_model_row`](../src/ebhi_dip/classification.py) kini menggantikan pengurutan berbasis `val_macro_f1` pada skrip evaluasi, memastikan keputusan penobatan model teratas (`svm_otsu_M0_C`, CV `0.4642`) 100% bebas bias `val`.

---

## 3. Kualitas Segmentasi Otsu vs K-Means dan Dampak Morfologi (*M0*, *M1*, *M2*)

Evaluasi segmentasi pada **372 citra `val`** (*equal-category mean* 6 kelas terhadap kontrol konstan `all_foreground` = Dice `0.7767`, IoU `0.6477`) memperlihatkan kesenjangan tajam antara tumpang-tindih piksel dengan representasi bentuk objek biologis:

| Konfigurasi | Val Cat Dice ($\pm$ vs Kontrol) | Val Cat IoU | Rerata FG `val` | Rerata Komponen (`train+val`) | Komponen Perimeter 0 (`M0`) | Fitur Set B Terkuat ($\eta^2$ `train+val`) |
|---|---:|---:|---:|---:|---:|---|
| `otsu_M0` | `0.7232` (`-0.0534`) | `0.5788` | `53.05%` | `201.07` | `143.526` (`32.13%`) | `mask_foreground_fraction` (`0.2790`) |
| `otsu_M1` | `0.7248` (`-0.0519`) | `0.5811` | `51.65%` | `62.49` | `0` (`0.00%`) | `mask_foreground_fraction` (`0.2747`) |
| `otsu_M2` | `0.7288` (`-0.0479`) | `0.5886` | `51.09%` | `69.53` | `0` (`0.00%`) | `mask_foreground_fraction` (`0.2832`) |
| `k4_M0` | `0.7761` (`-0.0006`) | `0.6427` | `85.08%` | `26.40` | `28.433` (`49.50%`) | `region_area_rel_median` (`0.1275`) |
| `k4_M1` | `0.7760` (`-0.0007`) | `0.6426` | `84.94%` | `5.22` | `0` (`0.00%`) | `region_perimeter_rel_mean` (`0.0272`) |
| `k4_M2` | `0.7753` (`-0.0013`) | `0.6425` | `86.86%` | `6.77` | `0` (`0.00%`) | `mask_foreground_fraction` (`0.0287`) |

- **Mengapa `k4` mencatat Dice lebih tinggi:** ground truth EBHI-SEG memberi anotasi padat pada area lesi/jaringan target yang rata-rata mencakup **`54.64%` (`Polyp`) hingga `79.04%` (`Serrated adenoma`)** piksel citra. Karena `k4` mengklasifikasikan `3 dari 4 cluster` Lab sebagai foreground (`~85.1%–86.9%` luas citra), `k4` perilakunya mendekati mask konstan `all_foreground` (`0.7767`)—menekan *False Negative* (terutama pada citra stroma padat ungu muda atau penuh jaringan) sambil menghasilkan *False Positive* besar pada lumen pucat dan stroma penyangga ([`segmentation_validation_dice.png`](figures/segmentation_validation_dice.png), [`segmentation_validation_category_heatmap.png`](figures/segmentation_validation_category_heatmap.png)).
- **Dampak morfologi `M0 -> M1 -> M2`:** pembersihan komponen `< 16 px` (`M1`) menghilangkan 100% piksel terisolasi berperimeter nol, menurunkan jumlah komponen Otsu dari `201.07 -> 62.49` dan sedikit menaikkan Dice (`+0.0016` ke `M1`, `+0.0056` ke `M2`). Namun *3×3 cross opening+closing* (`M2`) mengikis batas kelenjar beresolusi halus (`region_solidity_mean` turun di `M1` `0.8935 -> 0.7439` lalu pulih `0.8678`, dan memicu artefak `circularity_median = 1.0830` pada komponen kecil diskret). Pada `k4`, pereduksian `M1`/`M2` nyaris tidak mengubah Dice (`0.7761 -> 0.7753`), tetapi melenyapkan variasi komponen mikro (`26.40 -> 5.22` komponen).
- **Bukti visual kasus citra nyata (`val`) & histogram:** perbedaan ini tampak pada *paired scatter* ([`05b_paired_scatter_otsu_M0_vs_k4_M0.png`](figures/05b_paired_scatter_otsu_M0_vs_k4_M0.png)), kasus terbaik ([`05b_best_otsu_gallery.png`](figures/05b_best_otsu_gallery.png), [`05b_best_k4_gallery.png`](figures/05b_best_k4_gallery.png)), kasus ekstrem/gagal ([`05b_case_1_Adenocarcinoma__GT2100009-1-400-003_gallery.png`](figures/05b_case_1_Adenocarcinoma__GT2100009-1-400-003_gallery.png), [`05b_both_poor_distinct_gallery.png`](figures/05b_both_poor_distinct_gallery.png)), distribusi FG vs BG ([`05b_fg_vs_bg_otsu_advantage.png`](figures/05b_fg_vs_bg_otsu_advantage.png), [`05b_fg_vs_bg_k4_advantage.png`](figures/05b_fg_vs_bg_k4_advantage.png)), serta progresi morfologi berlawanan antara citra dengan celah stroma terang vs jaringan epitel rapat ([`05b_morphology_M0_M1_M2_otsu_gain.png`](figures/05b_morphology_M0_M1_M2_otsu_gain.png), [`05b_morphology_M0_M1_M2_tradeoff.png`](figures/05b_morphology_M0_M1_M2_tradeoff.png)).

---

## 4. Hasil Klasifikasi Enam Kelas: Perbandingan Set A, Set B, dan Set C (*v3*)

Tabel berikut merangkum seluruh **15 model terseleksi CV** pada protokol kanonik `v3` (`svm_2x3_strict_selected_summary.csv` & `svm_2x3_strict_bootstrap_comparison.csv`):

| Model (`v3` `cv_selected`) | Set Fitur | Hyperparameter (`kernel, wt, C, gamma`) | Strict CV Macro-F1 ($\pm$ SD) | Val Macro-F1 | 95% CI Unconditional | 95% CI Conditional All-6 | Val BalAcc | Val Acc |
|---|---|---|---:|---:|---|---|---:|---:|
| **`svm_shared_A`** | `A` (10) | `rbf, none, C=10, scale` | `0.3346 (±0.0371)` | `0.3140` | `[0.2304, 0.4166]` | `[0.2260, 0.3738]` | `0.3207` | `49.73%` |
| `svm_otsu_M0_B` | `B` (19) | `rbf, bal, C=10, 0.01` | `0.4126 (±0.0237)` | `0.3280` | `[0.2712, 0.4113]` | `[0.2666, 0.3840]` | `0.3781` | `43.55%` |
| **`svm_otsu_M0_C`** *(Top CV & Val)* | **`C` (29)** | **`linear, none, C=3, scale`** | **`0.4642 (±0.0570)`** | **`0.4058`** | **`[0.3316, 0.5025]`** | **`[0.3272, 0.4726]`** | `0.4016` | **`56.72%`** |
| `svm_otsu_M1_B` | `B` (19) | `rbf, bal, C=10, 0.01` | `0.3789 (±0.0509)` | `0.3033` | `[0.2398, 0.3847]` | `[0.2388, 0.3583]` | `0.3576` | `38.17%` |
| `svm_otsu_M1_C` | `C` (29) | `rbf, bal, C=3, 0.01` | `0.4481 (±0.0125)` | `0.3470` | `[0.2669, 0.4439]` | `[0.2641, 0.4156]` | `0.4171` | `45.16%` |
| `svm_otsu_M2_B` | `B` (19) | `rbf, bal, C=10, 0.01` | `0.4077 (±0.0216)` | `0.3055` | `[0.2366, 0.3929]` | `[0.2318, 0.3642]` | `0.3393` | `41.13%` |
| `svm_otsu_M2_C` | `C` (29) | `rbf, bal, C=3, 0.01` | `0.4622 (±0.0242)` | `0.3462` | `[0.2666, 0.4345]` | `[0.2619, 0.4197]` | `0.4240` | `44.62%` |
| `svm_k4_M0_B` | `B` (19) | `rbf, bal, C=10, scale` | `0.2744 (±0.0330)` | `0.2490` | `[0.1950, 0.3187]` | `[0.1925, 0.2874]` | `0.2790` | `33.33%` |
| `svm_k4_M0_C` | `C` (29) | `rbf, bal, C=3, scale` | `0.3707 (±0.0395)` | `0.3386` | `[0.2624, 0.4277]` | `[0.2582, 0.4025]` | `0.4018` | `44.62%` |
| `svm_k4_M1_B` | `B` (19) | `rbf, bal, C=10, 0.1` | `0.2797 (±0.0180)` | `0.2311` | `[0.1810, 0.3050]` | `[0.1790, 0.2676]` | `0.2829` | `32.80%` |
| `svm_k4_M1_C` | `C` (29) | `rbf, bal, C=10, 0.01` | `0.3643 (±0.0190)` | `0.2951` | `[0.2174, 0.3813]` | `[0.2135, 0.3592]` | `0.3489` | `38.98%` |
| `svm_k4_M2_B` | `B` (19) | `rbf, bal, C=3, 0.1` | `0.2633 (±0.0179)` | `0.2072` | `[0.1654, 0.2616]` | `[0.1638, 0.2391]` | `0.2173` | `29.57%` |
| `svm_k4_M2_C` | `C` (29) | `rbf, bal, C=3, scale` | `0.3766 (±0.0322)` | `0.3055` | `[0.2410, 0.3829]` | `[0.2359, 0.3647]` | `0.3587` | `41.13%` |
| `svm_k4_M0_A_seg` *(Sens.)* | `A_seg` (10) | `linear, bal, C=1, scale` | `0.3709 (±0.0171)` | `0.2575` | `[0.1851, 0.3447]` | `[0.1830, 0.3310]` | `0.3177` | `30.91%` |
| `svm_k4_M0_C_seg` *(Sens.)* | `C_seg` (29) | `rbf, bal, C=10, 0.01` | `0.3897 (±0.0324)` | `0.2572` | `[0.2002, 0.3340]` | `[0.1980, 0.2980]` | `0.2593` | `35.75%` |

Temuan inti lintas Set A, Set B, dan Set C ([`07_2x3_classification_overview.png`](figures/07_2x3_classification_overview.png), [`07_2x3_confusion_matrices_shared_A_best_B_best_C.png`](figures/07_2x3_confusion_matrices_shared_A_best_B_best_C.png)):

- **Set C mendominasi Set A maupun Set B pada `otsu_M0`:** kombinasi fitur `otsu_M0_C` (`0.4058`) mengungguli `Shared Set A` (`0.3140`) sebesar **`+0.0919 [+0.0190, +0.1788]`** (`prop>0 = 0.9890`) dan mengungguli `otsu_M0_B` (`0.3280`) sebesar **`+0.0778 [+0.0060, +0.1483]`** (`prop>0 = 0.9840`). Hal ini membuktikan sifat saling melengkapi: fitur warna/GLCM (*appearance* global) berpasangan dengan struktur komponen epitel Otsu, dan menghasilkan pemisah linear berdimensi bersih (`linear, C=3, none`) yang mempertahankan akurasi tinggi (`56.72%`).
- **Morfologi murni (`Set B`) Otsu jauh melampaui K-Means `k4`:** `otsu_M0_B` (`0.3280`) mengungguli `k4_M0_B` (`0.2490`) sebesar **`+0.0790 [+0.0316, +0.1417]`** (`prop>0 = 0.9995`), bahkan melampaui `Shared Set A` dalam CV (`0.4126` vs `0.3346`). Sebaliknya, seluruh konfigurasi `Set B` pada `k4` berada di bawah `Shared Set A` (`0.2072–0.2490`).
- **Sensitivitas `A_seg` / `C_seg` memperburuk generalisasi `k4_M0`:** menyempitkan fitur *appearance* ke mask `k4_M0` menghilangkan sinyal latar/lumen pembeda, menurunkan Val Macro-F1 dari `0.3386` (`k4_M0_C`) menjadi `0.2572` (`k4_M0_C_seg`; delta **`-0.0815 [-0.1296, -0.0300]`**, `prop>0 = 0.0010`), dan dari `0.3140` (`Shared A`) menjadi `0.2575` (`k4_M0_A_seg`; delta `-0.0564 [-0.1304, +0.0337]`). Trade-off antara `class_weight="balanced"` vs `"none"` terhadap akurasi keseluruhan terlihat pada [`07_2x3_cw_tradeoff.png`](figures/07_2x3_cw_tradeoff.png).

---

## 5. Mengapa Dice Segmentasi Tinggi Tidak Menjamin *Macro-F1* Klasifikasi Lebih Baik

Lintas $n=6$ konfigurasi (`otsu_M0..k4_M2`), asosiasi deskriptif antara Dice segmentasi dan skor klasifikasi hilir bersifat **negatif kuat** (`classification_2x3_strict_manifest.json`; [`07_2x3_seg_dice_vs_classification.png`](figures/07_2x3_seg_dice_vs_classification.png)):

- **Set B (19 fitur morfologi):** Dice `val` vs Strict CV Macro-F1 mencatat Pearson deskriptif `r = -0.9799` (Spearman `rho = -0.7714`); vs Val Macro-F1 `r = -0.9478` (`rho = -0.7143`); vs Val BalAcc `r = -0.9066` (`rho = -0.8286`); vs Val Accuracy `r = -0.9188` (`rho = -0.7143`).
- **Set C (29 fitur gabungan):** Dice `val` vs Strict CV Macro-F1 mencatat Pearson deskriptif `r = -0.9876` (Spearman `rho = -0.8857`); vs Val Macro-F1 `r = -0.7706` (`rho = -0.8286`).

Tiga mekanisme geometris menjelaskan paradoks ini ([`07_2x3_fg_components_vs_macro_f1.png`](figures/07_2x3_fg_components_vs_macro_f1.png), [`06_morphology_feature_histograms_2x3.png`](figures/06_morphology_feature_histograms_2x3.png)):

1. ***Over-coverage* pada `k4` melebur objek diagnostik:** dengan cakupan `85.27%–86.92%` (`train+val`), `99.18%–99.41%` piksel foreground `k4` tergabung ke dalam segelintir komponen yang menyentuh tepi citra (`5.22–26.40` komponen/citra). Peleburan kelenjar/nukleus dan stroma menjadi satu massa kontinu mematikan variasi morfologi antar-kelas histopatologi.
2. **Penghilangan noise `M1`/`M2` menghancurkan sisa sinyal `k4`:** pada `k4_M0`, fitur `Set B` paling informatif justru berasal dari median luas & perimeter *speck* kecil (`region_area_rel_median` $\eta^2 = 0.1275$, `28.433` komponen nol-perimeter). Begitu `M1` menghapus komponen `< 16 px` (`k4_M1`, `5.22` komponen), daya pisah **seluruh 19 fitur morfologi `k4` runtuh ke $\eta^2 \le 0.0287$**, menurunkan Val Macro-F1 `Set B` dari `0.2490` (`k4_M0_B`) ke `0.2311` (`k4_M1_B`) dan `0.2072` (`k4_M2_B`).
3. **Otsu menjaga struktur epitel/inti & fragmen diagnostik:** karena Otsu (`dark`) hanya menandai `~52.7%–54.3%` piksel tergelap (inti & kompleks epitel), fitur seperti `mask_foreground_fraction` (`eta^2 = 0.2790`) dan `region_area_rel_std` (`0.2567`) membedakan pola pertumbuhan padat (`Adenocarcinoma`) dari kelenjar polip teratur dengan sangat baik—lebih dari dua kali lipat $\eta^2$ tertinggi pada fitur Set A (`0.1379`). Di sisi lain, menghaluskan Otsu ke `M1`/`M2` mengurangi keragaman komponen alami (`201.07 -> 62.49–69.53`), sehingga **`otsu_M0_C` (`0.4058`) mengungguli `otsu_M1_C` (`0.3470`) dan `otsu_M2_C` (`0.3462`)** meski Dice segmentasinya `0.0056` lebih rendah.

---

## 6. Analisis Kesenjangan CV vs Validasi dan Performa per Kategori

Di seluruh model `v3`, terdapat kesenjangan moderat antara **5-fold Strict CV** pada `train` (`1.535` citra / `247` grup) dan **Validasi tetap** (`372` citra / `53` grup): mis. `otsu_M0_C` `0.4642 -> 0.4058` (`-0.0584`), `otsu_M0_B` `0.4126 -> 0.3280` (`-0.0846`), `otsu_M2_C` `0.4622 -> 0.3462` (`-0.1160`), dan `k4_M0_C` `0.3707 -> 0.3386` (`-0.0321`). Rincian *per-category F1* pada validasi (`svm_2x3_strict_per_category_val.csv`; [`07_2x3_per_category_f1.png`](figures/07_2x3_per_category_f1.png), [`07_2x3_confusion_matrices_set_C.png`](figures/07_2x3_confusion_matrices_set_C.png)) menjelaskan sumber kesenjangan ini:

| Model (`v3` Val F1) | `Adenocarcinoma` (`131` / `15` grp) | `High-grade IN` (`20` / `5` grp) | `Low-grade IN` (`107` / `21` grp) | `Normal` (`10` / `3` grp) | `Polyp` (`96` / `21` grp) | `Serrated adenoma` (`8` / `2` grp) | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `svm_shared_A` | `0.4649` | `0.0667` | `0.4964` | `0.1739` | `0.6818` | `0.0000` | `0.3140` |
| `svm_otsu_M0_B` | `0.3465` | `0.2647` | `0.5429` | `0.1111` | `0.6404` | `0.0625` | `0.3280` |
| **`svm_otsu_M0_C`** | **`0.5512`** | **`0.2759`** | `0.5593` | `0.0952` | **`0.7312`** | **`0.2222`** | **`0.4058`** |
| `svm_otsu_M2_C` | `0.4098` | `0.1639` | `0.5320` | `0.1887` | `0.6628` | `0.1200` | `0.3462` |
| `svm_k4_M0_C` | `0.3768` | `0.1923` | `0.4978` | **`0.2143`** | `0.6901` | `0.0606` | `0.3386` |
| `svm_k4_M0_C_seg` | `0.2564` | `0.0984` | `0.5094` | `0.0702` | `0.6087` | `0.0000` | `0.2572` |

- **Keunggulan kelas dengan dukungan memadai:** penggabungan `otsu_M0_C` mendorong F1 `Polyp` hingga `0.7312`, `Low-grade IN` hingga `0.5593`, serta mengangkat `Adenocarcinoma` dari `0.4649` (`Shared A`) / `0.3768` (`k4_M0_C`) menjadi `0.5512` (mengurangi kebingungan `Adenocarcinoma` $\rightarrow$ `Low-grade IN` menjadi `42/131` = `32%` serta mempertahankan `recall = 53%` dengan presisi tertinggi).
- **Bottleneck kelas minoritas:** Macro-F1 memberi bobot `1/6` setara pada `Serrated adenoma` (`8` citra, **hanya `2` grup `val`**) dan `Normal` (`10` citra, `3` grup `val`), serta `High-grade IN` (`20` citra, `5` grup). Keberhasilan atau kegagalan prediksi pada **1–2 spesimen (`group_id`)** saja mengubah F1 kelas tersebut secara ekstrem (`Serrated adenoma` berkisar `0.0000` s.d. `0.2222`), dan secara langsung menurunkan metrik rata-rata 6 kelas pada validasi dibanding CV.

---

## 7. Sensitivitas Bootstrap (*Unconditional* vs *Conditional All-6-Class*)

Pada bootstrap blok grup kanonik (`53` grup `val`, `B = 2.000`, `seed = 42`), **82.3%** penarikan ulang memuat keenam kelas secara bersamaan, sementara **~17.7%** kehilangan `Serrated adenoma` (2 grup) dan/atau `Normal` (3 grup). Dalam skema *unconditional*, Macro-F1 dihitung atas kategori dengan dukungan `> 0` di tiap replikasi, yang berpotensi menciptakan bias ekor atas ketika kelas minoritas ber-F1 rendah absen.

Analisis sensitivitas tambahan ([`bootstrap_sensitivity_summary.json`](../results/classification_2x3_strict/bootstrap_sensitivity_summary.json), [`svm_2x3_strict_bootstrap_comparison.csv`](../results/classification_2x3_strict/svm_2x3_strict_bootstrap_comparison.csv)) melakukan *group-blocked paired bootstrap* yang menerima **2.000 replikasi valid** hanya jika keenam kelas hadir (**total `2.432` penarikan; `432` ditolak; *acceptance rate* = `82.24%`**; [`07_2x3_bootstrap_uncond_vs_cond_all6.png`](figures/07_2x3_bootstrap_uncond_vs_cond_all6.png)):

- **Efek pada CI model tunggal:** mewajibkan keenam kelas hadir memangkas batas atas *95% CI* Macro-F1 di **15/15 model** (`rasio lebar cond/uncond` antara `0.7144` dan `0.9398`, median `0.8465`): contohnya `svm_otsu_M0_C` menyempit dari `[0.3316, 0.5025]` menjadi `[0.3272, 0.4726]` (`ratio = 0.8505`), `svm_shared_A` dari `[0.2304, 0.4166]` menjadi `[0.2260, 0.3738]` (`0.7937`), dan `svm_k4_M0_B` dari `[0.1950, 0.3187]` menjadi `[0.1925, 0.2874]` (`0.7672`).
- **Konsistensi kesimpulan kontras berpasangan:** rasio lebar CI kontras berkisar `0.9059–1.0319` (median `0.9488`), dan **seluruh kesimpulan substantif tetap konsisten** (**14/15** kontras Macro-F1 dan **15/15** kontras Balanced Accuracy / Accuracy mempertahankan status eksklusi nol):
  - `otsu_M0_C vs shared_A` (`+0.0919`): Uncond `[+0.0190, +0.1788] (0.9890)` $\rightarrow$ Cond `[+0.0238, +0.1748] (0.9965)` *(tetap di atas nol)*.
  - `otsu_M0_B vs k4_M0_B` (`+0.0790`): Uncond `[+0.0316, +0.1417] (0.9995)` $\rightarrow$ Cond `[+0.0340, +0.1385] (1.0000)` *(tetap di atas nol)*.
  - `otsu_M0_C vs otsu_M0_B` (`+0.0778`): Uncond `[+0.0060, +0.1483] (0.9840)` $\rightarrow$ Cond `[+0.0038, +0.1405] (0.9805)` *(tetap di atas nol)*.
  - `k4_M0_C vs k4_M1_C` (`+0.0436`): Uncond `[+0.0017, +0.0868] (0.9785)` $\rightarrow$ Cond `[+0.0022, +0.0840] (0.9805)` *(tetap di atas nol)*.
  - `k4_M0_C_seg vs k4_M0_C` (`-0.0815`): Uncond `[-0.1296, -0.0300] (0.0010)` $\rightarrow$ Cond `[-0.1252, -0.0322] (0.0010)` *(tetap negatif di bawah nol)*.
  - `otsu_M0_C vs otsu_M1_C` (`+0.0588`): satu-satunya kontras batas yang bergeser tipis melintasi nol sebesar `0.0018` (`[-0.0015, +0.1231] (0.9725)` $\rightarrow$ `[+0.0003, +0.1214] (0.9750)`), sementara `otsu_M0_C vs k4_M0_C` (`[-0.0052, +0.1494]`, `prop>0 = 0.9665`, serta `Val Acc +12.10 pp [+3.88, +20.73 pp]`) mempertahankan pola arah yang sama.
- **Catatan interpretasi:** angka `prop_bootstrap_delta_gt_0` (`prop>0`) hanya merepresentasikan proporsi empiris dari replikasi bootstrap yang bertanda positif—**bukan probabilitas posterior Bayesian** dan **bukan uji hipotesis konfirmatori** ([`07_2x3_strict_paired_bootstrap_contrasts.png`](figures/07_2x3_strict_paired_bootstrap_contrasts.png)).

---

## 8. Keterbatasan Penelitian dan Posisi Rekam Historis Tahap 08

1. **Dukungan grup minoritas sangat kecil di `val`:** hanya `2` grup (`8` citra) `Serrated adenoma`, `3` grup (`10` citra) `Normal`, dan `5` grup (`20` citra) `High-grade IN`, menyebabkan ketidakpastian estimasi per-kelas tinggi dan penolakan `17.76%` undian pada bootstrap conditional.
2. **Pengelompokan `group_id` sementara:** diturunkan dari prefix nama file sebelum `-` (*uppercase*), bukan identifikasi pasien klinis terverifikasi.
3. **Semantik kompartemen warna vs histologi klinis:** baik Otsu maupun Lab K-Means `k=4` memetakan kompartemen intensitas/warna pewarnaan H&E secara global, bukan segmentasi nukleus, lumen, atau batas tumor klinis.
4. **Cakupan korelasi $n=6$ & model linier/RBF tabular:** korelasi Dice vs Macro-F1 lintas 6 konfigurasi hanya bersifat deskriptif mekanistik pada citra validasi bersama, dan ruang fitur terbatas pada 29 statistik global/komponen.
5. **Posisi terpisah Rekam Historis Tahap 08 (`v1`):** evaluasi *single-shot held-out test* pada [`notebooks/08_final_test_evaluation.ipynb`](../notebooks/08_final_test_evaluation.ipynb) (`results/test/`) merupakan arsip khusus **`v1`** (di mana pada protokol historis `v1`, `svm_k4_M0_selected` memperoleh Test Macro-F1 `0.3507 [0.2545, 0.4171]`, Test BalAcc `0.3313`, Test Acc `48.28%`, sedangkan komparator `v1` `svm_otsu_M2_C` mencatat `0.4117 [0.3435, 0.4616]`, Acc `59.25%`). Subset `test` **`319` citra sama sekali tidak disentuh** (`0` citra/label diakses) dalam seluruh protokol **`v2` maupun `v3` (`2x3-strict`)**, dan Notebook 08 dikecualikan dari verifikasi rutin `01–07`.

---

## 9. Kesimpulan

1. **Model terbaik & validitas prosedural `v3`:** di bawah protokol ketat bebas-kebocoran **`2x3-strict` (`v3`)** dengan seleksi CV-only, **`svm_otsu_M0_C`** (`linear, none, C=3`, 29 fitur Set C) menjadi model teratas dengan Strict 5-fold CV Macro-F1 **`0.4642`** dan Validation Macro-F1 **`0.4058`** (Accuracy **`56.72%`**, BalAcc `0.4016`), unggul konsisten atas `Shared Set A` (`+0.0919`, 95% CI uncond `[+0.0190, +0.1788]`, cond `[+0.0238, +0.1748]`) dan `otsu_M0_B` (`+0.0778`, uncond `[+0.0060, +0.1483]`, cond `[+0.0038, +0.1405]`).
2. **Komplementaritas Set A dan Set B:** menggabungkan deskriptor warna/GLCM (`Set A`, $\eta^2$ puncak `0.1379`) bersama morfologi komponen epitel Otsu (`Set B`, $\eta^2$ puncak `0.2790`) menghasilkan peningkatan berpasangan yang nyata dibanding masing-masing himpunan tunggal, sedangkan pembatasan region ke mask `k4` (`C_seg`) menghilangkan konteks lumen/stroma (`-0.0815`).
3. **Paradoks Dice segmentasi vs klasifikasi:** pada EBHI-SEG, K-Means `k=4` meraih Dice validasi lebih tinggi (`0.7753–0.7761` vs `0.7232–0.7288`) terutama karena *over-coverage* `85%–87%` yang meniru kontrol `all_foreground` (`0.7767`), namun melebur komponen kelenjar menjadi sedikit blok besar sehingga daya pisah fitur morfologi anjlok ($\eta^2 \le 0.0287$ di `k4_M1`/`k4_M2`) dan berkorelasi negatif deskriptif terhadap Macro-F1 klasifikasi (`r = -0.9478` pada Set B, `-0.7706` pada Set C).
4. **Ketahanan statistik terhadap uji sensitivitas:** audit per-fold mengungkap invariansi total Otsu (`5/5` `"dark"`) serta sensitivitas ringan `k4` (`6/7` model berubah *hyperparameter* di `v3`), sementara uji sensitivitas *conditional all-6-class bootstrap* (`acceptance rate 82.24%`) memperketat ekor atas CI tanpa mengubah kesimpulan substantif utama penelitian.
