#include "ebhi_pipeline.hpp"

#include <QApplication>
#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QStatusBar>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

namespace {

QPixmap mat_rgb_to_pixmap(const cv::Mat& rgb, int target_size = 236) {
    if (rgb.empty()) {
        return QPixmap();
    }
    QImage qimg(
        rgb.data,
        rgb.cols,
        rgb.rows,
        static_cast<int>(rgb.step),
        QImage::Format_RGB888
    );
    return QPixmap::fromImage(qimg.copy()).scaled(
        target_size,
        target_size,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
}

QPixmap mat_gray_to_pixmap(const cv::Mat& gray, int target_size = 236) {
    if (gray.empty()) {
        return QPixmap();
    }
    QImage qimg(
        gray.data,
        gray.cols,
        gray.rows,
        static_cast<int>(gray.step),
        QImage::Format_Grayscale8
    );
    return QPixmap::fromImage(qimg.copy()).scaled(
        target_size,
        target_size,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
}

std::string format_double(double val, int precision = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << val;
    return oss.str();
}

std::string infer_reference_mask_path(const std::string& image_path) {
    fs::path p(image_path);
    if (p.parent_path().filename() == "image") {
        fs::path mask_p = p.parent_path().parent_path() / "label" / p.filename();
        if (fs::exists(mask_p)) {
            return mask_p.string();
        }
    }
    return "";
}

std::string infer_ground_truth_category(const std::string& image_path) {
    fs::path p(image_path);
    if (p.parent_path().filename() == "image") {
        return p.parent_path().parent_path().filename().string();
    }
    return "";
}

class EbhiMainWindow : public QMainWindow {
public:
    explicit EbhiMainWindow(
        const std::string& repo_root,
        const std::string& initial_image = "",
        const std::string& initial_seg = "k4_M0",
        const std::string& initial_svm = ""
    )
        : repo_root_(repo_root) {
        setWindowTitle("EBHI-DIP — Analisis Segmentasi & Klasifikasi SVM Histopatologi Kolorektal (C++ / Qt6)");
        resize(1360, 860);
        setup_ui();
        populate_presets();

        if (!initial_seg.empty()) {
            int idx = seg_combo_->findData(QString::fromStdString(initial_seg));
            if (idx >= 0) {
                seg_combo_->setCurrentIndex(idx);
            }
        }
        if (!initial_svm.empty()) {
            std::string svm_filename = fs::path(initial_svm).filename().string();
            int idx = svm_combo_->findData(QString::fromStdString(svm_filename));
            if (idx >= 0) {
                svm_combo_->setCurrentIndex(idx);
            }
        }

        if (!initial_image.empty() && fs::exists(initial_image)) {
            load_image_and_run(initial_image);
        } else if (preset_combo_->count() > 1) {
            preset_combo_->setCurrentIndex(1);
        }
    }

private:
    void setup_ui() {
        auto* central = new QWidget(this);
        setCentralWidget(central);
        auto* main_layout = new QVBoxLayout(central);
        main_layout->setContentsMargins(12, 10, 12, 10);
        main_layout->setSpacing(10);

        // Top control group
        auto* ctrl_group = new QGroupBox("Konfigurasi Citra, Metode Segmentasi, dan Model Klasifikasi SVM", this);
        auto* ctrl_layout = new QGridLayout(ctrl_group);
        ctrl_layout->setSpacing(8);

        auto* lbl_preset = new QLabel("Sampel Uji Cepat:", this);
        preset_combo_ = new QComboBox(this);
        auto* btn_open_img = new QPushButton("Pilih Citra RGB...", this);
        auto* btn_open_mask = new QPushButton("Pilih Mask GT...", this);

        auto* lbl_seg = new QLabel("Metode Segmentasi:", this);
        seg_combo_ = new QComboBox(this);
        seg_combo_->addItem("k4_M0 — Primary Selected (Lab K-Means k=4, Raw M0)", "k4_M0");
        seg_combo_->addItem("otsu_M2 — Best Classifier Comparator (Otsu Dark + M2)", "otsu_M2");
        seg_combo_->addItem("k4_M1 — Lab K-Means k=4 + Area Filtering >= 16 px (M1)", "k4_M1");
        seg_combo_->addItem("k4_M2 — Lab K-Means k=4 + Area + Opening/Closing (M2)", "k4_M2");
        seg_combo_->addItem("otsu_M0 — Otsu Global Dark Polarity (Raw M0)", "otsu_M0");
        seg_combo_->addItem("otsu_M1 — Otsu + Area Filtering >= 16 px (M1)", "otsu_M1");
        seg_combo_->addItem("k3_M0 — Lab K-Means k=3 (Raw M0)", "k3_M0");
        seg_combo_->addItem("k2_M0 — Lab K-Means k=2 (Raw M0)", "k2_M0");

        auto* lbl_svm = new QLabel("Artifact Model SVM:", this);
        svm_combo_ = new QComboBox(this);
        svm_combo_->addItem("svm_k4_M0_selected.json — Primary Locked (k4_M0, Set C: 29 Fitur)", "svm_k4_M0_selected.json");
        svm_combo_->addItem("svm_otsu_M2_C.json — Comparator Terbaik (otsu_M2, Set C: 29 Fitur)", "svm_otsu_M2_C.json");
        svm_combo_->addItem("svm_k4_M0_A.json — Ablasi Set A: Appearance-Only (10 Fitur Lab+GLCM)", "svm_k4_M0_A.json");
        svm_combo_->addItem("svm_k4_M0_B.json — Ablasi Set B: Morphology-Only (19 Fitur Mask)", "svm_k4_M0_B.json");
        svm_combo_->addItem("svm_otsu_M0_C.json — Comparator Otsu Raw (otsu_M0, Set C: 29 Fitur)", "svm_otsu_M0_C.json");
        svm_combo_->addItem("svm_k4_M1_C.json — Comparator K-Means M1 (k4_M1, Set C: 29 Fitur)", "svm_k4_M1_C.json");

        auto* btn_run = new QPushButton("Jalankan Pipeline", this);
        btn_run->setStyleSheet("font-weight: bold; padding: 5px 12px;");

        path_label_ = new QLabel("Citra aktif: (belum dipilih)", this);
        path_label_->setStyleSheet("color: #34495e; font-size: 11px;");

        ctrl_layout->addWidget(lbl_preset, 0, 0);
        ctrl_layout->addWidget(preset_combo_, 0, 1);
        ctrl_layout->addWidget(btn_open_img, 0, 2);
        ctrl_layout->addWidget(btn_open_mask, 0, 3);

        ctrl_layout->addWidget(lbl_seg, 1, 0);
        ctrl_layout->addWidget(seg_combo_, 1, 1);
        ctrl_layout->addWidget(lbl_svm, 1, 2);
        ctrl_layout->addWidget(svm_combo_, 1, 3);
        ctrl_layout->addWidget(btn_run, 1, 4);
        ctrl_layout->addWidget(path_label_, 2, 0, 1, 5);

        main_layout->addWidget(ctrl_group);

        // Visual panels row (4 panels side-by-side)
        auto* vis_group = new QGroupBox("Perbandingan Visual Citra, Prediksi Mask, Ground Truth, dan Error Overlay", this);
        auto* vis_layout = new QHBoxLayout(vis_group);
        vis_layout->setSpacing(12);

        img_rgb_label_ = create_image_panel(vis_layout, "1. Citra Histopatologi Asli (RGB)", &cap_rgb_label_);
        img_pred_label_ = create_image_panel(vis_layout, "2. Prediksi Mask Segmentasi", &cap_pred_label_);
        img_ref_label_ = create_image_panel(vis_layout, "3. Ground Truth Reference Mask", &cap_ref_label_);
        img_ov_label_ = create_image_panel(vis_layout, "4. Error Overlay (TP Hijau / FP Oranye / FN Magenta)", &cap_ov_label_);

        main_layout->addWidget(vis_group);

        // Bottom row: Left = Classification & Segmentation Summary; Right = 29 Feature Table
        auto* bottom_layout = new QHBoxLayout();
        bottom_layout->setSpacing(12);

        // Left summary group
        auto* summary_group = new QGroupBox("Hasil Klasifikasi SVM (6 Kelas) & Metrik Evaluasi Segmentasi", this);
        auto* summary_layout = new QVBoxLayout(summary_group);

        pred_banner_ = new QLabel("Prediksi: -", this);
        pred_banner_->setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 8px; "
            "background-color: #eef6ff; border: 1px solid #b6d4fe; border-radius: 4px;"
        );
        summary_layout->addWidget(pred_banner_);

        seg_metrics_label_ = new QLabel("Metrik Segmentasi: -", this);
        seg_metrics_label_->setStyleSheet(
            "font-size: 12px; padding: 6px; background-color: #f8f9fa; "
            "border: 1px solid #dee2e6; border-radius: 4px;"
        );
        summary_layout->addWidget(seg_metrics_label_);

        votes_table_ = new QTableWidget(6, 3, this);
        votes_table_->setHorizontalHeaderLabels({"Kategori Histologi", "Suara OVO (/15)", "Proporsi Suara"});
        votes_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        votes_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        votes_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
        votes_table_->verticalHeader()->setVisible(false);
        votes_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        votes_table_->setSelectionMode(QAbstractItemView::NoSelection);
        summary_layout->addWidget(votes_table_);

        bottom_layout->addWidget(summary_group, 5);

        // Right feature table group
        auto* feat_group = new QGroupBox("Vektor 29 Fitur Ekstraksi (Set A: 10 CIE Lab + GLCM | Set B: 19 Morfologi Mask)", this);
        auto* feat_layout = new QVBoxLayout(feat_group);

        feat_table_ = new QTableWidget(29, 4, this);
        feat_table_->setHorizontalHeaderLabels({"Nama Fitur", "Kelompok", "Nilai Mentah", "Z-Score (Scaler)"});
        feat_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        feat_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        feat_table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
        feat_table_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
        feat_table_->verticalHeader()->setVisible(false);
        feat_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        feat_layout->addWidget(feat_table_);

        bottom_layout->addWidget(feat_group, 6);
        main_layout->addLayout(bottom_layout, 1);

        statusBar()->showMessage("Siap. Pilih sampel uji atau buka citra histopatologi RGB (.png).");

        connect(btn_open_img, &QPushButton::clicked, this, &EbhiMainWindow::on_browse_image);
        connect(btn_open_mask, &QPushButton::clicked, this, &EbhiMainWindow::on_browse_mask);
        connect(btn_run, &QPushButton::clicked, this, &EbhiMainWindow::run_current);
        connect(preset_combo_, &QComboBox::currentIndexChanged, this, &EbhiMainWindow::on_preset_changed);
        connect(seg_combo_, &QComboBox::currentIndexChanged, this, &EbhiMainWindow::on_seg_method_changed);
        connect(svm_combo_, &QComboBox::currentIndexChanged, this, &EbhiMainWindow::run_current);
    }

    QLabel* create_image_panel(QHBoxLayout* parent_layout, const QString& title, QLabel** out_caption) {
        auto* box = new QVBoxLayout();
        auto* title_lbl = new QLabel(title, this);
        title_lbl->setAlignment(Qt::AlignCenter);
        title_lbl->setStyleSheet("font-weight: bold; font-size: 12px;");

        auto* img_lbl = new QLabel(this);
        img_lbl->setFixedSize(240, 240);
        img_lbl->setAlignment(Qt::AlignCenter);
        img_lbl->setStyleSheet("background-color: #ffffff; border: 1px solid #ced4da;");

        auto* cap_lbl = new QLabel("-", this);
        cap_lbl->setAlignment(Qt::AlignCenter);
        cap_lbl->setStyleSheet("font-size: 11px; color: #495057;");
        *out_caption = cap_lbl;

        box->addWidget(title_lbl);
        box->addWidget(img_lbl, 0, Qt::AlignCenter);
        box->addWidget(cap_lbl);
        parent_layout->addLayout(box);
        return img_lbl;
    }

    void populate_presets() {
        preset_combo_->addItem("-- Pilih Sampel Citra dari Dataset EBHI-SEG --", "");
        fs::path raw_root = fs::path(repo_root_) / "data" / "raw" / "EBHI-SEG";
        if (!fs::exists(raw_root)) {
            return;
        }
        const std::vector<std::string> categories = {
            "Adenocarcinoma",
            "High-grade IN",
            "Low-grade IN",
            "Normal",
            "Polyp",
            "Serrated adenoma"
        };
        for (const auto& cat : categories) {
            fs::path img_dir = raw_root / cat / "image";
            if (!fs::exists(img_dir)) {
                continue;
            }
            std::vector<fs::path> files;
            for (const auto& entry : fs::directory_iterator(img_dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".png") {
                    files.push_back(entry.path());
                }
            }
            std::sort(files.begin(), files.end());
            int added = 0;
            for (const auto& f : files) {
                QString label = QString::fromStdString("[" + cat + "] " + f.filename().string());
                preset_combo_->addItem(label, QString::fromStdString(f.string()));
                if (++added >= 2) {
                    break;
                }
            }
        }
    }

    void on_preset_changed(int index) {
        if (index <= 0) {
            return;
        }
        std::string img_path = preset_combo_->itemData(index).toString().toStdString();
        if (!img_path.empty() && fs::exists(img_path)) {
            load_image_and_run(img_path);
        }
    }

    void on_seg_method_changed(int /*index*/) {
        std::string seg = seg_combo_->currentData().toString().toStdString();
        // Auto-sync compatible default SVM artifact when switching between primary k4_M0 and otsu_M2
        if (seg == "otsu_M2") {
            int idx = svm_combo_->findData("svm_otsu_M2_C.json");
            if (idx >= 0 && svm_combo_->currentIndex() != idx) {
                svm_combo_->setCurrentIndex(idx);
                return;
            }
        } else if (seg == "k4_M0") {
            int idx = svm_combo_->findData("svm_k4_M0_selected.json");
            if (idx >= 0 && svm_combo_->currentIndex() != idx) {
                svm_combo_->setCurrentIndex(idx);
                return;
            }
        }
        run_current();
    }

    void on_browse_image() {
        fs::path start_dir = fs::path(repo_root_) / "data" / "raw" / "EBHI-SEG";
        if (!fs::exists(start_dir)) {
            start_dir = fs::path(repo_root_);
        }
        QString file = QFileDialog::getOpenFileName(
            this,
            "Pilih Citra Histopatologi RGB",
            QString::fromStdString(start_dir.string()),
            "PNG Images (*.png);;All Files (*.*)"
        );
        if (!file.isEmpty()) {
            load_image_and_run(file.toStdString());
        }
    }

    void on_browse_mask() {
        QString file = QFileDialog::getOpenFileName(
            this,
            "Pilih Ground Truth Reference Mask",
            QString::fromStdString(repo_root_),
            "PNG Masks (*.png);;All Files (*.*)"
        );
        if (!file.isEmpty()) {
            current_mask_path_ = file.toStdString();
            run_current();
        }
    }

    void load_image_and_run(const std::string& image_path) {
        current_image_path_ = image_path;
        current_mask_path_ = infer_reference_mask_path(image_path);
        run_current();
    }

    void run_current() {
        if (current_image_path_.empty() || !fs::exists(current_image_path_)) {
            return;
        }
        std::string seg_method = seg_combo_->currentData().toString().toStdString();
        std::string svm_file = svm_combo_->currentData().toString().toStdString();
        std::string svm_path = (fs::path(repo_root_) / "models" / svm_file).string();

        try {
            ebhi::SvmArtifact svm_art = ebhi::load_svm_artifact(svm_path);
            ebhi::PipelineResult res = ebhi::run_full_pipeline(
                current_image_path_,
                current_mask_path_,
                seg_method,
                svm_path,
                repo_root_
            );

            std::string gt_cat = infer_ground_truth_category(current_image_path_);
            path_label_->setText(QString::fromStdString(
                "Citra: " + current_image_path_ +
                (current_mask_path_.empty() ? " | Mask GT: (tidak tersedia)" : (" | Mask GT: " + current_mask_path_))
            ));

            // Update 4 visual panels
            img_rgb_label_->setPixmap(mat_rgb_to_pixmap(res.rgb_u8));
            cap_rgb_label_->setText(QString::fromStdString(
                std::to_string(res.rgb_u8.cols) + "x" + std::to_string(res.rgb_u8.rows) +
                (gt_cat.empty() ? " RGB" : (" | GT: " + gt_cat))
            ));

            img_pred_label_->setPixmap(mat_gray_to_pixmap(res.pred_mask_u8));
            std::string pred_cap = "Metode: " + seg_method +
                " | FG: " + format_double(res.overlap.pred_fg_fraction * 100.0, 1) + "%";
            if (res.otsu_threshold >= 0.0) {
                pred_cap += " (T=" + format_double(res.otsu_threshold, 0) + ")";
            }
            cap_pred_label_->setText(QString::fromStdString(pred_cap));

            if (res.overlap.available) {
                img_ref_label_->setPixmap(mat_gray_to_pixmap(res.ref_mask_u8));
                cap_ref_label_->setText(QString::fromStdString(
                    "Ref FG: " + format_double(res.overlap.ref_fg_fraction * 100.0, 1) + "%"
                ));
            } else {
                img_ref_label_->clear();
                img_ref_label_->setText("Ground Truth Mask\nTidak Tersedia\n(Unlabeled Mode)");
                cap_ref_label_->setText("Hanya overlay foreground");
            }

            img_ov_label_->setPixmap(mat_rgb_to_pixmap(res.overlay_rgb_u8));
            if (res.overlap.available) {
                cap_ov_label_->setText(QString::fromStdString(
                    "Dice: " + format_double(res.overlap.dice, 4) +
                    " | IoU: " + format_double(res.overlap.iou, 4)
                ));
            } else {
                cap_ov_label_->setText("Overlay Prediksi Foreground (Hijau)");
            }

            // Update Classification Banner
            int max_votes = res.classification.ovo_votes[res.classification.predicted_index];
            std::string banner_txt = "Prediksi SVM: " + res.classification.predicted_category +
                "  (" + std::to_string(max_votes) + " / 15 suara OVO | Model: " + svm_art.model_id + ")";
            if (!gt_cat.empty()) {
                bool match = (gt_cat == res.classification.predicted_category);
                banner_txt += "   |   Label Asli: " + gt_cat + (match ? " [SESUAI]" : " [BERBEDA]");
                pred_banner_->setStyleSheet(
                    match
                    ? "font-size: 13px; font-weight: bold; padding: 8px; background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; border-radius: 4px;"
                    : "font-size: 13px; font-weight: bold; padding: 8px; background-color: #fff3cd; color: #664d03; border: 1px solid #ffecb5; border-radius: 4px;"
                );
            }
            pred_banner_->setText(QString::fromStdString(banner_txt));

            // Update Segmentation Metrics Label
            int comp_cnt = static_cast<int>(res.features.at("mask_component_count"));
            if (res.overlap.available) {
                seg_metrics_label_->setText(QString::fromStdString(
                    "Evaluasi Segmentasi vs Ground Truth  —  Dice: " + format_double(res.overlap.dice, 4) +
                    "  |  IoU: " + format_double(res.overlap.iou, 4) +
                    "  |  Precision: " + format_double(res.overlap.precision, 4) +
                    "  |  Recall: " + format_double(res.overlap.recall, 4) +
                    "  |  Komponen: " + std::to_string(comp_cnt)
                ));
            } else {
                seg_metrics_label_->setText(QString::fromStdString(
                    "Statistik Mask Prediksi  —  Foreground Coverage: " +
                    format_double(res.overlap.pred_fg_fraction * 100.0, 2) + "%  |  Komponen Terhubung: " +
                    std::to_string(comp_cnt)
                ));
            }

            // Update OVO Votes table
            votes_table_->setRowCount(static_cast<int>(svm_art.classes.size()));
            for (size_t c = 0; c < svm_art.classes.size(); ++c) {
                const int row = static_cast<int>(c);
                const int v = res.classification.ovo_votes[c];
                auto* item_cat = new QTableWidgetItem(QString::fromStdString(svm_art.classes[c]));
                auto* item_vote = new QTableWidgetItem(QString::number(v) + " / 15");
                item_vote->setTextAlignment(Qt::AlignCenter);

                if (static_cast<int>(c) == res.classification.predicted_index) {
                    QFont f = item_cat->font();
                    f.setBold(true);
                    item_cat->setFont(f);
                    item_vote->setFont(f);
                }

                auto* bar = new QProgressBar(this);
                bar->setRange(0, 15);
                bar->setValue(v);
                bar->setFormat(QString::number(v) + " suara");

                votes_table_->setItem(row, 0, item_cat);
                votes_table_->setItem(row, 1, item_vote);
                votes_table_->setCellWidget(row, 2, bar);
            }

            // Map active standardized z-scores by feature name
            std::map<std::string, double> z_map;
            for (size_t i = 0; i < svm_art.feature_names.size(); ++i) {
                z_map[svm_art.feature_names[i]] = res.classification.standardized_features[i];
            }

            // Update 29 Features table
            const auto& cols = res.feature_names;
            feat_table_->setRowCount(static_cast<int>(cols.size()));
            for (size_t i = 0; i < cols.size(); ++i) {
                const int row = static_cast<int>(i);
                const std::string& fname = cols[i];
                const bool is_set_a = (i < 10);
                const double raw_val = res.features.at(fname);

                auto* item_name = new QTableWidgetItem(QString::fromStdString(fname));
                auto* item_grp = new QTableWidgetItem(is_set_a ? "Set A (Lab/GLCM)" : "Set B (Morphology)");
                auto* item_raw = new QTableWidgetItem(QString::fromStdString(format_double(raw_val, 6)));
                item_raw->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

                QString z_str = "n/a (tidak aktif)";
                auto z_it = z_map.find(fname);
                if (z_it != z_map.end()) {
                    z_str = QString::fromStdString(format_double(z_it->second, 4));
                }
                auto* item_z = new QTableWidgetItem(z_str);
                item_z->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

                feat_table_->setItem(row, 0, item_name);
                feat_table_->setItem(row, 1, item_grp);
                feat_table_->setItem(row, 2, item_raw);
                feat_table_->setItem(row, 3, item_z);
            }

            statusBar()->showMessage(QString::fromStdString(
                "Pipeline selesai: " + fs::path(current_image_path_).filename().string() +
                " | Segmentasi: " + seg_method +
                " | Prediksi SVM: " + res.classification.predicted_category
            ));
        } catch (const std::exception& ex) {
            QMessageBox::critical(this, "Kesalahan Eksekusi Pipeline", QString::fromStdString(ex.what()));
        }
    }

    std::string repo_root_;
    std::string current_image_path_;
    std::string current_mask_path_;

    QComboBox* preset_combo_ = nullptr;
    QComboBox* seg_combo_ = nullptr;
    QComboBox* svm_combo_ = nullptr;
    QLabel* path_label_ = nullptr;

    QLabel* img_rgb_label_ = nullptr;
    QLabel* img_pred_label_ = nullptr;
    QLabel* img_ref_label_ = nullptr;
    QLabel* img_ov_label_ = nullptr;

    QLabel* cap_rgb_label_ = nullptr;
    QLabel* cap_pred_label_ = nullptr;
    QLabel* cap_ref_label_ = nullptr;
    QLabel* cap_ov_label_ = nullptr;

    QLabel* pred_banner_ = nullptr;
    QLabel* seg_metrics_label_ = nullptr;
    QTableWidget* votes_table_ = nullptr;
    QTableWidget* feat_table_ = nullptr;
};

}  // namespace

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    std::string repo_root = ".";
    std::string initial_image;
    std::string initial_seg = "k4_M0";
    std::string initial_svm;
    std::string screenshot_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--repo-root" && i + 1 < argc) {
            repo_root = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            initial_image = argv[++i];
        } else if (arg == "--seg" && i + 1 < argc) {
            initial_seg = argv[++i];
        } else if (arg == "--svm" && i + 1 < argc) {
            initial_svm = argv[++i];
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
        }
    }

    if (!fs::exists(fs::path(repo_root) / "models" / "svm_k4_M0_selected.json")) {
        fs::path parent = fs::path(repo_root).parent_path();
        if (fs::exists(parent / "models" / "svm_k4_M0_selected.json")) {
            repo_root = parent.string();
        }
    }

    EbhiMainWindow window(repo_root, initial_image, initial_seg, initial_svm);
    window.show();

    if (!screenshot_path.empty()) {
        QTimer::singleShot(250, [&window, &app, screenshot_path]() {
            QPixmap pix = window.grab();
            pix.save(QString::fromStdString(screenshot_path), "PNG");
            app.quit();
        });
    }

    return app.exec();
}
