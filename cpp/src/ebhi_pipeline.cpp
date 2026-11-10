#include "ebhi_pipeline.hpp"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace ebhi {

namespace {

constexpr double kPi = 3.14159265358979323846;

QJsonObject read_json_object(const std::string& json_path) {
    QFile file(QString::fromStdString(json_path));
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Cannot open JSON file: " + json_path);
    }
    QByteArray raw = file.readAll();
    QJsonParseError parse_err;
    QJsonDocument doc = QJsonDocument::fromJson(raw, &parse_err);
    if (parse_err.error != QJsonParseError::NoError || !doc.isObject()) {
        throw std::runtime_error("Invalid JSON in file: " + json_path);
    }
    return doc.object();
}

std::array<double, 3> summarize_vec(const std::vector<double>& vals) {
    if (vals.empty()) {
        return {0.0, 0.0, 0.0};
    }
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    double mean = sum / static_cast<double>(vals.size());
    double sq_sum = 0.0;
    for (double v : vals) {
        double d = v - mean;
        sq_sum += d * d;
    }
    double std_v = std::sqrt(sq_sum / static_cast<double>(vals.size()));

    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    double median_v = (n % 2 == 1)
        ? sorted[n / 2]
        : (0.5 * (sorted[n / 2 - 1] + sorted[n / 2]));
    return {mean, std_v, median_v};
}

double compute_skimage_perimeter4(const cv::Mat& sub_u8) {
    const int H = sub_u8.rows;
    const int W = sub_u8.cols;
    int total_fg = 0;
    for (int r = 0; r < H; ++r) {
        const uchar* row_ptr = sub_u8.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            if (row_ptr[c] != 0) {
                ++total_fg;
            }
        }
    }
    if (total_fg <= 1) {
        return 0.0;
    }

    // 4-connected binary erosion with zero outside bounds
    cv::Mat border_u8 = cv::Mat::zeros(H, W, CV_8UC1);
    for (int r = 0; r < H; ++r) {
        const uchar* cur = sub_u8.ptr<uchar>(r);
        const uchar* up = (r > 0) ? sub_u8.ptr<uchar>(r - 1) : nullptr;
        const uchar* dn = (r + 1 < H) ? sub_u8.ptr<uchar>(r + 1) : nullptr;
        uchar* b_row = border_u8.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            if (cur[c] == 0) {
                continue;
            }
            bool eroded = (r > 0 && up[c] != 0) &&
                          (r + 1 < H && dn[c] != 0) &&
                          (c > 0 && cur[c - 1] != 0) &&
                          (c + 1 < W && cur[c + 1] != 0);
            if (!eroded) {
                b_row[c] = 1;
            }
        }
    }

    // Benkrid & Crookes (1999) lookup table used by skimage.measure.perimeter(neighborhood=4)
    static const double kSqrt2 = std::sqrt(2.0);
    static const double kDiagAvg = (1.0 + kSqrt2) / 2.0;
    std::array<double, 50> weights{};
    weights.fill(0.0);
    for (int idx : {5, 7, 15, 17, 25, 27}) {
        weights[idx] = 1.0;
    }
    for (int idx : {21, 33}) {
        weights[idx] = kSqrt2;
    }
    for (int idx : {13, 23}) {
        weights[idx] = kDiagAvg;
    }

    static const int kKernel[3][3] = {
        {10, 2, 10},
        { 2, 1,  2},
        {10, 2, 10}
    };

    double perimeter_total = 0.0;
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int code = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                int nr = r + dr;
                if (nr < 0 || nr >= H) {
                    continue;
                }
                const uchar* b_row = border_u8.ptr<uchar>(nr);
                for (int dc = -1; dc <= 1; ++dc) {
                    int nc = c + dc;
                    if (nc >= 0 && nc < W && b_row[nc] != 0) {
                        code += kKernel[dr + 1][dc + 1];
                    }
                }
            }
            if (code >= 0 && code < static_cast<int>(weights.size())) {
                perimeter_total += weights[code];
            }
        }
    }
    return perimeter_total;
}

int compute_convex_hull_pixel_area(const cv::Mat& sub_u8) {
    const int H = sub_u8.rows;
    const int W = sub_u8.cols;
    using Pt = std::pair<long long, long long>;
    std::vector<Pt> pts;
    pts.reserve(static_cast<size_t>(H) * 8);

    for (int r = 0; r < H; ++r) {
        const uchar* row_ptr = sub_u8.ptr<uchar>(r);
        int first_c = -1;
        int last_c = -1;
        for (int c = 0; c < W; ++c) {
            if (row_ptr[c] != 0) {
                if (first_c < 0) {
                    first_c = c;
                }
                last_c = c;
            }
        }
        if (first_c >= 0) {
            for (int c : {first_c, last_c}) {
                long long r2 = 2LL * r;
                long long c2 = 2LL * c;
                pts.push_back({r2 - 1, c2});
                pts.push_back({r2 + 1, c2});
                pts.push_back({r2, c2 - 1});
                pts.push_back({r2, c2 + 1});
            }
        }
    }
    if (pts.empty()) {
        return 0;
    }
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

    auto cross = [](const Pt& o, const Pt& a, const Pt& b) -> long long {
        return (a.first - o.first) * (b.second - o.second) -
               (a.second - o.second) * (b.first - o.first);
    };

    std::vector<Pt> lower;
    for (const auto& p : pts) {
        while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), p) <= 0) {
            lower.pop_back();
        }
        lower.push_back(p);
    }
    std::vector<Pt> upper;
    for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
        while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), *it) <= 0) {
            upper.pop_back();
        }
        upper.push_back(*it);
    }
    lower.pop_back();
    upper.pop_back();
    std::vector<Pt> hull = std::move(lower);
    hull.insert(hull.end(), upper.begin(), upper.end());

    const size_t n = hull.size();
    int count = 0;
    for (int r = 0; r < H; ++r) {
        const double pr = 2.0 * static_cast<double>(r);
        double c_min = -1e18;
        double c_max = 1e18;
        for (size_t i = 0; i < n; ++i) {
            const auto& p1 = hull[i];
            const auto& p2 = hull[(i + 1) % n];
            const double dr = static_cast<double>(p2.first - p1.first);
            const double dc = static_cast<double>(p2.second - p1.second);
            const double rhs = dc * (pr - static_cast<double>(p1.first));
            if (dr > 0.0) {
                c_min = std::max(c_min, static_cast<double>(p1.second) + rhs / dr);
            } else if (dr < 0.0) {
                c_max = std::min(c_max, static_cast<double>(p1.second) + rhs / dr);
            } else if (rhs > 0.0) {
                c_min = 1e18;
                break;
            }
        }
        int lo = std::max(0, static_cast<int>(std::ceil((c_min - 1e-12) / 2.0)));
        int hi = std::min(W - 1, static_cast<int>(std::floor((c_max + 1e-12) / 2.0)));
        if (hi >= lo) {
            count += (hi - lo + 1);
        }
    }
    return count;
}

double compute_eccentricity(const cv::Mat& sub_u8, int area_px) {
    if (area_px <= 1) {
        return 0.0;
    }
    const int H = sub_u8.rows;
    const int W = sub_u8.cols;
    double sum_r = 0.0;
    double sum_c = 0.0;
    for (int r = 0; r < H; ++r) {
        const uchar* row_ptr = sub_u8.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            if (row_ptr[c] != 0) {
                sum_r += static_cast<double>(r);
                sum_c += static_cast<double>(c);
            }
        }
    }
    const double area = static_cast<double>(area_px);
    const double mean_r = sum_r / area;
    const double mean_c = sum_c / area;

    double mu_20 = 0.0;
    double mu_02 = 0.0;
    double mu_11 = 0.0;
    for (int r = 0; r < H; ++r) {
        const uchar* row_ptr = sub_u8.ptr<uchar>(r);
        const double dr = static_cast<double>(r) - mean_r;
        for (int c = 0; c < W; ++c) {
            if (row_ptr[c] != 0) {
                const double dc = static_cast<double>(c) - mean_c;
                mu_20 += dr * dr;
                mu_02 += dc * dc;
                mu_11 += dr * dc;
            }
        }
    }

    // Inertia tensor eigenvalues of 2D region (skimage moments_central -> inertia_tensor_eigvals)
    const double a = mu_20 / area;
    const double b = mu_02 / area;
    const double c = -mu_11 / area;
    const double trace = a + b;
    const double diff = a - b;
    const double disc = std::sqrt(std::max(0.0, diff * diff + 4.0 * c * c));
    const double l1 = 0.5 * (trace + disc);
    const double l2 = 0.5 * (trace - disc);
    if (l1 <= 0.0) {
        return 0.0;
    }
    return std::sqrt(std::max(0.0, 1.0 - (l2 / l1)));
}

}  // namespace

const std::vector<std::string>& canonical_feature_columns() {
    static const std::vector<std::string> kColumns = {
        "lab_l_mean",
        "lab_l_std",
        "lab_a_mean",
        "lab_a_std",
        "lab_b_mean",
        "lab_b_std",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
        "mask_foreground_fraction",
        "mask_component_count",
        "mask_edge_foreground_fraction",
        "mask_is_empty",
        "region_area_rel_mean",
        "region_area_rel_std",
        "region_area_rel_median",
        "region_perimeter_rel_mean",
        "region_perimeter_rel_std",
        "region_perimeter_rel_median",
        "region_circularity_mean",
        "region_circularity_std",
        "region_circularity_median",
        "region_eccentricity_mean",
        "region_eccentricity_std",
        "region_eccentricity_median",
        "region_solidity_mean",
        "region_solidity_std",
        "region_solidity_median",
    };
    return kColumns;
}

OtsuArtifact load_otsu_artifact(const std::string& json_path) {
    QJsonObject obj = read_json_object(json_path);
    OtsuArtifact art;
    if (obj.contains("foreground_polarity")) {
        art.foreground_polarity = obj.value("foreground_polarity").toString().toStdString();
    }
    return art;
}

KMeansArtifact load_kmeans_artifact(const std::string& json_path) {
    QJsonObject obj = read_json_object(json_path);
    KMeansArtifact art;
    art.n_clusters = obj.value("n_clusters").toInt();
    QJsonArray centers = obj.value("cluster_centers").toArray();
    for (const auto& c_val : centers) {
        QJsonArray c_arr = c_val.toArray();
        if (c_arr.size() != 3) {
            throw std::runtime_error("Invalid cluster center dimension in: " + json_path);
        }
        art.cluster_centers.push_back({
            c_arr[0].toDouble(),
            c_arr[1].toDouble(),
            c_arr[2].toDouble()
        });
    }
    QJsonArray fg = obj.value("foreground_clusters").toArray();
    for (const auto& f_val : fg) {
        art.foreground_clusters.push_back(f_val.toInt());
    }
    return art;
}

SvmArtifact load_svm_artifact(const std::string& json_path) {
    QJsonObject obj = read_json_object(json_path);
    SvmArtifact art;
    art.model_id = obj.value("model_id").toString().toStdString();
    art.segmentation_method = obj.value("segmentation_method").toString().toStdString();
    art.feature_set = obj.value("feature_set").toString().toStdString();
    art.kernel = obj.value("kernel").toString().toStdString();
    art.C = obj.value("C").toDouble();
    art.gamma_value = obj.value("gamma_value").toDouble();

    for (const auto& v : obj.value("feature_names").toArray()) {
        art.feature_names.push_back(v.toString().toStdString());
    }
    for (const auto& v : obj.value("classes").toArray()) {
        art.classes.push_back(v.toString().toStdString());
    }
    QJsonObject scaler = obj.value("scaler").toObject();
    for (const auto& v : scaler.value("mean").toArray()) {
        art.scaler_mean.push_back(v.toDouble());
    }
    for (const auto& v : scaler.value("scale").toArray()) {
        art.scaler_scale.push_back(v.toDouble());
    }
    for (const auto& v : obj.value("n_support").toArray()) {
        art.n_support.push_back(v.toInt());
    }
    for (const auto& row_val : obj.value("support_vectors").toArray()) {
        std::vector<double> row;
        for (const auto& v : row_val.toArray()) {
            row.push_back(v.toDouble());
        }
        art.support_vectors.push_back(std::move(row));
    }
    for (const auto& row_val : obj.value("dual_coef").toArray()) {
        std::vector<double> row;
        for (const auto& v : row_val.toArray()) {
            row.push_back(v.toDouble());
        }
        art.dual_coef.push_back(std::move(row));
    }
    for (const auto& v : obj.value("intercept").toArray()) {
        art.intercept.push_back(v.toDouble());
    }
    if (obj.contains("linear_ovo_weights")) {
        for (const auto& row_val : obj.value("linear_ovo_weights").toArray()) {
            std::vector<double> row;
            for (const auto& v : row_val.toArray()) {
                row.push_back(v.toDouble());
            }
            art.linear_ovo_weights.push_back(std::move(row));
        }
    }
    return art;
}

cv::Mat load_rgb_image(const std::string& image_path) {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Cannot load RGB image: " + image_path);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

cv::Mat load_reference_mask(const std::string& mask_path) {
    if (mask_path.empty() || !fs::exists(mask_path)) {
        return cv::Mat();
    }
    cv::Mat gray = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        return cv::Mat();
    }
    cv::Mat bin(gray.rows, gray.cols, CV_8UC1);
    for (int r = 0; r < gray.rows; ++r) {
        const uchar* src = gray.ptr<uchar>(r);
        uchar* dst = bin.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c) {
            dst[c] = (src[c] >= 128) ? 255 : 0;
        }
    }
    return bin;
}

cv::Mat rgb_to_gray(const cv::Mat& rgb_u8) {
    cv::Mat gray;
    cv::cvtColor(rgb_u8, gray, cv::COLOR_RGB2GRAY);
    return gray;
}

cv::Mat rgb_to_lab(const cv::Mat& rgb_u8) {
    cv::Mat rgb_f32;
    rgb_u8.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);
    cv::Mat lab_f32;
    cv::cvtColor(rgb_f32, lab_f32, cv::COLOR_RGB2Lab);
    return lab_f32;
}

std::pair<cv::Mat, double> otsu_threshold_mask(
    const cv::Mat& gray_u8,
    const std::string& foreground_polarity
) {
    cv::Mat bright;
    double thresh = cv::threshold(gray_u8, bright, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat mask(gray_u8.rows, gray_u8.cols, CV_8UC1);
    const bool dark_fg = (foreground_polarity == "dark");
    for (int r = 0; r < gray_u8.rows; ++r) {
        const uchar* b_ptr = bright.ptr<uchar>(r);
        uchar* m_ptr = mask.ptr<uchar>(r);
        for (int c = 0; c < gray_u8.cols; ++c) {
            if (dark_fg) {
                m_ptr[c] = (b_ptr[c] == 0) ? 255 : 0;
            } else {
                m_ptr[c] = (b_ptr[c] != 0) ? 255 : 0;
            }
        }
    }
    return {mask, thresh};
}

cv::Mat kmeans_raw_mask(
    const cv::Mat& rgb_u8,
    const KMeansArtifact& artifact
) {
    cv::Mat lab = rgb_to_lab(rgb_u8);
    std::set<int> fg_set(artifact.foreground_clusters.begin(), artifact.foreground_clusters.end());
    cv::Mat mask(rgb_u8.rows, rgb_u8.cols, CV_8UC1);
    const int k = static_cast<int>(artifact.cluster_centers.size());

    for (int r = 0; r < lab.rows; ++r) {
        const cv::Vec3f* l_ptr = lab.ptr<cv::Vec3f>(r);
        uchar* m_ptr = mask.ptr<uchar>(r);
        for (int c = 0; c < lab.cols; ++c) {
            const double p0 = static_cast<double>(l_ptr[c][0]);
            const double p1 = static_cast<double>(l_ptr[c][1]);
            const double p2 = static_cast<double>(l_ptr[c][2]);
            int best_idx = 0;
            double best_dist = 1e300;
            for (int idx = 0; idx < k; ++idx) {
                const auto& ctr = artifact.cluster_centers[idx];
                const double d0 = p0 - ctr[0];
                const double d1 = p1 - ctr[1];
                const double d2 = p2 - ctr[2];
                const double dist = d0 * d0 + d1 * d1 + d2 * d2;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            m_ptr[c] = (fg_set.count(best_idx) > 0) ? 255 : 0;
        }
    }
    return mask;
}

std::map<std::string, cv::Mat> morphology_variants(
    const cv::Mat& raw_mask_u8,
    int min_area
) {
    cv::Mat m0(raw_mask_u8.rows, raw_mask_u8.cols, CV_8UC1);
    int fg_count = 0;
    for (int r = 0; r < raw_mask_u8.rows; ++r) {
        const uchar* src = raw_mask_u8.ptr<uchar>(r);
        uchar* dst = m0.ptr<uchar>(r);
        for (int c = 0; c < raw_mask_u8.cols; ++c) {
            uchar v = (src[c] != 0) ? 1 : 0;
            dst[c] = v;
            fg_count += v;
        }
    }

    cv::Mat m1 = cv::Mat::zeros(raw_mask_u8.rows, raw_mask_u8.cols, CV_8UC1);
    if (fg_count > 0) {
        cv::Mat labels, stats, centroids;
        int n_labels = cv::connectedComponentsWithStats(m0, labels, stats, centroids, 8, CV_32S);
        std::vector<uchar> keep(n_labels, 0);
        for (int lbl = 1; lbl < n_labels; ++lbl) {
            if (stats.at<int>(lbl, cv::CC_STAT_AREA) >= min_area) {
                keep[lbl] = 255;
            }
        }
        for (int r = 0; r < labels.rows; ++r) {
            const int* l_ptr = labels.ptr<int>(r);
            uchar* dst = m1.ptr<uchar>(r);
            for (int c = 0; c < labels.cols; ++c) {
                dst[c] = keep[l_ptr[c]];
            }
        }
    }

    // Scale m0 to 0/255 as well
    cv::Mat m0_255(raw_mask_u8.rows, raw_mask_u8.cols, CV_8UC1);
    for (int r = 0; r < m0.rows; ++r) {
        const uchar* src = m0.ptr<uchar>(r);
        uchar* dst = m0_255.ptr<uchar>(r);
        for (int c = 0; c < m0.cols; ++c) {
            dst[c] = src[c] ? 255 : 0;
        }
    }

    cv::Mat m1_01(raw_mask_u8.rows, raw_mask_u8.cols, CV_8UC1);
    for (int r = 0; r < m1.rows; ++r) {
        const uchar* src = m1.ptr<uchar>(r);
        uchar* dst = m1_01.ptr<uchar>(r);
        for (int c = 0; c < m1.cols; ++c) {
            dst[c] = src[c] ? 1 : 0;
        }
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::Mat opened, closed;
    cv::morphologyEx(
        m1_01, opened, cv::MORPH_OPEN, kernel,
        cv::Point(-1, -1), 1, cv::BORDER_REPLICATE
    );
    cv::morphologyEx(
        opened, closed, cv::MORPH_CLOSE, kernel,
        cv::Point(-1, -1), 1, cv::BORDER_REPLICATE
    );

    cv::Mat m2(raw_mask_u8.rows, raw_mask_u8.cols, CV_8UC1);
    for (int r = 0; r < closed.rows; ++r) {
        const uchar* src = closed.ptr<uchar>(r);
        uchar* dst = m2.ptr<uchar>(r);
        for (int c = 0; c < closed.cols; ++c) {
            dst[c] = src[c] ? 255 : 0;
        }
    }

    return {
        {"M0", m0_255},
        {"M1", m1},
        {"M2", m2},
    };
}

std::pair<cv::Mat, double> run_segmentation(
    const cv::Mat& rgb_u8,
    const std::string& method_key,
    const std::string& repo_root
) {
    fs::path root(repo_root);
    auto sep = method_key.find('_');
    if (sep == std::string::npos) {
        throw std::runtime_error("Invalid segmentation method key: " + method_key);
    }
    std::string family = method_key.substr(0, sep);
    std::string variant = method_key.substr(sep + 1);

    cv::Mat raw_mask;
    double otsu_thresh = -1.0;
    if (family == "otsu") {
        OtsuArtifact otsu_art = load_otsu_artifact((root / "configs" / "otsu_raw.json").string());
        cv::Mat gray = rgb_to_gray(rgb_u8);
        auto res = otsu_threshold_mask(gray, otsu_art.foreground_polarity);
        raw_mask = res.first;
        otsu_thresh = res.second;
    } else if (family == "k2" || family == "k3" || family == "k4") {
        std::string fname = "kmeans_lab_" + family + "_mapped.json";
        KMeansArtifact km_art = load_kmeans_artifact((root / "models" / fname).string());
        raw_mask = kmeans_raw_mask(rgb_u8, km_art);
    } else {
        throw std::runtime_error("Unsupported segmentation family: " + family);
    }

    auto variants = morphology_variants(raw_mask, 16);
    auto it = variants.find(variant);
    if (it == variants.end()) {
        throw std::runtime_error("Unsupported morphology variant: " + variant);
    }
    return {it->second, otsu_thresh};
}

std::map<std::string, double> extract_features(
    const cv::Mat& rgb_u8,
    const cv::Mat& mask_u8
) {
    std::map<std::string, double> out;
    const int H = rgb_u8.rows;
    const int W = rgb_u8.cols;
    const double total_pixels = static_cast<double>(H) * static_cast<double>(W);
    const double scale_len = std::sqrt(total_pixels);

    // 1. CIE Lab statistics
    cv::Mat lab = rgb_to_lab(rgb_u8);
    std::array<double, 3> sum_lab{0.0, 0.0, 0.0};
    for (int r = 0; r < H; ++r) {
        const cv::Vec3f* ptr = lab.ptr<cv::Vec3f>(r);
        for (int c = 0; c < W; ++c) {
            sum_lab[0] += static_cast<double>(ptr[c][0]);
            sum_lab[1] += static_cast<double>(ptr[c][1]);
            sum_lab[2] += static_cast<double>(ptr[c][2]);
        }
    }
    std::array<double, 3> mean_lab{
        sum_lab[0] / total_pixels,
        sum_lab[1] / total_pixels,
        sum_lab[2] / total_pixels
    };
    std::array<double, 3> sq_lab{0.0, 0.0, 0.0};
    for (int r = 0; r < H; ++r) {
        const cv::Vec3f* ptr = lab.ptr<cv::Vec3f>(r);
        for (int c = 0; c < W; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                double d = static_cast<double>(ptr[c][ch]) - mean_lab[ch];
                sq_lab[ch] += d * d;
            }
        }
    }
    out["lab_l_mean"] = mean_lab[0];
    out["lab_l_std"] = std::sqrt(sq_lab[0] / total_pixels);
    out["lab_a_mean"] = mean_lab[1];
    out["lab_a_std"] = std::sqrt(sq_lab[1] / total_pixels);
    out["lab_b_mean"] = mean_lab[2];
    out["lab_b_std"] = std::sqrt(sq_lab[2] / total_pixels);

    // 2. Quantized 16-level GLCM features
    cv::Mat gray = rgb_to_gray(rgb_u8);
    cv::Mat q_img(H, W, CV_8UC1);
    for (int r = 0; r < H; ++r) {
        const uchar* g_ptr = gray.ptr<uchar>(r);
        uchar* q_ptr = q_img.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            int bin = static_cast<int>(g_ptr[c]) / 16;
            q_ptr[c] = static_cast<uchar>(std::clamp(bin, 0, 15));
        }
    }

    // Exact (dr, dc) offsets matching skimage.feature.graycomatrix(distances=[1, 2], angles=[0, pi/4, pi/2, 3*pi/4])
    static const std::array<std::pair<int, int>, 8> kGlcmOffsets = {{
        {0, 1}, {1, 1}, {1, 0}, {1, -1},
        {0, 2}, {1, 1}, {2, 0}, {1, -1}
    }};

    double sum_contrast = 0.0;
    double sum_homogeneity = 0.0;
    double sum_energy = 0.0;
    double sum_correlation = 0.0;

    for (const auto& [dr, dc] : kGlcmOffsets) {
        double P[16][16] = {};
        double count = 0.0;
        for (int r = 0; r < H; ++r) {
            int nr = r + dr;
            if (nr < 0 || nr >= H) {
                continue;
            }
            const uchar* row_cur = q_img.ptr<uchar>(r);
            const uchar* row_nxt = q_img.ptr<uchar>(nr);
            for (int c = 0; c < W; ++c) {
                int nc = c + dc;
                if (nc >= 0 && nc < W) {
                    int i = row_cur[c];
                    int j = row_nxt[nc];
                    P[i][j] += 1.0;
                    P[j][i] += 1.0;
                    count += 2.0;
                }
            }
        }
        if (count > 0.0) {
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    P[i][j] /= count;
                }
            }
        }

        double contrast = 0.0;
        double homogeneity = 0.0;
        double energy_sq = 0.0;
        double mu_i = 0.0;
        double mu_j = 0.0;
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                const double p = P[i][j];
                if (p == 0.0) {
                    continue;
                }
                const double diff = static_cast<double>(i - j);
                const double diff2 = diff * diff;
                contrast += p * diff2;
                homogeneity += p / (1.0 + diff2);
                energy_sq += p * p;
                mu_i += static_cast<double>(i) * p;
                mu_j += static_cast<double>(j) * p;
            }
        }
        double var_i = 0.0;
        double var_j = 0.0;
        double cov = 0.0;
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                const double p = P[i][j];
                if (p == 0.0) {
                    continue;
                }
                const double di = static_cast<double>(i) - mu_i;
                const double dj = static_cast<double>(j) - mu_j;
                var_i += p * di * di;
                var_j += p * dj * dj;
                cov += p * di * dj;
            }
        }
        const double denom = std::sqrt(var_i * var_j);
        const double correlation = (denom > 0.0) ? (cov / denom) : 1.0;

        sum_contrast += contrast;
        sum_homogeneity += homogeneity;
        sum_energy += std::sqrt(energy_sq);
        sum_correlation += correlation;
    }

    out["glcm_contrast"] = sum_contrast / 8.0;
    out["glcm_homogeneity"] = sum_homogeneity / 8.0;
    out["glcm_energy"] = sum_energy / 8.0;
    out["glcm_correlation"] = sum_correlation / 8.0;

    // 3. Morphology & Connected-Component Descriptors (Set B)
    cv::Mat bin01(H, W, CV_8UC1);
    int fg_pixels = 0;
    for (int r = 0; r < H; ++r) {
        const uchar* m_ptr = mask_u8.ptr<uchar>(r);
        uchar* b_ptr = bin01.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            uchar v = (m_ptr[c] != 0) ? 1 : 0;
            b_ptr[c] = v;
            fg_pixels += v;
        }
    }

    if (fg_pixels == 0) {
        for (const auto& col : canonical_feature_columns()) {
            if (out.find(col) == out.end()) {
                out[col] = 0.0;
            }
        }
        out["mask_is_empty"] = 1.0;
        return out;
    }

    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(bin01, labels, stats, centroids, 8, CV_32S);
    int n_components = n_labels - 1;

    int edge_fg_pixels = 0;
    std::vector<double> areas_rel;
    std::vector<double> perims_rel;
    std::vector<double> circularities;
    std::vector<double> solidities;
    std::vector<double> eccentricities;

    areas_rel.reserve(n_components);
    perims_rel.reserve(n_components);
    circularities.reserve(n_components);
    solidities.reserve(n_components);
    eccentricities.reserve(n_components);

    for (int lbl = 1; lbl <= n_components; ++lbl) {
        const int x0 = stats.at<int>(lbl, cv::CC_STAT_LEFT);
        const int y0 = stats.at<int>(lbl, cv::CC_STAT_TOP);
        const int w = stats.at<int>(lbl, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(lbl, cv::CC_STAT_HEIGHT);
        const int area_px = stats.at<int>(lbl, cv::CC_STAT_AREA);

        if (x0 == 0 || y0 == 0 || (x0 + w) == W || (y0 + h) == H) {
            edge_fg_pixels += area_px;
        }

        cv::Mat sub(h, w, CV_8UC1);
        for (int r = 0; r < h; ++r) {
            const int* l_ptr = labels.ptr<int>(y0 + r);
            uchar* s_ptr = sub.ptr<uchar>(r);
            for (int c = 0; c < w; ++c) {
                s_ptr[c] = (l_ptr[x0 + c] == lbl) ? 1 : 0;
            }
        }

        const double area = static_cast<double>(area_px);
        const double perim = compute_skimage_perimeter4(sub);
        const double circ = (perim > 0.0) ? ((4.0 * kPi * area) / (perim * perim)) : 0.0;
        const int hull_px = compute_convex_hull_pixel_area(sub);
        const double solid = (hull_px > 0) ? (area / static_cast<double>(hull_px)) : 0.0;
        const double ecc = compute_eccentricity(sub, area_px);

        areas_rel.push_back(area / total_pixels);
        perims_rel.push_back(perim / scale_len);
        circularities.push_back(circ);
        solidities.push_back(solid);
        eccentricities.push_back(ecc);
    }

    out["mask_foreground_fraction"] = static_cast<double>(fg_pixels) / total_pixels;
    out["mask_component_count"] = static_cast<double>(n_components);
    out["mask_edge_foreground_fraction"] = static_cast<double>(edge_fg_pixels) / static_cast<double>(fg_pixels);
    out["mask_is_empty"] = 0.0;

    auto s_area = summarize_vec(areas_rel);
    out["region_area_rel_mean"] = s_area[0];
    out["region_area_rel_std"] = s_area[1];
    out["region_area_rel_median"] = s_area[2];

    auto s_perim = summarize_vec(perims_rel);
    out["region_perimeter_rel_mean"] = s_perim[0];
    out["region_perimeter_rel_std"] = s_perim[1];
    out["region_perimeter_rel_median"] = s_perim[2];

    auto s_circ = summarize_vec(circularities);
    out["region_circularity_mean"] = s_circ[0];
    out["region_circularity_std"] = s_circ[1];
    out["region_circularity_median"] = s_circ[2];

    auto s_ecc = summarize_vec(eccentricities);
    out["region_eccentricity_mean"] = s_ecc[0];
    out["region_eccentricity_std"] = s_ecc[1];
    out["region_eccentricity_median"] = s_ecc[2];

    auto s_solid = summarize_vec(solidities);
    out["region_solidity_mean"] = s_solid[0];
    out["region_solidity_std"] = s_solid[1];
    out["region_solidity_median"] = s_solid[2];

    return out;
}

OverlapMetrics compute_overlap_metrics(
    const cv::Mat& pred_mask_u8,
    const cv::Mat& ref_mask_u8
) {
    OverlapMetrics m;
    if (pred_mask_u8.empty()) {
        return m;
    }
    const int H = pred_mask_u8.rows;
    const int W = pred_mask_u8.cols;
    const double total_px = static_cast<double>(H) * static_cast<double>(W);

    int pred_fg = 0;
    for (int r = 0; r < H; ++r) {
        const uchar* p_ptr = pred_mask_u8.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            if (p_ptr[c] != 0) {
                ++pred_fg;
            }
        }
    }
    m.pred_fg_fraction = static_cast<double>(pred_fg) / total_px;

    if (ref_mask_u8.empty() || ref_mask_u8.rows != H || ref_mask_u8.cols != W) {
        m.available = false;
        return m;
    }

    m.available = true;
    for (int r = 0; r < H; ++r) {
        const uchar* p_ptr = pred_mask_u8.ptr<uchar>(r);
        const uchar* r_ptr = ref_mask_u8.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            bool p = (p_ptr[c] != 0);
            bool g = (r_ptr[c] != 0);
            if (p && g) {
                ++m.tp;
            } else if (p && !g) {
                ++m.fp;
            } else if (!p && g) {
                ++m.fn;
            } else {
                ++m.tn;
            }
        }
    }

    m.ref_fg_fraction = static_cast<double>(m.tp + m.fn) / total_px;
    const int denom_dice = 2 * m.tp + m.fp + m.fn;
    const int denom_iou = m.tp + m.fp + m.fn;
    m.dice = (denom_dice == 0) ? 1.0 : (2.0 * static_cast<double>(m.tp) / static_cast<double>(denom_dice));
    m.iou = (denom_iou == 0) ? 1.0 : (static_cast<double>(m.tp) / static_cast<double>(denom_iou));
    m.precision = (m.tp + m.fp == 0) ? 0.0 : (static_cast<double>(m.tp) / static_cast<double>(m.tp + m.fp));
    m.recall = (m.tp + m.fn == 0) ? 0.0 : (static_cast<double>(m.tp) / static_cast<double>(m.tp + m.fn));
    return m;
}

cv::Mat build_visual_overlay(
    const cv::Mat& rgb_u8,
    const cv::Mat& pred_mask_u8,
    const cv::Mat& ref_mask_u8
) {
    cv::Mat out = rgb_u8.clone();
    const int H = rgb_u8.rows;
    const int W = rgb_u8.cols;
    const bool has_ref = (!ref_mask_u8.empty() && ref_mask_u8.rows == H && ref_mask_u8.cols == W);

    for (int r = 0; r < H; ++r) {
        cv::Vec3b* o_ptr = out.ptr<cv::Vec3b>(r);
        const uchar* p_ptr = pred_mask_u8.ptr<uchar>(r);
        const uchar* g_ptr = has_ref ? ref_mask_u8.ptr<uchar>(r) : nullptr;
        for (int c = 0; c < W; ++c) {
            bool p = (p_ptr[c] != 0);
            if (has_ref) {
                bool g = (g_ptr[c] != 0);
                // TP: Green (46, 204, 113), FP: Orange (230, 126, 34), FN: Magenta (214, 51, 132)
                if (p && g) {
                    o_ptr[c] = cv::Vec3b(
                        static_cast<uchar>(0.55 * o_ptr[c][0] + 0.45 * 46),
                        static_cast<uchar>(0.55 * o_ptr[c][1] + 0.45 * 204),
                        static_cast<uchar>(0.55 * o_ptr[c][2] + 0.45 * 113)
                    );
                } else if (p && !g) {
                    o_ptr[c] = cv::Vec3b(
                        static_cast<uchar>(0.45 * o_ptr[c][0] + 0.55 * 230),
                        static_cast<uchar>(0.45 * o_ptr[c][1] + 0.55 * 126),
                        static_cast<uchar>(0.45 * o_ptr[c][2] + 0.55 * 34)
                    );
                } else if (!p && g) {
                    o_ptr[c] = cv::Vec3b(
                        static_cast<uchar>(0.45 * o_ptr[c][0] + 0.55 * 214),
                        static_cast<uchar>(0.45 * o_ptr[c][1] + 0.55 * 51),
                        static_cast<uchar>(0.45 * o_ptr[c][2] + 0.55 * 132)
                    );
                }
            } else if (p) {
                o_ptr[c] = cv::Vec3b(
                    static_cast<uchar>(0.60 * o_ptr[c][0] + 0.40 * 46),
                    static_cast<uchar>(0.60 * o_ptr[c][1] + 0.40 * 204),
                    static_cast<uchar>(0.60 * o_ptr[c][2] + 0.40 * 113)
                );
            }
        }
    }
    return out;
}

SvmPrediction predict_svm(
    const std::map<std::string, double>& features,
    const SvmArtifact& artifact
) {
    const size_t n_feat = artifact.feature_names.size();
    std::vector<double> x_scaled(n_feat, 0.0);
    for (size_t i = 0; i < n_feat; ++i) {
        auto it = features.find(artifact.feature_names[i]);
        if (it == features.end()) {
            throw std::runtime_error("Missing required feature: " + artifact.feature_names[i]);
        }
        double scale = artifact.scaler_scale[i];
        x_scaled[i] = (scale > 0.0) ? ((it->second - artifact.scaler_mean[i]) / scale) : 0.0;
    }

    const int n_classes = static_cast<int>(artifact.classes.size());
    const int n_pairs = n_classes * (n_classes - 1) / 2;
    std::vector<double> dec_ovo(n_pairs, 0.0);

    if (artifact.kernel == "linear" && !artifact.linear_ovo_weights.empty()) {
        for (int p = 0; p < n_pairs; ++p) {
            double dot = artifact.intercept[p];
            const auto& w = artifact.linear_ovo_weights[p];
            for (size_t f = 0; f < n_feat; ++f) {
                dot += x_scaled[f] * w[f];
            }
            dec_ovo[p] = dot;
        }
    } else {
        const size_t n_sv = artifact.support_vectors.size();
        std::vector<double> k_vals(n_sv, 0.0);
        if (artifact.kernel == "linear") {
            for (size_t s = 0; s < n_sv; ++s) {
                double dot = 0.0;
                const auto& sv = artifact.support_vectors[s];
                for (size_t f = 0; f < n_feat; ++f) {
                    dot += x_scaled[f] * sv[f];
                }
                k_vals[s] = dot;
            }
        } else if (artifact.kernel == "rbf") {
            for (size_t s = 0; s < n_sv; ++s) {
                double sq = 0.0;
                const auto& sv = artifact.support_vectors[s];
                for (size_t f = 0; f < n_feat; ++f) {
                    double d = x_scaled[f] - sv[f];
                    sq += d * d;
                }
                k_vals[s] = std::exp(-artifact.gamma_value * sq);
            }
        } else {
            throw std::runtime_error("Unsupported SVM kernel: " + artifact.kernel);
        }

        std::vector<int> starts(n_classes + 1, 0);
        for (int c = 0; c < n_classes; ++c) {
            starts[c + 1] = starts[c] + artifact.n_support[c];
        }

        int pair_idx = 0;
        for (int i = 0; i < n_classes; ++i) {
            const int si = starts[i];
            const int ei = starts[i + 1];
            for (int j = i + 1; j < n_classes; ++j) {
                const int sj = starts[j];
                const int ej = starts[j + 1];
                double val = artifact.intercept[pair_idx];
                const auto& dc_jm1 = artifact.dual_coef[j - 1];
                const auto& dc_i = artifact.dual_coef[i];
                for (int s = si; s < ei; ++s) {
                    val += k_vals[s] * dc_jm1[s];
                }
                for (int s = sj; s < ej; ++s) {
                    val += k_vals[s] * dc_i[s];
                }
                dec_ovo[pair_idx] = val;
                ++pair_idx;
            }
        }
    }

    std::vector<int> votes(n_classes, 0);
    int pair_idx = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            if (dec_ovo[pair_idx] > 0.0) {
                votes[i] += 1;
            } else {
                votes[j] += 1;
            }
            ++pair_idx;
        }
    }

    int best_idx = 0;
    for (int c = 1; c < n_classes; ++c) {
        if (votes[c] > votes[best_idx]) {
            best_idx = c;
        }
    }

    SvmPrediction pred;
    pred.predicted_index = best_idx;
    pred.predicted_category = artifact.classes[best_idx];
    pred.ovo_votes = std::move(votes);
    pred.ovo_decisions = std::move(dec_ovo);
    pred.standardized_features = std::move(x_scaled);
    return pred;
}

PipelineResult run_full_pipeline(
    const std::string& image_path,
    const std::string& reference_mask_path,
    const std::string& segmentation_method,
    const std::string& svm_artifact_path,
    const std::string& repo_root
) {
    PipelineResult res;
    res.rgb_u8 = load_rgb_image(image_path);
    res.ref_mask_u8 = load_reference_mask(reference_mask_path);

    auto seg = run_segmentation(res.rgb_u8, segmentation_method, repo_root);
    res.pred_mask_u8 = seg.first;
    res.otsu_threshold = seg.second;

    res.feature_names = canonical_feature_columns();
    res.features = extract_features(res.rgb_u8, res.pred_mask_u8);
    res.overlap = compute_overlap_metrics(res.pred_mask_u8, res.ref_mask_u8);
    res.overlay_rgb_u8 = build_visual_overlay(res.rgb_u8, res.pred_mask_u8, res.ref_mask_u8);

    SvmArtifact svm_art = load_svm_artifact(svm_artifact_path);
    res.classification = predict_svm(res.features, svm_art);
    return res;
}

}  // namespace ebhi
