#pragma once

#include <array>
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ebhi {

struct OtsuArtifact {
    std::string foreground_polarity = "dark";
};

struct KMeansArtifact {
    int n_clusters = 0;
    std::vector<std::array<double, 3>> cluster_centers;
    std::vector<int> foreground_clusters;
};

struct SvmArtifact {
    std::string model_id;
    std::string segmentation_method;
    std::string feature_set;
    std::vector<std::string> feature_names;
    std::vector<std::string> classes;
    std::string kernel;
    double C = 1.0;
    double gamma_value = 1.0;
    std::vector<double> scaler_mean;
    std::vector<double> scaler_scale;
    std::vector<int> n_support;
    std::vector<std::vector<double>> support_vectors;
    std::vector<std::vector<double>> dual_coef;
    std::vector<double> intercept;
    std::vector<std::vector<double>> linear_ovo_weights;
};

struct OverlapMetrics {
    bool available = false;
    double dice = 0.0;
    double iou = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double pred_fg_fraction = 0.0;
    double ref_fg_fraction = 0.0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    int tn = 0;
};

struct SvmPrediction {
    std::string predicted_category;
    int predicted_index = -1;
    std::vector<int> ovo_votes;
    std::vector<double> ovo_decisions;
    std::vector<double> standardized_features;
};

struct PipelineResult {
    cv::Mat rgb_u8;
    cv::Mat pred_mask_u8;
    cv::Mat ref_mask_u8;
    cv::Mat overlay_rgb_u8;
    double otsu_threshold = -1.0;
    std::vector<std::string> feature_names;
    std::map<std::string, double> features;
    OverlapMetrics overlap;
    SvmPrediction classification;
};

const std::vector<std::string>& canonical_feature_columns();

OtsuArtifact load_otsu_artifact(const std::string& json_path);
KMeansArtifact load_kmeans_artifact(const std::string& json_path);
SvmArtifact load_svm_artifact(const std::string& json_path);

cv::Mat load_rgb_image(const std::string& image_path);
cv::Mat load_reference_mask(const std::string& mask_path);

cv::Mat rgb_to_gray(const cv::Mat& rgb_u8);
cv::Mat rgb_to_lab(const cv::Mat& rgb_u8);

std::pair<cv::Mat, double> otsu_threshold_mask(
    const cv::Mat& gray_u8,
    const std::string& foreground_polarity = "dark"
);

cv::Mat kmeans_raw_mask(
    const cv::Mat& rgb_u8,
    const KMeansArtifact& artifact
);

std::map<std::string, cv::Mat> morphology_variants(
    const cv::Mat& raw_mask_u8,
    int min_area = 16
);

std::pair<cv::Mat, double> run_segmentation(
    const cv::Mat& rgb_u8,
    const std::string& method_key,
    const std::string& repo_root
);

std::map<std::string, double> extract_features(
    const cv::Mat& rgb_u8,
    const cv::Mat& mask_u8
);

OverlapMetrics compute_overlap_metrics(
    const cv::Mat& pred_mask_u8,
    const cv::Mat& ref_mask_u8
);

cv::Mat build_visual_overlay(
    const cv::Mat& rgb_u8,
    const cv::Mat& pred_mask_u8,
    const cv::Mat& ref_mask_u8 = cv::Mat()
);

SvmPrediction predict_svm(
    const std::map<std::string, double>& features,
    const SvmArtifact& artifact
);

PipelineResult run_full_pipeline(
    const std::string& image_path,
    const std::string& reference_mask_path,
    const std::string& segmentation_method,
    const std::string& svm_artifact_path,
    const std::string& repo_root
);

}  // namespace ebhi
