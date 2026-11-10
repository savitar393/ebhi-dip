#include "ebhi_pipeline.hpp"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_usage() {
    std::cout
        << "Usage: ebhi_cli --image <path> [options]\n"
        << "Options:\n"
        << "  --image <path>        Input RGB histopathology image (.png)\n"
        << "  --mask <path>         Optional reference ground-truth mask (.png)\n"
        << "  --seg <method>        Segmentation method (default: k4_M0)\n"
        << "  --svm <path>          SVM JSON artifact path (default: <repo_root>/models/svm_k4_M0_selected.json)\n"
        << "  --repo-root <path>    Repository root path (default: current directory)\n"
        << "  --output-json <path>  Optional path to write JSON output\n"
        << "  --save-mask <path>    Optional path to write predicted binary mask (.png)\n"
        << "  --save-overlay <path> Optional path to write visual overlay (.png)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string image_path;
    std::string mask_path;
    std::string seg_method = "k4_M0";
    std::string svm_path;
    std::string repo_root = ".";
    std::string output_json_path;
    std::string save_mask_path;
    std::string save_overlay_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--mask" && i + 1 < argc) {
            mask_path = argv[++i];
        } else if (arg == "--seg" && i + 1 < argc) {
            seg_method = argv[++i];
        } else if (arg == "--svm" && i + 1 < argc) {
            svm_path = argv[++i];
        } else if (arg == "--repo-root" && i + 1 < argc) {
            repo_root = argv[++i];
        } else if (arg == "--output-json" && i + 1 < argc) {
            output_json_path = argv[++i];
        } else if (arg == "--save-mask" && i + 1 < argc) {
            save_mask_path = argv[++i];
        } else if (arg == "--save-overlay" && i + 1 < argc) {
            save_overlay_path = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (image_path.empty()) {
        std::cerr << "Error: --image is required.\n";
        print_usage();
        return 1;
    }

    if (svm_path.empty()) {
        svm_path = (fs::path(repo_root) / "models" / "svm_k4_M0_selected.json").string();
    }

    try {
        ebhi::SvmArtifact svm_art = ebhi::load_svm_artifact(svm_path);
        ebhi::PipelineResult res = ebhi::run_full_pipeline(
            image_path,
            mask_path,
            seg_method,
            svm_path,
            repo_root
        );

        if (!save_mask_path.empty()) {
            cv::imwrite(save_mask_path, res.pred_mask_u8);
        }
        if (!save_overlay_path.empty()) {
            cv::Mat bgr_overlay;
            cv::cvtColor(res.overlay_rgb_u8, bgr_overlay, cv::COLOR_RGB2BGR);
            cv::imwrite(save_overlay_path, bgr_overlay);
        }

        QJsonObject root;
        root["image_path"] = QString::fromStdString(image_path);
        root["segmentation_method"] = QString::fromStdString(seg_method);
        root["svm_model_id"] = QString::fromStdString(svm_art.model_id);
        root["otsu_threshold"] = res.otsu_threshold;
        root["predicted_category"] = QString::fromStdString(res.classification.predicted_category);
        root["predicted_index"] = res.classification.predicted_index;

        QJsonArray votes_arr;
        for (int v : res.classification.ovo_votes) {
            votes_arr.append(v);
        }
        root["ovo_votes"] = votes_arr;

        QJsonArray dec_arr;
        for (double d : res.classification.ovo_decisions) {
            dec_arr.append(d);
        }
        root["ovo_decisions"] = dec_arr;

        QJsonObject ov_obj;
        ov_obj["available"] = res.overlap.available;
        ov_obj["dice"] = res.overlap.dice;
        ov_obj["iou"] = res.overlap.iou;
        ov_obj["precision"] = res.overlap.precision;
        ov_obj["recall"] = res.overlap.recall;
        ov_obj["pred_fg_fraction"] = res.overlap.pred_fg_fraction;
        ov_obj["ref_fg_fraction"] = res.overlap.ref_fg_fraction;
        ov_obj["tp"] = res.overlap.tp;
        ov_obj["fp"] = res.overlap.fp;
        ov_obj["fn"] = res.overlap.fn;
        ov_obj["tn"] = res.overlap.tn;
        root["overlap"] = ov_obj;

        QJsonObject feat_obj;
        for (const auto& name : res.feature_names) {
            feat_obj[QString::fromStdString(name)] = res.features.at(name);
        }
        root["features"] = feat_obj;

        QJsonDocument doc(root);
        QByteArray json_bytes = doc.toJson(QJsonDocument::Indented);

        if (!output_json_path.empty()) {
            QFile out_file(QString::fromStdString(output_json_path));
            if (!out_file.open(QIODevice::WriteOnly)) {
                std::cerr << "Cannot open output JSON path: " << output_json_path << "\n";
                return 1;
            }
            out_file.write(json_bytes);
        } else {
            std::cout << json_bytes.toStdString();
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Pipeline execution error: " << ex.what() << "\n";
        return 2;
    }
}
