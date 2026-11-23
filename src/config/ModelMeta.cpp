#include "config/ModelMeta.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <utility>

namespace {

// 필수 항목을 읽으면서 누락되거나 형식이 틀린 항목을 모아 둠
class MetaReader {
public:
    explicit MetaReader(const cv::FileStorage& fs) : fs_(fs) {}

    void readInt(const std::string& path, int& out)
    {
        const cv::FileNode node = find(path);
        if (node.isNone()) return;
        if (!node.isInt()) { markInvalid(path); return; }
        out = static_cast<int>(node);
    }

    void readNumber(const std::string& path, double& out)
    {
        const cv::FileNode node = find(path);
        if (node.isNone()) return;
        if (!node.isInt() && !node.isReal()) { markInvalid(path); return; }
        out = static_cast<double>(node);
    }

    void readString(const std::string& path, std::string& out)
    {
        const cv::FileNode node = find(path);
        if (node.isNone()) return;
        if (!node.isString() || static_cast<std::string>(node).empty()) { markInvalid(path); return; }
        out = static_cast<std::string>(node);
    }

    void readBool(const std::string& path, bool& out)
    {
        std::string text;
        readString(path, text);
        if (text.empty()) return;
        if (text != "true" && text != "false") { markInvalid(path); return; }
        out = text == "true";
    }

    void readIntList(const std::string& path, std::vector<int>& out)
    {
        const cv::FileNode node = find(path);
        if (node.isNone()) return;
        if (!node.isSeq() || node.empty()) { markInvalid(path); return; }
        out.clear();
        for (const auto& item : node) {
            if (!item.isInt()) { markInvalid(path); return; }
            out.push_back(static_cast<int>(item));
        }
    }

    void readStringList(const std::string& path, std::vector<std::string>& out)
    {
        const cv::FileNode node = find(path);
        if (node.isNone()) return;
        if (!node.isSeq() || node.empty()) { markInvalid(path); return; }
        out.clear();
        for (const auto& item : node) {
            if (!item.isString()) { markInvalid(path); return; }
            out.push_back(static_cast<std::string>(item));
        }
    }

    // 문제가 있으면 원인을 error에 적고 false를 반환함
    bool ok(std::string& error) const
    {
        if (!missing_.empty()) { error = "missing items: " + missing_; return false; }
        if (!invalid_.empty()) { error = "invalid value: " + invalid_; return false; }
        return true;
    }

private:
    // "preprocess.scale"처럼 점으로 구분한 경로의 값을 찾음
    cv::FileNode find(const std::string& path)
    {
        const auto dot = path.find('.');
        cv::FileNode node;
        if (dot == std::string::npos) {
            node = fs_[path];
        } else {
            const cv::FileNode parent = fs_[path.substr(0, dot)];
            if (parent.isMap()) node = parent[path.substr(dot + 1)];
        }
        if (node.isNone()) missing_ += (missing_.empty() ? "" : ", ") + path;
        return node;
    }

    void markInvalid(const std::string& path)
    {
        invalid_ += (invalid_.empty() ? "" : ", ") + path;
    }

    const cv::FileStorage& fs_;
    std::string missing_;
    std::string invalid_;
};

}  // namespace 끝

bool ModelMeta::load(const std::string& path, ModelMeta& out, std::string& error)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        error = "cannot open file: " + path;
        return false;
    }

    cv::FileStorage fs;
    try {
        if (!fs.open(path, cv::FileStorage::READ)) {
            error = "cannot open file: " + path;
            return false;
        }
    } catch (const cv::Exception&) {
        error = "invalid YAML format: " + path;
        return false;
    }

    // 1. 필수 항목 읽기
    ModelMeta meta;
    int width = 0;
    int height = 0;
    std::vector<int> input_shape;
    std::string resize, layout, dtype;

    MetaReader reader(fs);
    reader.readString("model_name", meta.model_name);
    reader.readInt("input.width", width);
    reader.readInt("input.height", height);
    reader.readIntList("input.shape", input_shape);
    reader.readIntList("output.shape", meta.output_shape);
    reader.readInt("nc", meta.nc);
    reader.readStringList("names", meta.names);
    reader.readString("preprocess.resize", resize);
    reader.readInt("preprocess.pad_value", meta.pad_value);
    reader.readNumber("preprocess.scale", meta.scale);
    reader.readBool("preprocess.swap_rb", meta.swap_rb);
    reader.readString("preprocess.layout", layout);
    reader.readString("preprocess.dtype", dtype);
    if (!reader.ok(error)) return false;

    // 2. 값끼리 서로 맞는지 검사
    if (width <= 0 || width != height) {
        error = "input size must be square: " + std::to_string(width) + "x" + std::to_string(height);
        return false;
    }
    if (input_shape != std::vector<int>{1, 3, height, width}) {
        error = "input.shape does not match input size " + std::to_string(width);
        return false;
    }
    // 학습 기록의 imgsz가 있으면 실제 입력 크기와 같아야 함
    const cv::FileNode train = fs["train"];
    if (train.isMap() && !train["imgsz"].isNone()) {
        const cv::FileNode train_imgsz = train["imgsz"];
        if (!train_imgsz.isInt() || static_cast<int>(train_imgsz) != width) {
            error = "train.imgsz does not match input size " + std::to_string(width);
            return false;
        }
    }
    if (meta.nc <= 0 || static_cast<int>(meta.names.size()) != meta.nc) {
        error = "names count does not match nc";
        return false;
    }
    if (meta.output_shape.size() != 3 || meta.output_shape[0] != 1
        || meta.output_shape[1] != 4 + meta.nc || meta.output_shape[2] <= 0) {
        error = "output.shape must be {1, 4 + nc, N}";
        return false;
    }
    if (meta.pad_value < 0 || meta.pad_value > 255) { error = "pad_value must be 0~255"; return false; }
    if (meta.scale <= 0.0) { error = "scale must be positive"; return false; }
    if (resize != "letterbox") { error = "unsupported resize: " + resize; return false; }
    if (layout != "NCHW") { error = "unsupported layout: " + layout; return false; }
    if (dtype != "float32") { error = "unsupported dtype: " + dtype; return false; }

    meta.imgsz = width;
    out = std::move(meta);
    return true;
}
