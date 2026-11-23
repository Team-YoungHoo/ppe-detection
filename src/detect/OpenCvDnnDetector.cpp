#include "detect/OpenCvDnnDetector.h"

#include <opencv2/core/utility.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

namespace {

// classes.txt를 한 줄씩 읽음. Windows 줄바꿈과 파일 끝의 빈 줄은 무시함
bool readClassNames(const std::string& path, std::vector<std::string>& names)
{
    std::ifstream file(path);
    if (!file.is_open()) return false;

    names.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        names.push_back(line);
    }
    while (!names.empty() && names.back().empty()) names.pop_back();
    return true;
}

// Mat의 형태를 "1x9x8400"처럼 문자열로 바꿈
std::string shapeText(const cv::Mat& mat)
{
    std::string text;
    for (int i = 0; i < mat.dims; ++i) {
        if (i > 0) text += "x";
        text += std::to_string(mat.size[i]);
    }
    return text.empty() ? "empty" : text;
}

}  // namespace 끝

bool OpenCvDnnDetector::load(const std::string& model_path, const std::string& names_path,
    const ModelMeta& meta, std::string& error)
{
    // 1. 파일 존재 확인
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model_path, ec)) {
        error = "Cannot load model: " + model_path;
        return false;
    }

    // 2. 클래스 이름 확인. 순서가 바뀌면 class_id의 뜻이 달라지므로 순서까지 비교함
    std::vector<std::string> names;
    if (!readClassNames(names_path, names)) {
        error = "Cannot load classes: " + names_path;
        return false;
    }
    if (static_cast<int>(names.size()) != meta.nc) {
        error = "classes count mismatch (expected: " + std::to_string(meta.nc)
            + ", actual: " + std::to_string(names.size()) + ")";
        return false;
    }
    if (names != meta.names) {
        error = "classes order differs from model.yaml: " + names_path;
        return false;
    }

    // 3. 모델 로드
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception&) {
        error = "Cannot load model: " + model_path;
        return false;
    }
    if (net.empty()) {
        error = "Cannot load model: " + model_path;
        return false;
    }

    // 4. 더미 이미지로 1회 추론(워밍업)하면서 실제 출력 형태를 얻음
    const int input_size[] = { 1, 3, meta.imgsz, meta.imgsz };
    const cv::Mat dummy(4, input_size, CV_32F, cv::Scalar(0));
    cv::Mat output;
    cv::TickMeter timer;
    try {
        timer.start();
        net.setInput(dummy);
        output = net.forward();
        timer.stop();
    } catch (const cv::Exception&) {
        error = "Cannot run model: " + model_path;
        return false;
    }

    // 5. 출력 형태 확인: {1, 4 + 클래스 수, N}
    const std::string expected = "1x" + std::to_string(4 + meta.nc) + "x"
        + std::to_string(meta.output_shape.at(2));
    const std::string actual = shapeText(output);
    if (actual != expected) {
        error = "Unexpected model output (expected: " + expected + ", actual: " + actual + ")";
        return false;
    }

    // Log.h(SYS-04)가 준비되면 교체
    std::cout << "[INFO] Model loaded: " << model_path << "\n"
        << "[INFO] Input shape: " << shapeText(dummy) << "\n"
        << "[INFO] Output shape: " << actual << "\n"
        << "[INFO] Classes: " << names.size() << "\n"
        << "[INFO] Warm-up: " << timer.getTimeMilli() << " ms\n";

    net_ = std::move(net);
    names_ = std::move(names);
    return true;
}
