#include "detect/OpenCvDnnDetector.h"

#include "detect/Detector.h"

#include <opencv2/core/utility.hpp>

#include <filesystem>
#include <iostream>
#include <utility>

namespace {

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

    // 2. 클래스 이름 확인
    std::vector<std::string> names;
    if (!loadClassNames(names_path, meta, names, error)) return false;

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
    conf_thresholds_.assign(names_.size(), kDefaultConfThreshold);
    return true;
}

std::vector<Detection> OpenCvDnnDetector::detect(const cv::Mat& blob, const LetterboxInfo& info)
{
    if (net_.empty() || blob.empty()) return {};

    net_.setInput(blob);

    // 명세 순서: 출력 해석 -> 신뢰도 필터 -> NMS -> 원본 좌표 복원
    const auto candidates = decodeCandidates(net_.forward());
    const auto confident = filterByConfidence(candidates, conf_thresholds_);
    return restoreToOriginal(applyNms(confident, nms_iou_), info);
}

bool OpenCvDnnDetector::setConfThresholds(const std::vector<float>& thresholds, std::string& error)
{
    if (!isValidConfThresholds(thresholds, static_cast<int>(names_.size()))) {
        error = "Invalid confidence thresholds (need one 0~1 value per class)";
        return false;
    }
    conf_thresholds_ = thresholds;
    return true;
}

bool OpenCvDnnDetector::setNmsIou(float iou, std::string& error)
{
    if (!isValidNmsIou(iou)) {
        error = "Invalid NMS IoU (need a 0~1 value)";
        return false;
    }
    nms_iou_ = iou;
    return true;
}
