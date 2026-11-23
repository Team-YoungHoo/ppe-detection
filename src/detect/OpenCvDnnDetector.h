// DET-01 OpenCV DNN 엔진으로 ONNX 모델 로드

#pragma once

#include "detect/Detector.h"
#include "detect/IDetector.h"

#include <opencv2/dnn.hpp>

class OpenCvDnnDetector : public IDetector {
public:
    bool load(const std::string& model_path, const std::string& names_path,
        const ModelMeta& meta, std::string& error) override;

    std::vector<Detection> detect(const cv::Mat& blob, const LetterboxInfo& info) override;

    bool setConfThresholds(const std::vector<float>& thresholds, std::string& error) override;

    bool setNmsIou(float iou, std::string& error) override;

    const std::vector<std::string>& classNames() const override { return names_; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> names_;
    std::vector<float> conf_thresholds_;  // 클래스별 신뢰도 임계값 (순서 = class_id)
    float nms_iou_ = kDefaultNmsIou;      // NMS IoU 값
};
