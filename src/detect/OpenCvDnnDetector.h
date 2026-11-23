// DET-01 OpenCV DNN 엔진으로 ONNX 모델 로드

#pragma once

#include "detect/IDetector.h"

#include <opencv2/dnn.hpp>

class OpenCvDnnDetector : public IDetector {
public:
    bool load(const std::string& model_path, const std::string& names_path,
        const ModelMeta& meta, std::string& error) override;

    const std::vector<std::string>& classNames() const override { return names_; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> names_;
};
