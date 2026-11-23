// DET-05 ONNX Runtime 엔진 (CUDA 1순위, CPU는 직접 지정할 때만 사용)
//
// CUDA를 지정했는데 쓸 수 없으면 CPU로 넘어가지 않고 load가 실패함 (종료 코드 3)
// CUDA에서는 Python 검증과 같은 정밀도를 내도록 TF32를 끔

#pragma once

#include "detect/Detector.h"
#include "detect/IDetector.h"

#include <onnxruntime_cxx_api.h>

#include <memory>

// detect.device 설정값
enum class OrtDevice { Cuda, Cpu };

class OrtDetector : public IDetector {
public:
    explicit OrtDetector(OrtDevice device = OrtDevice::Cuda);

    bool load(const std::string& model_path, const std::string& names_path,
        const ModelMeta& meta, std::string& error) override;

    // 추론이 실패하면 예외를 던짐. 호출한 쪽에서 그 프레임을 건너뜀 (SYS-05)
    std::vector<Detection> detect(const cv::Mat& blob, const LetterboxInfo& info) override;

    bool setConfThresholds(const std::vector<float>& thresholds, std::string& error) override;

    bool setNmsIou(float iou, std::string& error) override;

    const std::vector<std::string>& classNames() const override { return names_; }

private:
    // blob을 추론해 {1, 4 + 클래스 수, N} 출력을 받음
    Ort::Value run(const cv::Mat& blob);

    OrtDevice device_;
    Ort::Env env_;                          // session_보다 먼저 만들고 나중에 해제되어야 함
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    std::vector<std::string> names_;
    std::vector<float> conf_thresholds_;    // 클래스별 신뢰도 임계값 (순서 = class_id)
    float nms_iou_ = kDefaultNmsIou;        // NMS IoU 값
};
