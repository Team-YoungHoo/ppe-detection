// DET-01 추론 엔진 공통 인터페이스
//
// 엔진(OpenCV DNN, ONNX Runtime)을 바꿔도 다른 모듈이 영향받지 않도록 이 틀 뒤에 숨김

#pragma once

#include "common/Types.h"
#include "config/ModelMeta.h"
#include "preprocess/Letterbox.h"

#include <string>
#include <vector>

class IDetector {
public:
    virtual ~IDetector() = default;

    // 모델과 클래스 이름을 불러와 추론 준비를 함. 시작할 때 한 번만 호출함
    // 실패하면 false와 이유를 반환하고 호출한 쪽에서 종료 코드 3으로 끝냄
    virtual bool load(const std::string& model_path, const std::string& names_path,
        const ModelMeta& meta, std::string& error) = 0;

    // DET-02. blob을 추론해 원본 프레임 좌표의 탐지 결과를 반환함
    virtual std::vector<Detection> detect(const cv::Mat& blob, const LetterboxInfo& info) = 0;

    // DET-03. 클래스별 신뢰도 임계값을 바꿈. 잘못된 값이면 false를 반환하고 기존 값을 유지함
    virtual bool setConfThresholds(const std::vector<float>& thresholds, std::string& error) = 0;

    // 클래스 이름 (순서 = class_id)
    virtual const std::vector<std::string>& classNames() const = 0;
};
