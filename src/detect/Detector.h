// DET-02 모델 출력 해석
//
// 엔진(OpenCV DNN, ONNX Runtime)과 상관없이 같은 규칙으로 출력을 풀어 원본 프레임 좌표로 되돌림

#pragma once

#include "common/Types.h"
#include "preprocess/Letterbox.h"

#include <opencv2/core.hpp>

#include <vector>

// output: {1, 4 + 클래스 수, N} float32. 앞 4행 = cx, cy, w, h (모델 입력 픽셀), 나머지 행 = 클래스별 점수
// 후보마다 점수가 가장 높은 클래스를 고르고, 원본 프레임 안으로 자른 뒤 1픽셀 미만 박스는 버림
// 탐지 결과가 없으면 빈 vector를 반환함
std::vector<Detection> decodeDetections(const cv::Mat& output, const LetterboxInfo& info);

// DET-03 시작용 신뢰도 임계값. 최종값은 검증 영상 실험으로 정함
inline constexpr float kDefaultConfThreshold = 0.25f;

// DET-03. 임계값이 클래스 수만큼 있고 모두 0~1 범위인지 확인함
// false면 호출한 쪽에서 설정 오류(종료 코드 4)로 끝냄
bool isValidConfThresholds(const std::vector<float>& thresholds, int nc);

// DET-03. 점수가 자기 클래스의 임계값보다 낮은 후보를 버림 (같으면 남김)
// thresholds의 순서 = class_id
std::vector<Detection> filterByConfidence(const std::vector<Detection>& detections,
    const std::vector<float>& thresholds);
