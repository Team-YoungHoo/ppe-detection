// DET-02~04 모델 출력 후처리
//
// 엔진(OpenCV DNN, ONNX Runtime)과 상관없이 같은 순서로 처리함
// 출력 해석 -> 신뢰도 필터(DET-03) -> NMS(DET-04) -> 원본 프레임 좌표 복원

#pragma once

#include "common/Types.h"
#include "preprocess/Letterbox.h"

#include <opencv2/core.hpp>

#include <vector>

// DET-02. output: {1, 4 + 클래스 수, N} float32. 앞 4행 = cx, cy, w, h (모델 입력 픽셀), 나머지 행 = 클래스별 점수
// 후보마다 점수가 가장 높은 클래스를 고름. 반환하는 box는 아직 모델 입력 좌표임
std::vector<Detection> decodeCandidates(const cv::Mat& output);

// DET-03 시작용 신뢰도 임계값. 최종값은 검증 영상 실험으로 정함
inline constexpr float kDefaultConfThreshold = 0.25f;

// DET-03. 임계값이 클래스 수만큼 있고 모두 0~1 범위인지 확인함
// false면 호출한 쪽에서 설정 오류(종료 코드 4)로 끝냄
bool isValidConfThresholds(const std::vector<float>& thresholds, int nc);

// DET-03. 점수가 자기 클래스의 임계값보다 낮은 후보를 버림 (같으면 남김)
// thresholds의 순서 = class_id
std::vector<Detection> filterByConfidence(const std::vector<Detection>& detections,
    const std::vector<float>& thresholds);

// DET-04 시작용 NMS IoU 값. 최종값은 실험으로 정함
inline constexpr float kDefaultNmsIou = 0.45f;

// DET-04. IoU 값이 0~1 범위인지 확인함
bool isValidNmsIou(float iou);

// DET-04. 같은 물체에 겹친 박스를 클래스별로 하나로 줄임
// 클래스끼리는 서로 지우지 않으므로 같은 머리에 helmet과 no-helmet이 둘 다 남을 수 있음
std::vector<Detection> applyNms(const std::vector<Detection>& detections, float iou);

// DET-02. 모델 입력 좌표 -> 원본 프레임 좌표
// [0, 원본너비-1], [0, 원본높이-1]로 자른 뒤 너비나 높이가 1픽셀 미만인 박스는 버림
std::vector<Detection> restoreToOriginal(const std::vector<Detection>& detections,
    const LetterboxInfo& info);
