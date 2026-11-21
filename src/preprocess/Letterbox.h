// VIN-02 Letterbox / VIN-03 blob 변환
//
// letterbox() : 어떤 크기의 프레임이든 비율을 유지해 S x S로 맞추고 남는 곳은 회색(pad_value)으로 채움
//               Ultralytics 학습·변환과 같은 규칙 (models/model.yaml의 preprocess)
// toBlob()    : S x S BGR 이미지 -> {1, 3, S, S} float32 RGB 텐서 (값 범위 0~1)
// toOriginal(): 모델 입력 좌표 -> 원본 프레임 좌표

#pragma once

#include <opencv2/core.hpp>

// 프레임을 어떻게 변환했는지 기록함. 탐지 결과를 원본 좌표로 되돌릴 때 필요함
struct LetterboxInfo {
    float scale = 1.0f;      // 계산한 배율: min(S / 세로, S / 가로)
    float scale_x = 1.0f;    // 반올림 후 실제 가로 배율: resized.width / 가로
    float scale_y = 1.0f;    // 반올림 후 실제 세로 배율: resized.height / 세로
    int pad_x = 0;           // 왼쪽에 넣은 회색 픽셀 수
    int pad_y = 0;           // 위쪽에 넣은 회색 픽셀 수
    cv::Size src_size;       // 원본 프레임 크기 (가로, 세로)
    cv::Size resized;        // S x S 안에서 패딩을 뺀 실제 이미지 크기
};

struct LetterboxResult {
    cv::Mat image;           // S x S BGR 이미지 (입력이 비어 있으면 빈 이미지)
    LetterboxInfo info;
};

// VIN-02. size = 모델 입력 크기 S (model.yaml의 input.width), pad_value = 114
// 빈 프레임이 들어오면 예외 없이 경고만 출력하고 빈 이미지를 반환함
// 호출한 쪽에서 그 프레임을 건너뛰면 됨
LetterboxResult letterbox(const cv::Mat& src, int size, int pad_value = 114);

// VIN-03. scale = 1/255, swap_rb = true (BGR -> RGB), NCHW 배치, float32
cv::Mat toBlob(const cv::Mat& letterboxed, double scale = 1.0 / 255.0, bool swap_rb = true);

// 점이나 박스를 모델 입력 좌표에서 원본 프레임 좌표로 되돌림
// 박스가 프레임 밖으로 나가면 프레임 안으로 잘라 냄
cv::Point2f toOriginal(const cv::Point2f& p, const LetterboxInfo& info);
cv::Rect2f toOriginal(const cv::Rect2f& box, const LetterboxInfo& info);
