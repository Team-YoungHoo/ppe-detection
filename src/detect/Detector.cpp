#include "detect/Detector.h"

#include <algorithm>
#include <iostream>

std::vector<Detection> decodeDetections(const cv::Mat& output, const LetterboxInfo& info)
{
    std::vector<Detection> detections;
    if (output.empty()) return detections;

    if (output.dims != 3 || output.size[0] != 1 || output.size[1] <= 4
        || output.type() != CV_32F || !output.isContinuous()) {
        // Log.h(SYS-04)가 준비되면 교체
        std::cerr << "[WARN] Unexpected output format, frame skipped\n";
        return detections;
    }

    const int channels = output.size[1];
    const int count = output.size[2];
    const float* data = output.ptr<float>();

    // 원본 프레임의 마지막 픽셀 좌표
    const float max_x = static_cast<float>(info.orig_w - 1);
    const float max_y = static_cast<float>(info.orig_h - 1);

    detections.reserve(count);
    for (int i = 0; i < count; ++i) {
        // {1, C, N} 배열에서 c행 i번째 값의 위치는 c * N + i
        const float cx = data[0 * count + i];
        const float cy = data[1 * count + i];
        const float w = data[2 * count + i];
        const float h = data[3 * count + i];

        // 1. 점수가 가장 높은 클래스 선택
        int class_id = 0;
        float score = data[4 * count + i];
        for (int c = 1; c < channels - 4; ++c) {
            const float value = data[(4 + c) * count + i];
            if (value > score) {
                score = value;
                class_id = c;
            }
        }

        // 2. 모델 입력 좌표 -> 원본 프레임 좌표
        const cv::Point2f p1 = toOriginal({ cx - w / 2.0f, cy - h / 2.0f }, info);
        const cv::Point2f p2 = toOriginal({ cx + w / 2.0f, cy + h / 2.0f }, info);

        // 3. 원본 프레임 범위로 자름
        const float x1 = std::clamp(p1.x, 0.0f, max_x);
        const float y1 = std::clamp(p1.y, 0.0f, max_y);
        const float x2 = std::clamp(p2.x, 0.0f, max_x);
        const float y2 = std::clamp(p2.y, 0.0f, max_y);

        // 4. 잘린 뒤 너비나 높이가 1픽셀 미만이면 버림
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;

        detections.push_back({ class_id, score, cv::Rect2f(x1, y1, x2 - x1, y2 - y1) });
    }
    return detections;
}

bool isValidConfThresholds(const std::vector<float>& thresholds, int nc)
{
    if (nc <= 0 || static_cast<int>(thresholds.size()) != nc) return false;

    // NaN은 어떤 비교도 거짓이므로 이 조건에서 함께 걸러짐
    return std::all_of(thresholds.begin(), thresholds.end(),
        [](float value) { return value >= 0.0f && value <= 1.0f; });
}

std::vector<Detection> filterByConfidence(const std::vector<Detection>& detections,
    const std::vector<float>& thresholds)
{
    std::vector<Detection> kept;
    kept.reserve(detections.size());
    for (const Detection& d : detections) {
        if (d.class_id < 0 || d.class_id >= static_cast<int>(thresholds.size())) continue;
        if (d.score >= thresholds[d.class_id]) kept.push_back(d);
    }
    return kept;
}
