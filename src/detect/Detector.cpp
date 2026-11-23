#include "detect/Detector.h"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <iostream>

std::vector<Detection> decodeCandidates(const cv::Mat& output)
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

    detections.reserve(count);
    for (int i = 0; i < count; ++i) {
        // {1, C, N} 배열에서 c행 i번째 값의 위치는 c * N + i
        const float cx = data[0 * count + i];
        const float cy = data[1 * count + i];
        const float w = data[2 * count + i];
        const float h = data[3 * count + i];

        // 점수가 가장 높은 클래스 선택
        int class_id = 0;
        float score = data[4 * count + i];
        for (int c = 1; c < channels - 4; ++c) {
            const float value = data[(4 + c) * count + i];
            if (value > score) {
                score = value;
                class_id = c;
            }
        }

        detections.push_back({ class_id, score, cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h) });
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

bool isValidNmsIou(float iou)
{
    return iou >= 0.0f && iou <= 1.0f;
}

std::vector<Detection> applyNms(const std::vector<Detection>& detections, float iou)
{
    std::vector<Detection> kept;
    if (detections.empty()) return kept;

    // 1. 클래스별로 나눔 (사람 박스가 안전모 박스를 지우지 않게 함)
    int max_class = 0;
    for (const Detection& d : detections) max_class = std::max(max_class, d.class_id);
    std::vector<std::vector<int>> by_class(max_class + 1);
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (detections[i].class_id >= 0) by_class[detections[i].class_id].push_back(i);
    }

    // 2. 클래스마다 NMS 수행. 신뢰도 필터는 이미 끝났으므로 score 기준은 0으로 둠
    for (const std::vector<int>& members : by_class) {
        if (members.empty()) continue;

        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        boxes.reserve(members.size());
        scores.reserve(members.size());
        for (int index : members) {
            boxes.emplace_back(detections[index].box);
            scores.push_back(detections[index].score);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.0f, iou, indices);

        // 3. 살아남은 박스를 합침
        for (int index : indices) kept.push_back(detections[members[index]]);
    }
    return kept;
}

std::vector<Detection> restoreToOriginal(const std::vector<Detection>& detections,
    const LetterboxInfo& info)
{
    // 원본 프레임의 마지막 픽셀 좌표
    const float max_x = static_cast<float>(info.orig_w - 1);
    const float max_y = static_cast<float>(info.orig_h - 1);

    std::vector<Detection> restored;
    restored.reserve(detections.size());
    for (const Detection& d : detections) {
        const cv::Point2f p1 = toOriginal({ d.box.x, d.box.y }, info);
        const cv::Point2f p2 = toOriginal({ d.box.x + d.box.width, d.box.y + d.box.height }, info);

        const float x1 = std::clamp(p1.x, 0.0f, max_x);
        const float y1 = std::clamp(p1.y, 0.0f, max_y);
        const float x2 = std::clamp(p2.x, 0.0f, max_x);
        const float y2 = std::clamp(p2.y, 0.0f, max_y);

        // 잘린 뒤 너비나 높이가 1픽셀 미만이면 버림
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;

        restored.push_back({ d.class_id, d.score, cv::Rect2f(x1, y1, x2 - x1, y2 - y1) });
    }
    return restored;
}
