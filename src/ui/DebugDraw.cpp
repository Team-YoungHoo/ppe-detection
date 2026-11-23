#include "ui/DebugDraw.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace {

const int kBoxThickness = 2;
const int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
const double kFontScale = 0.5;
const int kFontThickness = 1;
const int kLabelPadding = 3;   // 글자 주변 여백

// 글자 배경을 박스 위에 놓고, 위쪽 공간이 없으면 박스 안쪽으로 내림
void drawLabel(cv::Mat& image, const std::string& text, const cv::Point& box_top_left,
    const cv::Scalar& color)
{
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, kFontFace, kFontScale, kFontThickness, &baseline);
    const int label_w = std::min(text_size.width + kLabelPadding * 2, image.cols);
    const int label_h = std::min(text_size.height + baseline + kLabelPadding * 2, image.rows);

    const int x = std::clamp(box_top_left.x, 0, image.cols - label_w);
    int y = box_top_left.y - label_h;
    if (y < 0) y = std::min(box_top_left.y, image.rows - label_h);

    cv::rectangle(image, cv::Rect(x, y, label_w, label_h), color, cv::FILLED);
    cv::putText(image, text, cv::Point(x + kLabelPadding, y + kLabelPadding + text_size.height),
        kFontFace, kFontScale, cv::Scalar(255, 255, 255), kFontThickness, cv::LINE_AA);
}

}  // namespace 끝

bool debugColor(int class_id, cv::Scalar& color)
{
    static const cv::Scalar kColors[] = {
        cv::Scalar(128, 128, 128),  // person 회색
        cv::Scalar(255, 0, 0),      // helmet 파랑
        cv::Scalar(0, 165, 255),    // vest 주황
        cv::Scalar(255, 0, 255),    // no-helmet 자홍
        cv::Scalar(128, 0, 128),    // no-vest 보라
    };
    if (class_id < 0 || class_id >= static_cast<int>(std::size(kColors))) return false;
    color = kColors[class_id];
    return true;
}

std::string detectionLabel(const std::string& class_name, float score)
{
    char text[16];
    std::snprintf(text, sizeof(text), "%.2f", score);
    return class_name + " " + text;
}

cv::Mat drawDetections(const cv::Mat& frame, const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names)
{
    cv::Mat view = frame.clone();
    if (view.empty()) return view;

    const int max_x = view.cols - 1;
    const int max_y = view.rows - 1;

    for (const Detection& d : detections) {
        cv::Scalar color;
        if (!debugColor(d.class_id, color) || d.class_id >= static_cast<int>(class_names.size())) continue;

        const float right = d.box.x + d.box.width;
        const float bottom = d.box.y + d.box.height;

        // 완전히 화면 밖인 박스는 그리지 않음
        if (right < 0.0f || bottom < 0.0f || d.box.x > max_x || d.box.y > max_y) continue;

        // 화면 안으로 잘라서 그림
        const cv::Point p1(std::clamp(static_cast<int>(std::lround(d.box.x)), 0, max_x),
            std::clamp(static_cast<int>(std::lround(d.box.y)), 0, max_y));
        const cv::Point p2(std::clamp(static_cast<int>(std::lround(right)), 0, max_x),
            std::clamp(static_cast<int>(std::lround(bottom)), 0, max_y));

        cv::rectangle(view, p1, p2, color, kBoxThickness);
        drawLabel(view, detectionLabel(class_names[d.class_id], d.score), p1, color);
    }
    return view;
}
