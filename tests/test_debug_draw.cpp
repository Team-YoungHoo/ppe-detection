// DET-06 단위 테스트 (src/ui/DebugDraw)
//
// 사용법:
//   test_debug_draw
//       코드로 만든 프레임으로 전체 검사 (모델 파일 불필요)
//   test_debug_draw --model <best.onnx> <classes.txt> <model.yaml> <이미지> <저장할 이미지>
//       실제 모델의 탐지 결과를 그려 파일로 저장 (눈으로 확인용)

#include "detect/OpenCvDnnDetector.h"
#include "ui/DebugDraw.h"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

const std::vector<std::string> kNames = { "person", "helmet", "vest", "no-helmet", "no-vest" };
const cv::Vec3b kBackground(30, 30, 30);

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

cv::Mat blankFrame()
{
    return cv::Mat(240, 320, CV_8UC3, cv::Scalar(kBackground));
}

bool sameImage(const cv::Mat& a, const cv::Mat& b)
{
    return a.size() == b.size() && a.type() == b.type() && cv::norm(a, b, cv::NORM_INF) == 0.0;
}

cv::Vec3b toPixel(const cv::Scalar& color)
{
    return cv::Vec3b(static_cast<uchar>(color[0]), static_cast<uchar>(color[1]), static_cast<uchar>(color[2]));
}

void testLabelsAndColors()
{
    check(detectionLabel("person", 0.872565f) == "person 0.87", "label uses 2 decimal places");
    check(detectionLabel("no-helmet", 0.5f) == "no-helmet 0.50", "label keeps trailing zero");

    const std::vector<cv::Scalar> expected = {
        cv::Scalar(128, 128, 128), cv::Scalar(255, 0, 0), cv::Scalar(0, 165, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(128, 0, 128),
    };
    bool colors_ok = true;
    for (int c = 0; c < 5; ++c) {
        cv::Scalar color;
        colors_ok = colors_ok && debugColor(c, color) && color == expected[c];
        // UI-02의 착용/미착용 색(초록·빨강)과 겹치면 안 됨
        colors_ok = colors_ok && color != cv::Scalar(0, 255, 0) && color != cv::Scalar(0, 0, 255);
    }
    check(colors_ok, "5 class colors follow the spec and avoid UI-02 colors");

    cv::Scalar color;
    check(!debugColor(-1, color) && !debugColor(5, color), "unknown class has no color");
}

void testDrawing()
{
    const cv::Mat frame = blankFrame();
    const cv::Mat before = frame.clone();

    for (int c = 0; c < 5; ++c) {
        const Detection d{ c, 0.9f, cv::Rect2f(60, 80, 100, 100) };
        const cv::Mat view = drawDetections(frame, { d }, kNames);
        cv::Scalar color;
        debugColor(c, color);
        check(view.at<cv::Vec3b>(180, 110) == toPixel(color), kNames[c] + " box uses its color");
    }

    const cv::Mat view = drawDetections(frame, { { 0, 0.9f, cv::Rect2f(60, 80, 100, 100) } }, kNames);
    check(sameImage(frame, before), "input frame is not changed");
    check(view.size() == frame.size() && view.type() == frame.type(), "output keeps size and type");
    check(view.at<cv::Vec3b>(130, 110) == kBackground, "box inside is not filled");

    const cv::Mat above = view(cv::Rect(60, 55, 100, 24));
    check(cv::countNonZero(cv::Mat(above != cv::Scalar(kBackground)).reshape(1)) > 0, "label is drawn above the box");

    check(sameImage(drawDetections(frame, {}, kNames), frame), "no detections leaves frame as is");
    check(drawDetections(cv::Mat(), { { 0, 0.9f, cv::Rect2f(1, 1, 5, 5) } }, kNames).empty(), "empty frame is safe");
}

void testClipping()
{
    const cv::Mat frame = blankFrame();
    cv::Scalar person;
    debugColor(0, person);

    const cv::Mat clipped = drawDetections(frame, { { 0, 0.9f, cv::Rect2f(200, 120, 500, 500) } }, kNames);
    check(clipped.at<cv::Vec3b>(180, 319) == toPixel(person), "box over right edge is clipped to the frame");
    check(clipped.at<cv::Vec3b>(239, 260) == toPixel(person), "box over bottom edge is clipped to the frame");

    const cv::Mat left_top = drawDetections(frame, { { 0, 0.9f, cv::Rect2f(-50, -50, 150, 150) } }, kNames);
    check(left_top.at<cv::Vec3b>(100, 60) == toPixel(person), "box over left and top edges is drawn");

    const cv::Mat at_top = drawDetections(frame, { { 0, 0.9f, cv::Rect2f(10, 0, 80, 80) } }, kNames);
    check(!sameImage(at_top(cv::Rect(12, 2, 60, 14)), frame(cv::Rect(12, 2, 60, 14))),
        "label moves inside when there is no room above");

    const cv::Mat at_right = drawDetections(frame, { { 3, 0.9f, cv::Rect2f(300, 100, 19, 50) } }, kNames);
    check(at_right.size() == frame.size(), "long label at right edge is safe");

    const cv::Mat tiny(10, 12, CV_8UC3, cv::Scalar(kBackground));
    check(drawDetections(tiny, { { 3, 0.9f, cv::Rect2f(1, 1, 5, 5) } }, kNames).size() == tiny.size(),
        "label larger than frame is safe");

    check(sameImage(drawDetections(frame, { { 0, 0.9f, cv::Rect2f(400, 300, 20, 20) } }, kNames), frame),
        "box fully outside is not drawn");
    check(sameImage(drawDetections(frame, { { 7, 0.9f, cv::Rect2f(10, 10, 20, 20) } }, kNames), frame),
        "unknown class is not drawn");
}

int runModel(const std::string& model, const std::string& names, const std::string& yaml,
    const std::string& image_path, const std::string& save_path)
{
    ModelMeta meta;
    std::string error;
    OpenCvDnnDetector detector;
    if (!ModelMeta::load(yaml, meta, error) || !detector.load(model, names, meta, error)) {
        std::cerr << "[ERROR] " << error << "\n";
        return 3;
    }
    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << image_path << "\n";
        return 2;
    }

    const LetterboxResult boxed = letterbox(image, meta.imgsz, meta.pad_value);
    const auto result = detector.detect(toBlob(boxed.image, meta.scale, meta.swap_rb), boxed.info);
    if (!cv::imwrite(save_path, drawDetections(image, result, detector.classNames()))) {
        std::cerr << "[ERROR] Cannot save image: " << save_path << "\n";
        return 1;
    }
    std::cout << "[INFO] Detections: " << result.size() << "\n"
        << "[INFO] Saved: " << save_path << "\n";
    return 0;
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc >= 7 && std::string(argv[1]) == "--model") {
        return runModel(argv[2], argv[3], argv[4], argv[5], argv[6]);
    }

    testLabelsAndColors();
    testDrawing();
    testClipping();

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
