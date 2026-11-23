// DET-02 단위 테스트 (src/detect/Detector의 decodeDetections)
//
// 사용법:
//   test_detector_decode
//       코드로 만든 출력값으로 전체 검사 (모델 파일 불필요)
//   test_detector_decode --model <best.onnx> <classes.txt> <model.yaml> <이미지>
//       실제 모델로 추론해 후보 수와 점수 상위 5개를 출력

#include "detect/Detector.h"
#include "detect/OpenCvDnnDetector.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

bool near(float a, float b, float tolerance = 0.01f)
{
    return std::abs(a - b) <= tolerance;
}

struct Candidate {
    float cx, cy, w, h;
    std::vector<float> scores;
};

// 후보 목록을 모델 출력과 같은 {1, 4 + nc, N} 형태로 만듦
cv::Mat makeOutput(const std::vector<Candidate>& candidates, int nc = 5)
{
    const int size[] = { 1, 4 + nc, static_cast<int>(candidates.size()) };
    cv::Mat output(3, size, CV_32F, cv::Scalar(0));
    float* data = output.ptr<float>();
    const int n = size[2];
    for (int i = 0; i < n; ++i) {
        const Candidate& c = candidates[i];
        data[0 * n + i] = c.cx;
        data[1 * n + i] = c.cy;
        data[2 * n + i] = c.w;
        data[3 * n + i] = c.h;
        for (int k = 0; k < nc; ++k) data[(4 + k) * n + i] = c.scores[k];
    }
    return output;
}

LetterboxInfo infoFor(int width, int height)
{
    return letterbox(cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0)), 640).info;
}

const std::vector<float> kPerson = { 0.9f, 0.1f, 0.0f, 0.0f, 0.0f };

void testClasses()
{
    const LetterboxInfo info = infoFor(640, 640);
    std::vector<Candidate> candidates;
    for (int k = 0; k < 5; ++k) {
        std::vector<float> scores(5, 0.05f);
        scores[k] = 0.5f + k * 0.1f;
        candidates.push_back({ 100.0f, 100.0f, 40.0f, 40.0f, scores });
    }
    const auto result = decodeDetections(makeOutput(candidates), info);
    bool ok = result.size() == 5;
    for (int k = 0; ok && k < 5; ++k) {
        ok = result[k].class_id == k && near(result[k].score, 0.5f + k * 0.1f);
    }
    check(ok, "all 5 classes: highest score decides class_id");
}

void testCoordinates()
{
    auto result = decodeDetections(makeOutput({ { 100, 200, 40, 60, kPerson } }), infoFor(640, 640));
    check(result.size() == 1 && near(result[0].box.x, 80) && near(result[0].box.y, 170)
        && near(result[0].box.width, 40) && near(result[0].box.height, 60),
        "cx, cy, w, h -> x, y, w, h");

    // 1280x720 -> 배율 0.5, 위쪽 패딩 140
    result = decodeDetections(makeOutput({ { 320, 320, 100, 50, kPerson } }), infoFor(1280, 720));
    check(result.size() == 1 && near(result[0].box.x, 540) && near(result[0].box.y, 310)
        && near(result[0].box.width, 200) && near(result[0].box.height, 100),
        "landscape frame returns to original pixels");

    // 720x1280 -> 배율 0.5, 왼쪽 패딩 140
    result = decodeDetections(makeOutput({ { 320, 320, 50, 100, kPerson } }), infoFor(720, 1280));
    check(result.size() == 1 && near(result[0].box.x, 310) && near(result[0].box.y, 540)
        && near(result[0].box.width, 100) && near(result[0].box.height, 200),
        "portrait frame returns to original pixels");
}

void testClipping()
{
    const LetterboxInfo info = infoFor(640, 480);

    auto result = decodeDetections(makeOutput({ { 10, 90, 40, 40, kPerson } }), info);
    check(result.size() == 1 && near(result[0].box.x, 0) && near(result[0].box.y, 0),
        "box is clipped at left and top");

    result = decodeDetections(makeOutput({ { 630, 550, 40, 40, kPerson } }), info);
    check(result.size() == 1 && near(result[0].box.x + result[0].box.width, 639)
        && near(result[0].box.y + result[0].box.height, 479),
        "box is clipped at w-1 and h-1");

    result = decodeDetections(makeOutput({ { 320, 20, 40, 20, kPerson } }), info);
    check(result.empty(), "box inside the padding is dropped");

    result = decodeDetections(makeOutput({ { 320, 320, 0.5f, 40, kPerson } }), info);
    check(result.empty(), "width under 1 pixel is dropped");

    result = decodeDetections(makeOutput({ { 320, 320, 40, 0.5f, kPerson } }), info);
    check(result.empty(), "height under 1 pixel is dropped");
}

void testInvalidOutput()
{
    const LetterboxInfo info = infoFor(640, 640);
    check(decodeDetections(cv::Mat(), info).empty(), "empty output returns no detections");

    const int two_dims[] = { 9, 10 };
    check(decodeDetections(cv::Mat(2, two_dims, CV_32F, cv::Scalar(0)), info).empty(), "wrong dims");

    const int no_scores[] = { 1, 4, 10 };
    check(decodeDetections(cv::Mat(3, no_scores, CV_32F, cv::Scalar(0)), info).empty(), "no score rows");

    const int size[] = { 1, 9, 10 };
    check(decodeDetections(cv::Mat(3, size, CV_8U, cv::Scalar(0)), info).empty(), "wrong data type");
}

int runModel(const std::string& model, const std::string& names, const std::string& yaml,
    const std::string& image_path)
{
    ModelMeta meta;
    std::string error;
    OpenCvDnnDetector detector;
    if (!ModelMeta::load(yaml, meta, error) || !detector.load(model, names, meta, error)) {
        std::cerr << "[ERROR] " << error << "\n";
        return 3;
    }
    // 해석 결과 전체를 보기 위해 신뢰도 필터(DET-03)를 끔
    detector.setConfThresholds(std::vector<float>(meta.nc, 0.0f), error);

    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << image_path << "\n";
        return 2;
    }

    const LetterboxResult boxed = letterbox(image, meta.imgsz, meta.pad_value);
    auto result = detector.detect(toBlob(boxed.image, meta.scale, meta.swap_rb), boxed.info);
    std::sort(result.begin(), result.end(),
        [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::cout << "[INFO] Image: " << image.cols << "x" << image.rows << "\n"
        << "[INFO] Candidates: " << result.size() << "\n";
    for (size_t i = 0; i < std::min<size_t>(5, result.size()); ++i) {
        const Detection& d = result[i];
        std::cout << "[INFO] " << detector.classNames()[d.class_id] << " score=" << d.score
            << " box=(" << d.box.x << ", " << d.box.y << ", "
            << d.box.x + d.box.width << ", " << d.box.y + d.box.height << ")\n";
    }
    return 0;
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc >= 6 && std::string(argv[1]) == "--model") {
        return runModel(argv[2], argv[3], argv[4], argv[5]);
    }

    testClasses();
    testCoordinates();
    testClipping();
    testInvalidOutput();

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
