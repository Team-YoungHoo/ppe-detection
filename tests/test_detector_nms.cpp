// DET-04 단위 테스트 (src/detect/Detector의 applyNms)
//
// 사용법:
//   test_detector_nms
//       코드로 만든 박스로 전체 검사 (모델 파일 불필요)
//   test_detector_nms --model <best.onnx> <classes.txt> <model.yaml> <이미지>
//       실제 모델의 최종 탐지 결과를 출력

#include "detect/Detector.h"
#include "detect/OpenCvDnnDetector.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

Detection box(int class_id, float score, float x, float y, float w = 100.0f, float h = 100.0f)
{
    return { class_id, score, cv::Rect2f(x, y, w, h) };
}

void testNms()
{
    check(applyNms({}, kDefaultNmsIou).empty(), "no boxes returns empty");

    auto kept = applyNms({ box(0, 0.6f, 10, 10), box(0, 0.9f, 12, 10), box(0, 0.7f, 10, 12) }, 0.45f);
    check(kept.size() == 1 && kept[0].score == 0.9f, "overlapping boxes keep the highest score");
    check(kept.size() == 1 && kept[0].box == cv::Rect2f(12, 10, 100, 100), "kept box is unchanged");

    kept = applyNms({ box(0, 0.9f, 0, 0), box(0, 0.8f, 300, 300) }, 0.45f);
    check(kept.size() == 2, "separate people stay separate");

    // 가로로 50 겹침: IoU = 50 / 150 = 0.33
    kept = applyNms({ box(0, 0.9f, 0, 0), box(0, 0.8f, 50, 0) }, 0.45f);
    check(kept.size() == 2, "two close people with IoU 0.33 are not merged");

    // 가로로 80 겹침: IoU = 80 / 120 = 0.67
    kept = applyNms({ box(0, 0.9f, 0, 0), box(0, 0.8f, 20, 0) }, 0.45f);
    check(kept.size() == 1, "IoU 0.67 is merged");

    kept = applyNms({ box(1, 0.9f, 10, 10), box(3, 0.8f, 10, 10) }, 0.45f);
    check(kept.size() == 2, "helmet and no-helmet on one head both remain");

    kept = applyNms({ box(0, 0.9f, 0, 0, 100, 200), box(1, 0.8f, 30, 0, 40, 40) }, 0.45f);
    check(kept.size() == 2, "person box does not remove helmet box");

    std::vector<Detection> all;
    for (int c = 0; c < 5; ++c) {
        all.push_back(box(c, 0.9f, 10, 10));
        all.push_back(box(c, 0.5f, 12, 10));
    }
    kept = applyNms(all, 0.45f);
    bool one_each = kept.size() == 5;
    for (const Detection& d : kept) one_each = one_each && d.score == 0.9f;
    check(one_each, "NMS runs separately for all 5 classes");
}

void testIouValues()
{
    check(isValidNmsIou(0.0f) && isValidNmsIou(0.45f) && isValidNmsIou(1.0f), "0, 0.45, 1 are valid");
    check(!isValidNmsIou(-0.01f), "negative IoU is invalid");
    check(!isValidNmsIou(1.01f), "IoU over 1 is invalid");
    check(!isValidNmsIou(std::numeric_limits<float>::quiet_NaN()), "NaN is invalid");

    OpenCvDnnDetector detector;
    std::string error;
    check(!detector.setNmsIou(2.0f, error) && !error.empty(), "detector rejects invalid IoU");
    check(detector.setNmsIou(0.5f, error), "detector accepts valid IoU");
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
    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << image_path << "\n";
        return 2;
    }

    const LetterboxResult boxed = letterbox(image, meta.imgsz, meta.pad_value);
    auto result = detector.detect(toBlob(boxed.image, meta.scale, meta.swap_rb), boxed.info);
    std::sort(result.begin(), result.end(),
        [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::cout << "[INFO] Detections: " << result.size() << "\n";
    for (const Detection& d : result) {
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

    testNms();
    testIouValues();

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
