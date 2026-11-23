// DET-03 단위 테스트 (src/detect/Detector의 신뢰도 필터)
//
// 사용법:
//   test_detector_filter
//       코드로 만든 후보로 전체 검사 (모델 파일 불필요)
//   test_detector_filter --model <best.onnx> <classes.txt> <model.yaml> <이미지>
//       실제 모델로 임계값 0.25를 적용해 클래스별 개수를 출력

#include "detect/Detector.h"
#include "detect/OpenCvDnnDetector.h"

#include <opencv2/imgcodecs.hpp>

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

Detection candidate(int class_id, float score)
{
    return { class_id, score, cv::Rect2f(1.0f, 2.0f, 30.0f, 40.0f) };
}

void testFilter()
{
    const std::vector<float> same(5, 0.25f);

    auto kept = filterByConfidence({ candidate(0, 0.24f) }, same);
    check(kept.empty(), "score below threshold is removed");

    kept = filterByConfidence({ candidate(0, 0.25f) }, same);
    check(kept.size() == 1, "score equal to threshold is kept");

    kept = filterByConfidence({ candidate(0, 0.26f) }, same);
    check(kept.size() == 1 && kept[0].box == cv::Rect2f(1.0f, 2.0f, 30.0f, 40.0f),
        "kept candidate is unchanged");

    check(filterByConfidence({}, same).empty(), "no candidates returns empty");

    // 같은 점수라도 클래스별 임계값에 따라 결과가 달라야 함
    const std::vector<float> per_class = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    std::vector<Detection> all;
    for (int c = 0; c < 5; ++c) all.push_back(candidate(c, 0.35f));
    kept = filterByConfidence(all, per_class);
    check(kept.size() == 3 && kept[0].class_id == 0 && kept[1].class_id == 1
        && kept[2].class_id == 2, "each class uses its own threshold");

    kept = filterByConfidence({ candidate(-1, 0.9f), candidate(5, 0.9f) }, same);
    check(kept.empty(), "unknown class_id is removed");
}

void testThresholdValues()
{
    check(isValidConfThresholds({ 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }, 5), "0 and 1 are valid");
    check(!isValidConfThresholds({ -0.01f, 0.25f, 0.25f, 0.25f, 0.25f }, 5), "negative value is invalid");
    check(!isValidConfThresholds({ 0.25f, 0.25f, 0.25f, 0.25f, 1.01f }, 5), "value over 1 is invalid");
    check(!isValidConfThresholds({ 0.25f, std::numeric_limits<float>::quiet_NaN(), 0.25f, 0.25f, 0.25f }, 5),
        "NaN is invalid");
    check(!isValidConfThresholds({ 0.25f, 0.25f }, 5), "wrong count is invalid");
    check(!isValidConfThresholds({}, 0), "no classes is invalid");

    OpenCvDnnDetector detector;
    std::string error;
    check(!detector.setConfThresholds(std::vector<float>(5, 0.25f), error), "cannot set before model load");
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

    // 신뢰도 필터만의 결과를 보기 위해 NMS(DET-04)를 끔
    detector.setNmsIou(1.0f, error);

    const LetterboxResult boxed = letterbox(image, meta.imgsz, meta.pad_value);
    const cv::Mat blob = toBlob(boxed.image, meta.scale, meta.swap_rb);
    const auto kept = detector.detect(blob, boxed.info);

    std::vector<int> counts(meta.nc, 0);
    for (const Detection& d : kept) ++counts[d.class_id];
    std::cout << "[INFO] Kept at 0.25: " << kept.size() << "\n";
    for (int c = 0; c < meta.nc; ++c) {
        std::cout << "[INFO] " << detector.classNames()[c] << ": " << counts[c] << "\n";
    }

    // 잘못된 값은 거부하고 기존 임계값을 유지해야 함
    const bool rejected = !detector.setConfThresholds(std::vector<float>(meta.nc, 2.0f), error);
    check(rejected && detector.detect(blob, boxed.info).size() == kept.size(),
        "invalid thresholds keep previous values");

    check(detector.setConfThresholds(std::vector<float>(meta.nc, 1.0f), error)
        && detector.detect(blob, boxed.info).empty(), "threshold 1.0 removes all candidates");

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc >= 6 && std::string(argv[1]) == "--model") {
        return runModel(argv[2], argv[3], argv[4], argv[5]);
    }

    testFilter();
    testThresholdValues();

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
