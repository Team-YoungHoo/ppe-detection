// DET-05 단위 테스트 (src/detect/OrtDetector)
//
// 사용법:
//   test_ort_detector
//       모델 파일 없이 할 수 있는 실패 상황 검사
//   test_ort_detector --model <best.onnx> <classes.txt> <model.yaml> <이미지> <cpu|cuda|no-cuda>
//       cpu, cuda : 실제 모델로 탐지하고 OpenCV DNN 결과와 비교
//       no-cuda   : GPU가 없는 PC에서 CUDA 지정 시 로드가 실패하는지 확인

#include "detect/OpenCvDnnDetector.h"
#include "detect/OrtDetector.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

const float kToleranceBox = 0.5f;      // verify_onnx.py와 같은 허용 박스 차이 (px)
const float kToleranceScore = 0.001f;  // verify_onnx.py와 같은 허용 점수 차이

int g_failed = 0;
fs::path g_dir;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

std::string writeFile(const std::string& name, const std::string& text)
{
    const fs::path path = g_dir / name;
    std::ofstream file(path, std::ios::binary);
    file << text;
    return path.string();
}

ModelMeta sampleMeta()
{
    ModelMeta meta;
    meta.imgsz = 640;
    meta.nc = 5;
    meta.names = { "person", "helmet", "vest", "no-helmet", "no-vest" };
    meta.output_shape = { 1, 9, 8400 };
    return meta;
}

void expectError(const std::string& model, const std::string& names, const ModelMeta& meta,
    const std::string& word, const std::string& name)
{
    OrtDetector detector(OrtDevice::Cpu);
    std::string error;
    const bool loaded = detector.load(model, names, meta, error);
    check(!loaded && error.find(word) != std::string::npos && detector.classNames().empty(),
        name + " (" + error.substr(0, 90) + ")");
}

void testWithoutModel()
{
    const ModelMeta meta = sampleMeta();
    const std::string good = writeFile("good.txt", "person\nhelmet\nvest\nno-helmet\nno-vest\n");
    const std::string fake = writeFile("fake.onnx", "this is not an onnx model");

    expectError((g_dir / "none.onnx").string(), good, meta, "Cannot load model", "missing model file");
    expectError(fake, (g_dir / "none.txt").string(), meta, "Cannot load classes", "missing classes file");
    expectError(fake, writeFile("short.txt", "person\nhelmet\n"), meta, "count mismatch", "too few classes");
    expectError(fake, writeFile("order.txt", "helmet\nperson\nvest\nno-helmet\nno-vest\n"), meta,
        "order", "wrong class order");
    expectError(fake, good, meta, "Cannot load model", "broken model file");

    OrtDetector empty(OrtDevice::Cpu);
    check(empty.detect(cv::Mat(), LetterboxInfo{}).empty(), "detect before load returns nothing");
}

// 두 엔진의 결과를 같은 순서로 맞춤
std::vector<Detection> sorted(std::vector<Detection> items)
{
    std::sort(items.begin(), items.end(), [](const Detection& a, const Detection& b) {
        if (a.class_id != b.class_id) return a.class_id < b.class_id;
        return a.score > b.score;
    });
    return items;
}

void compare(const std::vector<Detection>& ort_result, const std::vector<Detection>& cv_result)
{
    check(!ort_result.empty(), "detections found: " + std::to_string(ort_result.size()));
    check(ort_result.size() == cv_result.size(), "same count as OpenCV DNN: "
        + std::to_string(ort_result.size()) + " / " + std::to_string(cv_result.size()));
    if (ort_result.size() != cv_result.size()) return;

    const auto a = sorted(ort_result);
    const auto b = sorted(cv_result);
    bool same_class = true;
    float max_box = 0.0f, max_score = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        same_class = same_class && a[i].class_id == b[i].class_id;
        max_score = std::max(max_score, std::abs(a[i].score - b[i].score));
        max_box = std::max({ max_box, std::abs(a[i].box.x - b[i].box.x), std::abs(a[i].box.y - b[i].box.y),
            std::abs(a[i].box.width - b[i].box.width), std::abs(a[i].box.height - b[i].box.height) });
    }
    check(same_class, "same classes as OpenCV DNN");
    check(max_box <= kToleranceBox, "box difference: " + std::to_string(max_box) + " px");
    check(max_score <= kToleranceScore, "score difference: " + std::to_string(max_score));
}

void testWithModel(const std::string& model, const std::string& names, const std::string& yaml,
    const std::string& image_path, const std::string& device)
{
    ModelMeta meta;
    std::string error;
    check(ModelMeta::load(yaml, meta, error), "model.yaml loads " + error);
    if (!error.empty()) return;

    if (device == "no-cuda") {
        OrtDetector detector(OrtDevice::Cuda);
        const bool loaded = detector.load(model, names, meta, error);
        check(!loaded && error.find("CUDA") != std::string::npos && detector.classNames().empty(),
            "CUDA failure stops loading (" + error.substr(0, 90) + ")");
        return;
    }

    OrtDetector detector(device == "cuda" ? OrtDevice::Cuda : OrtDevice::Cpu);
    check(detector.load(model, names, meta, error), "real model loads " + error);
    if (!error.empty()) return;
    check(detector.classNames() == meta.names, "class names keep their order");

    const cv::Mat image = cv::imread(image_path);
    check(!image.empty(), "image loads");
    if (image.empty()) return;
    const LetterboxResult input = letterbox(image, meta.imgsz, meta.pad_value);
    const cv::Mat blob = toBlob(input.image, meta.scale, meta.swap_rb);

    OpenCvDnnDetector reference;
    check(reference.load(model, names, meta, error), "OpenCV DNN loads " + error);
    if (!error.empty()) return;
    compare(detector.detect(blob, input.info), reference.detect(blob, input.info));

    ModelMeta wrong_n = meta;
    wrong_n.output_shape[2] = 100;
    OrtDetector wrong(OrtDevice::Cpu);
    error.clear();
    check(!wrong.load(model, names, wrong_n, error) && error.find("Unexpected model output") != std::string::npos,
        "output size differs (" + error + ")");
}

}  // namespace 끝

int main(int argc, char** argv)
{
    g_dir = fs::temp_directory_path() / "ppe_test_ort_detector";
    std::error_code ec;
    fs::remove_all(g_dir, ec);
    fs::create_directories(g_dir);

    testWithoutModel();
    if (argc >= 7 && std::string(argv[1]) == "--model") {
        testWithModel(argv[2], argv[3], argv[4], argv[5], argv[6]);
    }

    fs::remove_all(g_dir, ec);
    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
