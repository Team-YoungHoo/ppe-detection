// DET-01 단위 테스트 (src/detect/OpenCvDnnDetector의 load)
//
// 사용법:
//   test_detector_load
//       모델 파일 없이 할 수 있는 실패 상황 검사
//   test_detector_load --model <best.onnx> <classes.txt> <model.yaml>
//       실제 모델로 로드, 출력 형태, 실패 상황까지 검사

#include "detect/OpenCvDnnDetector.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

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

// 로드가 실패해야 하고, 오류에 원인 문구가 들어 있어야 함
void expectError(const std::string& model, const std::string& names, const ModelMeta& meta,
    const std::string& word, const std::string& name)
{
    OpenCvDnnDetector detector;
    std::string error;
    const bool loaded = detector.load(model, names, meta, error);
    check(!loaded && error.find(word) != std::string::npos && detector.classNames().empty(),
        name + " (" + error + ")");
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

void testWithoutModel()
{
    const ModelMeta meta = sampleMeta();
    const std::string good = writeFile("good.txt", "person\nhelmet\nvest\nno-helmet\nno-vest\n");
    const std::string fake = writeFile("fake.onnx", "this is not an onnx model");

    expectError((g_dir / "none.onnx").string(), good, meta, "Cannot load model", "missing model file");
    expectError(fake, (g_dir / "none.txt").string(), meta, "Cannot load classes", "missing classes file");
    expectError(fake, writeFile("short.txt", "person\nhelmet\n"), meta, "count mismatch", "too few classes");
    expectError(fake, writeFile("long.txt", "person\nhelmet\nvest\nno-helmet\nno-vest\nextra\n"), meta,
        "count mismatch", "too many classes");
    expectError(fake, writeFile("order.txt", "helmet\nperson\nvest\nno-helmet\nno-vest\n"), meta,
        "order", "wrong class order");
    expectError(fake, good, meta, "Cannot load model", "broken model file");
}

void testWithModel(const std::string& model, const std::string& names, const std::string& yaml)
{
    ModelMeta meta;
    std::string error;
    check(ModelMeta::load(yaml, meta, error), "model.yaml loads " + error);
    if (!error.empty()) return;

    OpenCvDnnDetector detector;
    check(detector.load(model, names, meta, error), "real model loads " + error);
    check(detector.classNames() == meta.names, "class names keep their order");

    OpenCvDnnDetector crlf;
    error.clear();
    const std::string crlf_names = writeFile("crlf.txt", "person\r\nhelmet\r\nvest\r\nno-helmet\r\nno-vest\r\n");
    check(crlf.load(model, crlf_names, meta, error), "Windows line endings in classes " + error);

    ModelMeta wrong_n = meta;
    wrong_n.output_shape[2] = 100;
    expectError(model, names, wrong_n, "Unexpected model output", "output size differs");

    ModelMeta wrong_nc = meta;
    wrong_nc.nc = 6;
    wrong_nc.names.push_back("extra");
    const std::string six = writeFile("six.txt", "person\nhelmet\nvest\nno-helmet\nno-vest\nextra\n");
    expectError(model, six, wrong_nc, "Unexpected model output", "output channels differ from 4 + nc");
}

}  // namespace 끝

int main(int argc, char** argv)
{
    g_dir = fs::temp_directory_path() / "ppe_test_detector_load";
    std::error_code ec;
    fs::remove_all(g_dir, ec);
    fs::create_directories(g_dir);

    testWithoutModel();
    if (argc >= 5 && std::string(argv[1]) == "--model") {
        testWithModel(argv[2], argv[3], argv[4]);
    }

    fs::remove_all(g_dir, ec);
    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
