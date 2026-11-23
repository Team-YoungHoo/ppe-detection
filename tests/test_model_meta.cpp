// DET-07 단위 테스트 (src/config/ModelMeta)
//
// 사용법:
//   test_model_meta                  코드로 만든 model.yaml로 전체 검사
//   test_model_meta --info <파일>    실제 model.yaml을 읽어 값을 출력
//                                    읽을 수 없으면 종료 코드 3

#include "config/ModelMeta.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

// export_onnx.py가 저장하는 형식과 같은 예시
const char* const kValidYaml = R"yaml(%YAML 1.1
---
# TRN-06 / DET-07 주석 줄
model_name: "test_model"
onnx_file: "best.onnx"
input:
  name: "images"
  shape:
    - 1
    - 3
    - 640
    - 640
  width: 640
  height: 640
output:
  name: "output0"
  shape:
    - 1
    - 9
    - 8400
  format: "[1, 4+nc, N], 행 0~3 = cx, cy, w, h (입력 이미지 픽셀)"
  nms_included: false
nc: 5
names:
  - "person"
  - "helmet"
  - "vest"
  - "no-helmet"
  - "no-vest"
preprocess:
  resize: "letterbox"
  pad_value: 114
  scale: 0.00392156862745098
  swap_rb: true
  layout: "NCHW"
  dtype: "float32"
train:
  data: "construction safety.v3-release.yolov8"
  imgsz: 640
  scale: 0.7
onnx_sha256: "1f3ca87a3a2410a5584c38d710d92b63eee467003af48bf85630a88149046a5c"
exported_at: "2026-11-21T02:04:39"
)yaml";

int g_failed = 0;
std::string g_path;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

std::string replaced(std::string text, const std::string& from, const std::string& to)
{
    const auto pos = text.find(from);
    if (pos != std::string::npos) text.replace(pos, from.size(), to);
    return text;
}

bool loadText(const std::string& text, ModelMeta& meta, std::string& error)
{
    {
        std::ofstream file(g_path, std::ios::binary);
        file << text;
    }
    return ModelMeta::load(g_path, meta, error);
}

// 잘못된 파일은 실패해야 하고, 오류에 원인 이름이 들어 있어야 함
void expectError(const std::string& text, const std::string& word, const std::string& name)
{
    ModelMeta meta;
    std::string error;
    const bool loaded = loadText(text, meta, error);
    check(!loaded && error.find(word) != std::string::npos, name + " (" + error + ")");
}

void testValid()
{
    ModelMeta meta;
    std::string error;
    check(loadText(kValidYaml, meta, error), "valid file loads");
    check(meta.model_name == "test_model", "model name");
    check(meta.imgsz == 640, "input size 640");
    check(meta.nc == 5, "class count 5");
    check(meta.names == std::vector<std::string>{"person", "helmet", "vest", "no-helmet", "no-vest"},
        "class names keep their order");
    check(meta.output_shape == std::vector<int>{1, 9, 8400}, "output shape {1, 9, 8400}");
    check(meta.pad_value == 114, "pad value 114");
    check(std::abs(meta.scale - 1.0 / 255.0) < 1e-12, "scale from preprocess, not from train");
    check(meta.swap_rb, "swap_rb true");
}

void testLineEndings()
{
    ModelMeta meta;
    std::string error;
    std::string crlf;
    for (const char c : std::string(kValidYaml)) {
        if (c == '\n') crlf += '\r';
        crlf += c;
    }
    check(loadText(crlf, meta, error) && meta.names.back() == "no-vest", "Windows line endings load");
}

void testErrors()
{
    const std::string valid = kValidYaml;
    ModelMeta meta;
    std::string error;

    check(!ModelMeta::load("no_such_model.yaml", meta, error), "missing file fails");

    expectError("model_name: test\nnames:\n- person\n", "invalid YAML", "unsupported YAML format");
    expectError(replaced(valid, "  pad_value: 114\n", ""), "preprocess.pad_value", "missing pad_value");
    expectError(replaced(replaced(valid, "nc: 5\n", ""), "  swap_rb: true\n", ""), "swap_rb",
        "all missing items are listed");
    expectError(replaced(valid, "  width: 640", "  width: 960"), "square", "wrong imgsz (DET-07 check)");
    expectError(replaced(replaced(valid, "  width: 640", "  width: 960"), "  height: 640", "  height: 960"),
        "input.shape", "imgsz different from input.shape");
    expectError(replaced(valid, "  imgsz: 640", "  imgsz: 320"), "train.imgsz", "train.imgsz differs from input size");
    expectError(replaced(valid, "  imgsz: 640", "  imgsz: \"abc\""), "train.imgsz", "train.imgsz is not a number");
    check(loadText(replaced(valid, "  imgsz: 640\n", ""), meta, error), "train.imgsz is optional");
    expectError(replaced(valid, "  - \"no-vest\"\n", ""), "names", "names count differs from nc");
    expectError(replaced(valid, "    - 9\n", "    - 8\n"), "output.shape", "output channels differ");
    expectError(replaced(valid, "pad_value: 114", "pad_value: \"abc\""), "pad_value", "pad value is not a number");
    expectError(replaced(valid, "pad_value: 114", "pad_value: 300"), "pad_value", "pad value out of range");
    expectError(replaced(valid, "scale: 0.00392156862745098", "scale: 0"), "scale", "scale is zero");
    expectError(replaced(valid, "swap_rb: true", "swap_rb: maybe"), "swap_rb", "swap_rb is not a bool");
    expectError(replaced(valid, "resize: \"letterbox\"", "resize: \"stretch\""), "resize", "unsupported resize");
    expectError(replaced(valid, "layout: \"NCHW\"", "layout: \"NHWC\""), "layout", "unsupported layout");
}

int info(const std::string& path)
{
    ModelMeta meta;
    std::string error;
    if (!ModelMeta::load(path, meta, error)) {
        std::cerr << "[ERROR] Cannot load model meta: " << error << "\n";
        return 3;
    }
    std::cout << "[INFO] Model: " << meta.model_name << "\n"
        << "[INFO] Input size: " << meta.imgsz << "\n"
        << "[INFO] Classes (" << meta.nc << "):";
    for (const auto& name : meta.names) std::cout << " " << name;
    std::cout << "\n[INFO] Output shape: {" << meta.output_shape[0] << ", " << meta.output_shape[1]
        << ", " << meta.output_shape[2] << "}\n"
        << "[INFO] Preprocess: pad " << meta.pad_value << ", scale " << meta.scale
        << ", swap_rb " << (meta.swap_rb ? "true" : "false") << "\n";
    return 0;
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string(argv[1]) == "--info") {
        return info(argv[2]);
    }

    g_path = (std::filesystem::temp_directory_path() / "ppe_test_model_meta.yaml").string();

    testValid();
    testLineEndings();
    testErrors();

    std::error_code ec;
    std::filesystem::remove(g_path, ec);

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
