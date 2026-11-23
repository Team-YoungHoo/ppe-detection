#include "detect/OrtDetector.h"

#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

// 형태를 "1x9x8400"처럼 문자열로 바꿈
std::string shapeText(const std::vector<int64_t>& shape)
{
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) text += "x";
        text += std::to_string(shape[i]);
    }
    return text.empty() ? "empty" : text;
}

// CUDA 실행 장치를 등록함. TF32를 켜면 Python 검증의 허용 오차를 넘으므로 끔
void appendCuda(Ort::SessionOptions& options)
{
    const OrtApi& api = Ort::GetApi();

    OrtCUDAProviderOptionsV2* raw = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
    const std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
        cuda(raw, api.ReleaseCUDAProviderOptions);

    const std::array<const char*, 2> keys = { "device_id", "use_tf32" };
    const std::array<const char*, 2> values = { "0", "0" };
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda.get(), keys.data(), values.data(), keys.size()));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda.get()));
}

bool isCudaAvailable()
{
    const auto providers = Ort::GetAvailableProviders();
    return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
}

}  // namespace 끝

OrtDetector::OrtDetector(OrtDevice device)
    : device_(device), env_(ORT_LOGGING_LEVEL_WARNING, "ppe")
{
}

bool OrtDetector::load(const std::string& model_path, const std::string& names_path,
    const ModelMeta& meta, std::string& error)
{
    // 1. 파일 존재 확인
    std::error_code ec;
    const std::filesystem::path path(model_path);
    if (!std::filesystem::is_regular_file(path, ec)) {
        error = "Cannot load model: " + model_path;
        return false;
    }

    // 2. 클래스 이름 확인
    std::vector<std::string> names;
    if (!loadClassNames(names_path, meta, names, error)) return false;

    // 3. 실행 장치 등록. CUDA를 못 쓰면 CPU로 넘어가지 않고 실패함
    Ort::SessionOptions options;
    if (device_ == OrtDevice::Cuda) {
        if (!isCudaAvailable()) {
            error = "CUDA is not available in this ONNX Runtime build";
            return false;
        }
        try {
            appendCuda(options);
        } catch (const Ort::Exception& e) {
            error = std::string("CUDA is not available: ") + e.what();
            return false;
        }
    }

    // 4. 모델 로드
    std::unique_ptr<Ort::Session> session;
    std::string input_name, output_name;
    try {
        session = std::make_unique<Ort::Session>(env_, path.c_str(), options);
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1) {
            error = "Unexpected model inputs or outputs: " + model_path;
            return false;
        }
        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session->GetInputNameAllocated(0, allocator).get();
        output_name = session->GetOutputNameAllocated(0, allocator).get();
    } catch (const Ort::Exception& e) {
        error = (device_ == OrtDevice::Cuda ? "Cannot load model on CUDA: " : "Cannot load model: ")
            + model_path + " (" + e.what() + ")";
        return false;
    }

    // 5. 더미 이미지로 1회 추론(워밍업)하면서 실제 출력 형태를 얻음
    const int input_size[] = { 1, 3, meta.imgsz, meta.imgsz };
    cv::Mat dummy(4, input_size, CV_32F, cv::Scalar(0));
    const std::array<int64_t, 4> input_shape = { 1, 3, meta.imgsz, meta.imgsz };
    std::vector<int64_t> output_shape;
    cv::TickMeter timer;
    try {
        const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const Ort::Value input = Ort::Value::CreateTensor<float>(memory, dummy.ptr<float>(), dummy.total(),
            input_shape.data(), input_shape.size());
        const char* input_names[] = { input_name.c_str() };
        const char* output_names[] = { output_name.c_str() };
        timer.start();
        const auto outputs = session->Run(Ort::RunOptions{ nullptr }, input_names, &input, 1, output_names, 1);
        timer.stop();
        output_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    } catch (const Ort::Exception& e) {
        error = "Cannot run model: " + model_path + " (" + e.what() + ")";
        return false;
    }

    // 6. 출력 형태 확인: {1, 4 + 클래스 수, N}
    const std::string expected = "1x" + std::to_string(4 + meta.nc) + "x"
        + std::to_string(meta.output_shape.at(2));
    const std::string actual = shapeText(output_shape);
    if (actual != expected) {
        error = "Unexpected model output (expected: " + expected + ", actual: " + actual + ")";
        return false;
    }

    // Log.h(SYS-04)가 준비되면 교체
    std::cout << "[INFO] Model loaded: " << model_path << "\n"
        << "[INFO] Engine: ONNX Runtime " << Ort::GetVersionString() << "\n"
        << "[INFO] Device: " << (device_ == OrtDevice::Cuda ? "CUDA (TF32 off)" : "CPU") << "\n"
        << "[INFO] Input shape: " << shapeText({ input_shape.begin(), input_shape.end() }) << "\n"
        << "[INFO] Output shape: " << actual << "\n"
        << "[INFO] Classes: " << names.size() << "\n"
        << "[INFO] Warm-up: " << timer.getTimeMilli() << " ms\n";

    session_ = std::move(session);
    input_name_ = std::move(input_name);
    output_name_ = std::move(output_name);
    names_ = std::move(names);
    conf_thresholds_.assign(names_.size(), kDefaultConfThreshold);
    return true;
}

Ort::Value OrtDetector::run(const cv::Mat& blob)
{
    if (blob.dims != 4 || blob.type() != CV_32F || !blob.isContinuous()) {
        throw std::invalid_argument("blob must be a continuous 4D float32 tensor");
    }

    const std::array<int64_t, 4> shape = { blob.size[0], blob.size[1], blob.size[2], blob.size[3] };
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ONNX Runtime은 입력을 읽기만 하지만 API가 쓰기 가능한 포인터를 요구함
    const Ort::Value input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(blob.ptr<float>()),
        blob.total(), shape.data(), shape.size());
    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };
    auto outputs = session_->Run(Ort::RunOptions{ nullptr }, input_names, &input, 1, output_names, 1);
    return std::move(outputs.front());
}

std::vector<Detection> OrtDetector::detect(const cv::Mat& blob, const LetterboxInfo& info)
{
    if (!session_ || blob.empty()) return {};

    Ort::Value result = run(blob);
    const std::vector<int64_t> shape = result.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3) throw std::runtime_error("Unexpected model output: " + shapeText(shape));

    // 복사 없이 출력 메모리를 Mat으로 감쌈. result가 살아 있는 동안만 사용함
    const int sizes[] = { static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]) };
    const cv::Mat output(3, sizes, CV_32F, result.GetTensorMutableData<float>());

    // 명세 순서: 출력 해석 -> 신뢰도 필터 -> NMS -> 원본 좌표 복원
    const auto candidates = decodeCandidates(output);
    const auto confident = filterByConfidence(candidates, conf_thresholds_);
    return restoreToOriginal(applyNms(confident, nms_iou_), info);
}

bool OrtDetector::setConfThresholds(const std::vector<float>& thresholds, std::string& error)
{
    if (!isValidConfThresholds(thresholds, static_cast<int>(names_.size()))) {
        error = "Invalid confidence thresholds (need one 0~1 value per class)";
        return false;
    }
    conf_thresholds_ = thresholds;
    return true;
}

bool OrtDetector::setNmsIou(float iou, std::string& error)
{
    if (!isValidNmsIou(iou)) {
        error = "Invalid NMS IoU (need a 0~1 value)";
        return false;
    }
    nms_iou_ = iou;
    return true;
}
