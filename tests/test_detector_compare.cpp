// DET-02, DET-05 실제 이미지 탐지 결과 저장 및 엔진 속도 측정
//
// 사용법:
//   test_detector_compare <opencv|ort-cpu|ort-cuda> <best.onnx> <classes.txt> <model.yaml>
//                         <이미지 폴더 또는 파일> <결과.csv> [이미지 수, 기본 10]
//
// 결과.csv는 tests/check_detections.py로 Python 결과와 비교함
// 속도는 워밍업과 첫 번째 전체 추론을 뺀 뒤 3회 평균을 냄 (DET-05)
// 종료 코드: 0 = 정상, 1 = 추론 실패, 2 = 이미지 없음, 3 = 모델 로드 실패

#include "detect/OpenCvDnnDetector.h"

#ifdef PPE_WITH_ORT
#include "detect/OrtDetector.h"
#endif

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

const int kSpeedRounds = 3;

struct Sample {
    std::string name;
    cv::Mat blob;
    LetterboxInfo info;
};

std::unique_ptr<IDetector> makeDetector(const std::string& engine)
{
    if (engine == "opencv") return std::make_unique<OpenCvDnnDetector>();
#ifdef PPE_WITH_ORT
    if (engine == "ort-cpu") return std::make_unique<OrtDetector>(OrtDevice::Cpu);
    if (engine == "ort-cuda") return std::make_unique<OrtDetector>(OrtDevice::Cuda);
#endif
    return nullptr;
}

bool isImage(const fs::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

// 폴더면 파일 이름순으로 앞에서 limit장을 고름
std::vector<fs::path> imageList(const fs::path& input, int limit)
{
    std::vector<fs::path> files;
    std::error_code ec;
    if (fs::is_regular_file(input, ec)) {
        files.push_back(input);
    } else if (fs::is_directory(input, ec)) {
        for (const auto& entry : fs::directory_iterator(input, ec)) {
            if (entry.is_regular_file() && isImage(entry.path())) files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            return a.filename().string() < b.filename().string();
        });
        if (static_cast<int>(files.size()) > limit) files.resize(limit);
    }
    return files;
}

void writeRows(std::ofstream& csv, const std::string& name, const std::vector<Detection>& detections)
{
    // 탐지가 0개인 이미지도 비교할 수 있도록 class_id -1로 한 줄 남김
    if (detections.empty()) {
        csv << '"' << name << "\",-1,0,0,0,0,0\n";
        return;
    }
    for (const Detection& d : detections) {
        csv << '"' << name << "\"," << d.class_id << ',' << d.score << ',' << d.box.x << ',' << d.box.y
            << ',' << d.box.x + d.box.width << ',' << d.box.y + d.box.height << '\n';
    }
}

double runRound(IDetector& detector, const std::vector<Sample>& samples)
{
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    for (const Sample& sample : samples) detector.detect(sample.blob, sample.info);
    const std::chrono::duration<double, std::milli> elapsed = Clock::now() - start;
    return elapsed.count() / samples.size();
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc < 7) {
        std::cerr << "[ERROR] Usage: " << argv[0]
            << " <opencv|ort-cpu|ort-cuda> <best.onnx> <classes.txt> <model.yaml> <images> <out.csv> [count]\n";
        return 1;
    }

    const auto detector = makeDetector(argv[1]);
    if (!detector) {
        std::cerr << "[ERROR] Unknown or unavailable engine: " << argv[1] << "\n";
        return 3;
    }

    ModelMeta meta;
    std::string error;
    if (!ModelMeta::load(argv[4], meta, error) || !detector->load(argv[2], argv[3], meta, error)) {
        std::cerr << "[ERROR] " << error << "\n";
        return 3;
    }

    // 디스크 읽기와 전처리가 속도에 섞이지 않도록 미리 준비함
    const int limit = argc >= 8 ? std::max(1, std::atoi(argv[7])) : 10;
    std::vector<Sample> samples;
    for (const fs::path& path : imageList(argv[5], limit)) {
        const cv::Mat image = cv::imread(path.string());
        if (image.empty()) {
            std::cerr << "[WARN] Cannot read image: " << path.string() << "\n";
            continue;
        }
        const LetterboxResult input = letterbox(image, meta.imgsz, meta.pad_value);
        samples.push_back({ path.filename().string(), toBlob(input.image, meta.scale, meta.swap_rb), input.info });
    }
    if (samples.empty()) {
        std::cerr << "[ERROR] No images: " << argv[5] << "\n";
        return 2;
    }

    std::ofstream csv(argv[6]);
    if (!csv.is_open()) {
        std::cerr << "[ERROR] Cannot write: " << argv[6] << "\n";
        return 1;
    }
    csv << "image,class_id,score,x1,y1,x2,y2\n" << std::fixed << std::setprecision(6);

    try {
        // 첫 번째 전체 추론: 결과만 저장하고 시간은 재지 않음
        int total = 0;
        for (const Sample& sample : samples) {
            const auto detections = detector->detect(sample.blob, sample.info);
            writeRows(csv, sample.name, detections);
            total += static_cast<int>(detections.size());
        }
        csv.close();
        std::cout << "[INFO] Images: " << samples.size() << ", detections: " << total << "\n";

        double sum = 0.0;
        for (int round = 1; round <= kSpeedRounds; ++round) {
            const double ms = runRound(*detector, samples);
            sum += ms;
            std::cout << "[INFO] Round " << round << ": " << ms << " ms/frame\n";
        }
        const double average = sum / kSpeedRounds;
        std::cout << "[INFO] Engine " << argv[1] << " average: " << average << " ms/frame ("
            << 1000.0 / average << " FPS, detect stage only)\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
