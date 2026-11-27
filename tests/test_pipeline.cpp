// SYS-05 Pipeline 뼈대 테스트 (src/pipeline/Pipeline)
//
// 사용법:
//   test_pipeline                             코드로 만든 테스트 영상으로 창 없이 검사
//   test_pipeline --play <영상>               실제 영상을 창으로 끝까지 처리
//   test_pipeline --play <영상> --no-display  실제 영상을 창 없이 끝까지 처리
//                                             영상을 열 수 없으면 종료 코드 2
//   test_pipeline --detect <opencv|ort-cpu|ort-cuda> <best.onnx> <classes.txt> <model.yaml> <영상>
//                 [--debug-draw] [--no-display] [--track-log <csv>]
//                                             --track-log: 프레임별 트랙(frame,id,state,x,y,w,h)을 CSV로 저장
//                                             실제 모델로 탐지까지 처리. 모델 로드 실패는 종료 코드 3

#include "detect/OpenCvDnnDetector.h"
#include "pipeline/Pipeline.h"

#ifdef PPE_WITH_ORT
#include "detect/OrtDetector.h"
#endif

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

// 정해 둔 프레임에서 실패하는 가짜 탐지기. 모델 없이 파이프라인 동작만 검사함
class FakeDetector : public IDetector {
public:
    explicit FakeDetector(std::function<bool(int)> fail_at) : fail_at_(std::move(fail_at)) {}

    bool load(const std::string&, const std::string&, const ModelMeta&, std::string&) override { return true; }

    std::vector<Detection> detect(const cv::Mat& blob, const LetterboxInfo& info) override
    {
        const int call = calls++;
        valid_input = valid_input && blob.dims == 4 && blob.size[1] == 3 && blob.size[2] == 640
            && info.orig_w == 160 && info.orig_h == 120;
        if (fail_at_(call)) throw std::runtime_error("fake failure");
        return { Detection{ 0, 0.9f, cv::Rect2f(10.0f, 10.0f, 40.0f, 40.0f) } };
    }

    bool setConfThresholds(const std::vector<float>&, std::string&) override { return true; }
    bool setNmsIou(float, std::string&) override { return true; }
    const std::vector<std::string>& classNames() const override { return names_; }

    int calls = 0;
    bool valid_input = true;

private:
    std::function<bool(int)> fail_at_;
    std::vector<std::string> names_ = { "person" };
};

const int kFrames = 30;

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

bool writeVideo(const std::string& path)
{
    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, cv::Size(160, 120));
    if (!writer.isOpened()) return false;
    for (int i = 0; i < kFrames; ++i) {
        const int v = 10 + i * 8;
        writer.write(cv::Mat(120, 160, CV_8UC3, cv::Scalar(v, v, v)));
    }
    return true;
}

PipelineOptions noDisplay(const std::string& path)
{
    PipelineOptions options;
    options.input = path;
    options.display = false;
    return options;
}

void testBadPath()
{
    Pipeline pipeline(noDisplay("no_such_video.mp4"));
    check(pipeline.run() == 2, "missing file: exit code 2");
    check(pipeline.processedFrames() == 0, "missing file: no frame is processed");
}

void testRun(const std::string& path)
{
    Pipeline pipeline(noDisplay(path));
    check(pipeline.run() == 0, "valid video: exit code 0");
    check(pipeline.processedFrames() == kFrames, "valid video: all frames are processed");

    check(pipeline.run() == 0 && pipeline.processedFrames() == kFrames,
        "second run: frame count starts again");
}

void testDetector(const std::string& path)
{
    {
        FakeDetector detector([](int) { return false; });
        PipelineOptions options = noDisplay(path);
        options.detector = &detector;
        Pipeline pipeline(options);
        check(pipeline.run() == 0 && pipeline.processedFrames() == kFrames && pipeline.failedFrames() == 0,
            "detector: all frames are processed");
        check(detector.calls == kFrames, "detector: called once per frame");
        check(detector.valid_input, "detector: receives blob and original frame size");
        check(pipeline.createdTracks() == 1 && pipeline.confirmedTracks() == 1,
            "tracker: one still person keeps one track id");
    }
    {
        FakeDetector detector([](int call) { return call < 5; });
        PipelineOptions options = noDisplay(path);
        options.detector = &detector;
        Pipeline pipeline(options);
        check(pipeline.run() == 0 && pipeline.processedFrames() == kFrames - 5 && pipeline.failedFrames() == 5,
            "failure: failed frames are skipped");
    }
    {
        FakeDetector detector([](int call) { return call % 2 == 0; });
        PipelineOptions options = noDisplay(path);
        options.detector = &detector;
        options.max_consecutive_fail = 2;
        Pipeline pipeline(options);
        check(pipeline.run() == 0 && detector.calls == kFrames, "failure: success resets the failure count");
    }
    {
        FakeDetector detector([](int) { return true; });
        PipelineOptions options = noDisplay(path);
        options.detector = &detector;
        options.max_consecutive_fail = 10;
        Pipeline pipeline(options);
        check(pipeline.run() == 1 && detector.calls == 10 && pipeline.processedFrames() == 0,
            "failure: stops with exit code 1 at the limit");
    }
    {
        FakeDetector detector([](int) { return true; });
        PipelineOptions options = noDisplay(path);
        options.detector = &detector;
        Pipeline pipeline(options);
        check(pipeline.run() == 1 && detector.calls == 30, "failure: default limit is 30 frames");
    }
}

int play(const std::string& path, bool display)
{
    PipelineOptions options;
    options.input = path;
    options.display = display;

    Pipeline pipeline(options);
    return pipeline.run();
}

std::unique_ptr<IDetector> makeDetector(const std::string& engine)
{
    if (engine == "opencv") return std::make_unique<OpenCvDnnDetector>();
#ifdef PPE_WITH_ORT
    if (engine == "ort-cpu") return std::make_unique<OrtDetector>(OrtDevice::Cpu);
    if (engine == "ort-cuda") return std::make_unique<OrtDetector>(OrtDevice::Cuda);
#endif
    return nullptr;
}

// 명세의 시작 순서: 모델 로드 -> 영상 열기 -> 창 생성
int runDetect(int argc, char** argv)
{
    const auto detector = makeDetector(argv[2]);
    if (!detector) {
        std::cerr << "[ERROR] Unknown or unavailable engine: " << argv[2] << "\n";
        return 3;
    }

    ModelMeta meta;
    std::string error;
    if (!ModelMeta::load(argv[5], meta, error) || !detector->load(argv[3], argv[4], meta, error)) {
        std::cerr << "[ERROR] " << error << "\n";
        return 3;
    }

    PipelineOptions options;
    options.input = argv[6];
    options.model_size = meta.imgsz;
    options.pad_value = meta.pad_value;
    options.scale = meta.scale;
    options.swap_rb = meta.swap_rb;
    options.detector = detector.get();
    for (int i = 7; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--debug-draw") options.debug_draw = true;
        if (flag == "--no-display") options.display = false;
        if (flag == "--track-log" && i + 1 < argc) options.track_log = argv[++i];
    }

    Pipeline pipeline(options);
    return pipeline.run();
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string(argv[1]) == "--play") {
        const bool no_display = argc >= 4 && std::string(argv[3]) == "--no-display";
        return play(argv[2], !no_display);
    }
    if (argc >= 7 && std::string(argv[1]) == "--detect") {
        return runDetect(argc, argv);
    }

    const auto dir = std::filesystem::temp_directory_path();
    const std::string path = (dir / "ppe_test_pipeline.avi").string();
    if (!writeVideo(path)) {
        std::cerr << "[ERROR] Cannot create the test video: " << path << "\n";
        return 1;
    }

    testBadPath();
    testRun(path);
    testDetector(path);

    std::error_code ec;
    std::filesystem::remove(path, ec);

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
