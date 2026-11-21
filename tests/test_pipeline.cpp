// SYS-05 Pipeline 뼈대 테스트 (src/pipeline/Pipeline)
//
// 사용법:
//   test_pipeline                             코드로 만든 테스트 영상으로 창 없이 검사
//   test_pipeline --play <영상>               실제 영상을 창으로 끝까지 처리
//   test_pipeline --play <영상> --no-display  실제 영상을 창 없이 끝까지 처리
//                                             영상을 열 수 없으면 종료 코드 2

#include "pipeline/Pipeline.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace {

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

int play(const std::string& path, bool display)
{
    PipelineOptions options;
    options.input = path;
    options.display = display;

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

    const auto dir = std::filesystem::temp_directory_path();
    const std::string path = (dir / "ppe_test_pipeline.avi").string();
    if (!writeVideo(path)) {
        std::cerr << "[ERROR] Cannot create the test video: " << path << "\n";
        return 1;
    }

    testBadPath();
    testRun(path);

    std::error_code ec;
    std::filesystem::remove(path, ec);

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
