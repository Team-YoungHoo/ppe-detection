// VIN-01 테스트 (src/io/VideoSource)
//
// 사용법:
//   test_video_source.exe --info "<영상>"   실제 영상을 읽어 fps와 프레임 수를 출력
//   영상을 열 수 없으면 종료 코드 2

#include "io/VideoSource.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

const int kFrames = 30;
const double kFps = 10.0;

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

int brightness(int idx) { return 10 + idx * 8; }

// 이 프레임의 픽셀이 idx번 프레임에 기록한 밝기와 같으면 true
bool isFrame(const cv::Mat& frame, int idx)
{
    return std::abs(cv::mean(frame)[0] - brightness(idx)) < 3.0;
}

bool writeVideo(const std::string& path)
{
    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), kFps, cv::Size(160, 120));
    if (!writer.isOpened()) return false;
    for (int i = 0; i < kFrames; ++i) {
        const int v = brightness(i);
        writer.write(cv::Mat(120, 160, CV_8UC3, cv::Scalar(v, v, v)));
    }
    return true;
}

void testBadPath()
{
    VideoSource source;
    FrameData f{};
    check(!source.open("no_such_video.mp4"), "missing file: open() returns false");
    check(!source.next(f), "missing file: next() returns false");
}

void testReadAll(const std::string& path)
{
    VideoSource source;
    check(source.open(path), "open() succeeds");
    check(source.fps() == kFps, "fps matches the video");
    check(source.frameCount() == kFrames, "frame count matches the video");
    check(source.width() == 160 && source.height() == 120, "width and height match the video");

    FrameData f{};
    int count = 0;
    bool idx_ok = true, time_ok = true, content_ok = true;
    while (source.next(f)) {
        idx_ok = idx_ok && f.frame_idx == count;
        time_ok = time_ok && std::abs(f.timestamp_sec - count / kFps) < 1e-9;
        content_ok = content_ok && isFrame(f.frame, count);
        ++count;
    }
    check(count == kFrames, "all frames are read (" + std::to_string(count) + ")");
    check(idx_ok, "frame_idx counts from 0");
    check(time_ok, "timestamp_sec = frame_idx / fps");
    check(content_ok, "each frame has the expected content");
    check(!source.next(f), "next() stays false after the end");
}

void testStartFrame(const std::string& path)
{
    VideoSource source;
    source.open(path, 12);
    FrameData f{};
    const bool ok = source.next(f);
    check(ok && f.frame_idx == 12 && isFrame(f.frame, 12), "start_frame 12: first frame is 12");
    check(ok && std::abs(f.timestamp_sec - 1.2) < 1e-9, "start_frame 12: timestamp is 1.2 s");

    VideoSource past;
    past.open(path, kFrames + 5);
    check(!past.next(f), "start_frame past the end: next() returns false");
}

void testFrameSkip(const std::string& path)
{
    VideoSource source;
    source.open(path, 1, 2);  // 1, 4, 7, ... 번 프레임을 읽음
    FrameData f{};
    std::vector<int> got;
    bool content_ok = true;
    while (source.next(f)) {
        content_ok = content_ok && isFrame(f.frame, f.frame_idx);
        got.push_back(f.frame_idx);
    }
    bool idx_ok = got.size() == 10;
    for (size_t i = 0; idx_ok && i < got.size(); ++i) {
        idx_ok = got[i] == 1 + static_cast<int>(i) * 3;
    }
    check(idx_ok, "frame_skip 2: keeps the original frame_idx (1, 4, 7, ...)");
    check(content_ok, "frame_skip 2: content matches frame_idx");
}

void testKeptFrame(const std::string& path)
{
    VideoSource source;
    source.open(path);
    FrameData first{}, second{};
    source.next(first);            // 0번 프레임 읽기
    const FrameData kept = first;  // 뒤 모듈이 보관하는 상황처럼 얕은 복사
    source.next(second);           // 1번 프레임 읽기
    // kept가 여전히 0번 프레임인지 검사
    check(kept.frame.data != second.frame.data && isFrame(kept.frame, 0),
        "a kept frame is not overwritten by next()");
}

int info(const std::string& path)
{
    VideoSource source;
    if (!source.open(path)) return 2;

    FrameData f{};
    int count = 0;
    double last_time = 0.0;
    while (source.next(f)) {
        last_time = f.timestamp_sec;
        ++count;
    }
    std::cout << "[INFO] fps: " << source.fps() << "\n"
        << "[INFO] frames (file metadata): " << source.frameCount() << "\n"
        << "[INFO] frames (actually read): " << count << "\n"
        << "[INFO] last frame time: " << last_time << " s\n";
    return 0;
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string(argv[1]) == "--info") {
        return info(argv[2]);
    }

    const auto dir = std::filesystem::temp_directory_path();
    const std::string path = (dir / "ppe_test_video_source.avi").string();
    if (!writeVideo(path)) {
        std::cerr << "[ERROR] Cannot create the test video: " << path << "\n";
        return 1;
    }

    testBadPath();
    testReadAll(path);
    testStartFrame(path);
    testFrameSkip(path);
    testKeptFrame(path);

    std::error_code ec;
    std::filesystem::remove(path, ec);

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
