// UI-01 단위 테스트 (src/ui/MonitorWindow)
//
// 사용법:
//   test_monitor_window                 창 없이 키 처리, FPS, 창 생성 실패, 입력 보존 검사
//   test_monitor_window --play <영상>   실제 창으로 재생 (원래 영상 속도)
//                                       스페이스 = 일시정지/재생, q 또는 ESC = 종료
//                                       영상을 열 수 없으면 종료 코드 2

#include "io/VideoSource.h"
#include "ui/MonitorWindow.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

void testKeys()
{
    check(MonitorWindow::actionForKey('q') == UiAction::Quit, "q quits");
    check(MonitorWindow::actionForKey('Q') == UiAction::Quit, "Q quits");
    check(MonitorWindow::actionForKey(27) == UiAction::Quit, "ESC quits");
    check(MonitorWindow::actionForKey(' ') == UiAction::TogglePause, "space toggles pause");
    check(MonitorWindow::actionForKey(-1) == UiAction::Continue, "no key continues");
    check(MonitorWindow::actionForKey('a') == UiAction::Continue, "other key continues");
}

void testNoDisplay()
{
    MonitorWindow window;
    window.open(false);
    check(!window.displayEnabled(), "no-display: window is not created");

    const cv::Mat frame(120, 160, CV_8UC3, cv::Scalar(10, 20, 30));
    check(window.show(frame), "no-display: show() keeps running");
    check(window.fps() == 0.0, "no-display: fps is 0 before the second frame");

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    window.show(frame);
    check(window.fps() > 0.0 && window.fps() < 200.0, "no-display: fps is measured");
    check(window.show(cv::Mat()), "no-display: empty frame is safe");
}

void testWindowFail()
{
    // 빈 창 이름은 OpenCV가 거부하므로 창 생성 실패 상황을 재현할 수 있음
    MonitorWindow window("");
    window.open(true);
    check(!window.displayEnabled(), "window failure: switches to no-display");

    const cv::Mat frame(120, 160, CV_8UC3, cv::Scalar(10, 20, 30));
    const bool first = window.show(frame);
    // 두 호출 사이 시간이 0으로 측정되면 FPS를 계산하지 않으므로 잠깐 기다림
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    check(first && window.show(frame), "window failure: processing keeps running");
    check(window.fps() > 0.0, "window failure: fps is still measured");
}

void testOverlay()
{
    const cv::Mat frame(120, 160, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat before = frame.clone();
    const cv::Mat view = MonitorWindow::withFps(frame, 12.3);

    check(cv::norm(frame, before, cv::NORM_INF) == 0.0, "FPS overlay does not change the input frame");
    check(cv::norm(view, before, cv::NORM_INF) > 0.0, "FPS is drawn on the output frame");
    check(view.size() == frame.size() && view.type() == frame.type(), "output keeps size and type");
}

int play(const std::string& path)
{
    VideoSource source;
    if (!source.open(path)) return 2;

    MonitorWindow window;
    window.open(true);

    using Clock = std::chrono::steady_clock;
    const auto frame_time = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0 / source.fps()));

    FrameData f{};
    int shown = 0;
    bool quit = false;
    auto next_time = Clock::now();
    while (source.next(f)) {
        if (!window.show(f.frame)) {
            quit = true;
            break;
        }
        ++shown;

        // 테스트에서 눈으로 볼 수 있게 원래 영상 속도에 맞춤
        next_time += frame_time;
        const auto now = Clock::now();
        if (next_time > now) {
            std::this_thread::sleep_until(next_time);
        } else {
            next_time = now;  // 일시정지 뒤 밀린 시간을 한 번에 따라잡지 않음
        }
    }
    std::cout << "[INFO] Frames shown: " << shown << "\n"
        << "[INFO] Stopped by: " << (quit ? "key or window close" : "end of video") << "\n";
    return 0;
}

}  // 익명 네임스페이스 끝

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string(argv[1]) == "--play") {
        return play(argv[2]);
    }

    testKeys();
    testNoDisplay();
    testWindowFail();
    testOverlay();

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
