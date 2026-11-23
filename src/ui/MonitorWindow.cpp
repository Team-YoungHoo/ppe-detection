#include "ui/MonitorWindow.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <iostream>

namespace {
const int kEscKey = 27;
const int kPauseWaitMs = 30;     // 일시정지 중 키를 확인하는 간격
const double kFpsSmooth = 0.9;   // FPS 숫자가 튀지 않게 이전 값을 반영하는 비율
}

MonitorWindow::MonitorWindow(const std::string& window_name, bool show_fps)
    : window_name_(window_name), show_fps_(show_fps)
{
}

MonitorWindow::~MonitorWindow()
{
    close();
}

void MonitorWindow::open(bool display)
{
    close();
    has_tick_ = false;
    has_fps_ = false;
    fps_ = 0.0;
    paused_sec_ = 0.0;

    if (!display) return;

    try {
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        display_ = true;
    } catch (const cv::Exception&) {
        // Log.h(SYS-04)가 준비되면 교체
        std::cerr << "[WARN] Cannot create window, switching to no-display mode\n";
        display_ = false;
    }
}

bool MonitorWindow::show(const cv::Mat& frame)
{
    updateFps();
    if (!display_ || frame.empty()) return true;

    cv::imshow(window_name_, show_fps_ ? withFps(frame, fps_) : frame);

    const UiAction action = actionForKey(cv::waitKey(1));
    if (action == UiAction::Quit) return false;
    if (!isWindowOpen()) return false;  // X 버튼으로 창을 닫은 경우
    if (action == UiAction::TogglePause) return waitWhilePaused();
    return true;
}

void MonitorWindow::close()
{
    if (!display_) return;
    display_ = false;
    try {
        cv::destroyWindow(window_name_);
    } catch (const cv::Exception&) {
        // 사용자가 이미 창을 닫았으면 지울 창이 없음
    }
}

UiAction MonitorWindow::actionForKey(int key)
{
    if (key == kEscKey || key == 'q' || key == 'Q') return UiAction::Quit;
    if (key == ' ') return UiAction::TogglePause;
    return UiAction::Continue;
}

cv::Mat MonitorWindow::withFps(const cv::Mat& frame, double fps)
{
    cv::Mat view = frame.clone();

    char text[64];
    std::snprintf(text, sizeof(text), "FPS: %.1f", fps);
    cv::putText(view, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 0, 0), 4);     // 글자 테두리
    cv::putText(view, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 255, 0), 2);   // 글자
    return view;
}

void MonitorWindow::updateFps()
{
    const Clock::time_point now = Clock::now();
    if (has_tick_) {
        const double dt = std::chrono::duration<double>(now - last_tick_).count();
        if (dt > 0.0) {
            const double inst = 1.0 / dt;
            fps_ = has_fps_ ? kFpsSmooth * fps_ + (1.0 - kFpsSmooth) * inst : inst;
            has_fps_ = true;
        }
    }
    last_tick_ = now;
    has_tick_ = true;
}

bool MonitorWindow::waitWhilePaused()
{
    const Clock::time_point pause_start = Clock::now();
    bool keep_running = true;
    while (true) {
        const UiAction action = actionForKey(cv::waitKey(kPauseWaitMs));
        if (action == UiAction::Quit || !isWindowOpen()) {
            keep_running = false;
            break;
        }
        if (action == UiAction::TogglePause) break;
    }
    // 멈춰 있던 시간은 처리 시간이 아니므로 FPS 계산에서 뺌
    const Clock::duration paused = Clock::now() - pause_start;
    last_tick_ += paused;
    paused_sec_ += std::chrono::duration<double>(paused).count();
    return keep_running;
}

bool MonitorWindow::isWindowOpen() const
{
    try {
        return cv::getWindowProperty(window_name_, cv::WND_PROP_VISIBLE) >= 1;
    } catch (const cv::Exception&) {
        return false;
    }
}
