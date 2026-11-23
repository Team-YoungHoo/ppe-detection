// UI-01 처리 결과 영상을 창에 보여 줌 (OpenCV highgui만 사용)
//
// 키 입력: q 또는 ESC = 종료, 스페이스 = 일시정지/재생
// --no-display일 때는 창 없이 FPS만 계산함 (평가·벤치마크용)

#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <string>

enum class UiAction { Continue, TogglePause, Quit };

class MonitorWindow {
public:
    explicit MonitorWindow(const std::string& window_name = "Safety Monitor", bool show_fps = true);
    ~MonitorWindow();

    // 창은 하나만 관리하므로 복사하지 않음
    MonitorWindow(const MonitorWindow&) = delete;
    MonitorWindow& operator=(const MonitorWindow&) = delete;

    // display = false이면 창을 만들지 않음
    // 창 생성에 실패하면 경고를 출력하고 창 없는 모드로 전환함
    void open(bool display = true);

    // 프레임 한 장을 보여 주고 키를 확인함
    // 종료 키를 눌렀거나 창을 닫았으면 false를 반환함
    bool show(const cv::Mat& frame);

    void close();

    bool displayEnabled() const { return display_; }
    double fps() const { return fps_; }

    // 스페이스로 멈춰 있던 전체 시간(초). 평균 FPS 계산에서 뺌
    double pausedSeconds() const { return paused_sec_; }

    // 눌린 키를 동작으로 바꿈
    static UiAction actionForKey(int key);

    // 왼쪽 위에 FPS를 그린 복사본을 반환함 (입력 프레임은 바꾸지 않음)
    static cv::Mat withFps(const cv::Mat& frame, double fps);

private:
    using Clock = std::chrono::steady_clock;

    void updateFps();
    bool waitWhilePaused();
    bool isWindowOpen() const;

    std::string window_name_;
    bool show_fps_;
    bool display_ = false;
    double fps_ = 0.0;
    double paused_sec_ = 0.0;
    bool has_tick_ = false;
    bool has_fps_ = false;
    Clock::time_point last_tick_;
};
