#include "pipeline/Pipeline.h"

#include "common/Types.h"
#include "detect/IDetector.h"
#include "io/VideoSource.h"
#include "preprocess/Letterbox.h"
#include "ui/DebugDraw.h"
#include "ui/MonitorWindow.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

const int kExitOk = 0;
const int kExitUnknownError = 1;
const int kExitVideoError = 2;

double elapsedMs(Clock::time_point from, Clock::time_point to)
{
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// 일시정지처럼 한 번 튄 값에 흔들리지 않도록 중앙값을 씀
double medianMs(std::vector<double>& values)
{
    if (values.empty()) return 0.0;
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

}  // namespace 끝

Pipeline::Pipeline(const PipelineOptions& options) : options_(options)
{
}

int Pipeline::run()
{
    processed_ = 0;
    failed_ = 0;

    VideoSource source;
    if (!source.open(options_.input)) return kExitVideoError;

    MonitorWindow window;
    window.open(options_.display);

    std::vector<double> read_ms, preprocess_ms, detect_ms, display_ms;
    FrameData data{};
    int consecutive_fail = 0;
    int exit_code = kExitOk;
    const auto t_begin = Clock::now();

    while (true) {
        // 1. 영상 읽기 (VIN-01)
        const auto t_start = Clock::now();
        if (!source.next(data)) break;
        const auto t_read = Clock::now();

        // 2. 전처리 (VIN-02, VIN-03)
        const LetterboxResult input = letterbox(data.frame, options_.model_size, options_.pad_value);
        if (input.image.empty()) continue;
        const cv::Mat blob = toBlob(input.image, options_.scale, options_.swap_rb);
        const auto t_preprocess = Clock::now();

        // 3. 탐지 + 신뢰도 필터 + NMS + 좌표 역변환 (DET-02~04)
        // 실패하면 그 프레임을 건너뛰고, 연속으로 실패하면 종료 코드 1로 끝냄
        std::vector<Detection> detections;
        if (options_.detector != nullptr) {
            try {
                detections = options_.detector->detect(blob, input.info);
                consecutive_fail = 0;
            } catch (const std::exception& e) {
                ++failed_;
                ++consecutive_fail;
                std::cerr << "[WARN] Detection failed at frame " << data.frame_idx << ": " << e.what() << "\n";
                if (consecutive_fail >= options_.max_consecutive_fail) {
                    std::cerr << "[ERROR] Detection failed " << consecutive_fail << " frames in a row\n";
                    exit_code = kExitUnknownError;
                    break;
                }
                continue;
            }
        }
        const auto t_detect = Clock::now();

        // 4. 탐지 결과 분류 자리
        // 5. 추적 자리
        // 6. 착용 판단 자리
        // 7. 경고 자리

        // 8. 화면 출력 (UI-01). 디버그 표시는 --debug-draw일 때만 그림 (DET-06)
        const bool draw = options_.display && options_.debug_draw && options_.detector != nullptr;
        const cv::Mat shown = draw
            ? drawDetections(data.frame, detections, options_.detector->classNames())
            : data.frame;
        const bool keep_running = window.show(shown);
        const auto t_display = Clock::now();

        // 9. 단계별 소요 시간 기록
        read_ms.push_back(elapsedMs(t_start, t_read));
        preprocess_ms.push_back(elapsedMs(t_read, t_preprocess));
        detect_ms.push_back(elapsedMs(t_preprocess, t_detect));
        display_ms.push_back(elapsedMs(t_detect, t_display));
        ++processed_;

        if (!keep_running) break;
    }

    const double total_sec = elapsedMs(t_begin, Clock::now()) / 1000.0;

    // Log.h(SYS-04)가 준비되면 교체
    std::cout << "[INFO] Processed frames: " << processed_ << "\n"
        << "[INFO] Failed frames: " << failed_ << "\n"
        << "[INFO] Average FPS: " << (total_sec > 0.0 ? processed_ / total_sec : 0.0) << "\n"
        << "[INFO] Stage time (median ms): read " << medianMs(read_ms)
        << ", preprocess " << medianMs(preprocess_ms)
        << ", detect " << medianMs(detect_ms)
        << ", display " << medianMs(display_ms) << "\n";
    return exit_code;
}
