#include "pipeline/Pipeline.h"

#include "common/Types.h"
#include "io/VideoSource.h"
#include "preprocess/Letterbox.h"
#include "ui/MonitorWindow.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

const int kExitOk = 0;
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

    VideoSource source;
    if (!source.open(options_.input)) return kExitVideoError;

    MonitorWindow window;
    window.open(options_.display);

    std::vector<double> read_ms, preprocess_ms, display_ms;
    FrameData data{};

    while (true) {
        // 1. 영상 읽기 (VIN-01)
        const auto t_start = Clock::now();
        if (!source.next(data)) break;
        const auto t_read = Clock::now();

        // 2. 전처리 (VIN-02, VIN-03)
        const LetterboxResult input = letterbox(data.frame, options_.model_size);
        if (input.image.empty()) continue;
        const cv::Mat blob = toBlob(input.image);
        const auto t_preprocess = Clock::now();

        // 3. 탐지 자리: blob과 input.info를 넘겨 원본 좌표의 결과를 받음
        // 4. 추적 자리
        // 5. 착용 판단 자리
        // 6. 경고 자리
        // 7. 결과 그리기 자리

        // 8. 화면 출력 (UI-01). 좌표 기준인 원본 프레임을 보여 줌
        const bool keep_running = window.show(data.frame);
        const auto t_display = Clock::now();

        read_ms.push_back(elapsedMs(t_start, t_read));
        preprocess_ms.push_back(elapsedMs(t_read, t_preprocess));
        display_ms.push_back(elapsedMs(t_preprocess, t_display));
        ++processed_;

        if (!keep_running) break;
    }

    // Log.h(SYS-04)가 준비되면 교체
    std::cout << "[INFO] Processed frames: " << processed_ << "\n"
        << "[INFO] Stage time (median ms): read " << medianMs(read_ms)
        << ", preprocess " << medianMs(preprocess_ms)
        << ", display " << medianMs(display_ms) << "\n";
    return kExitOk;
}
