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
#include <fstream>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

const int kExitOk = 0;
const int kExitUnknownError = 1;
const int kExitVideoError = 2;

const int kPersonClassId = 0;  // classes.txt 첫 줄 (순서 변경 금지)

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
    created_tracks_ = 0;
    confirmed_tracks_ = 0;

    VideoSource source;
    if (!source.open(options_.input)) return kExitVideoError;

    MonitorWindow window;
    window.open(options_.display);

    Tracker tracker(options_.track);
    std::ofstream track_log;
    if (!options_.track_log.empty()) {
        track_log.open(options_.track_log);
        if (track_log) track_log << "frame,id,state,x,y,w,h\n";
        else std::cerr << "[WARN] Cannot open track log: " << options_.track_log << "\n";
    }

    std::vector<double> read_ms, preprocess_ms, detect_ms, track_ms, display_ms;
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

        // 4. 탐지 결과 분류: 추적에는 person(class_id 0) 박스만 넘김
        //    착용 판단(JDG)용 helmet·vest 분류는 해당 이슈에서 채움
        std::vector<Detection> persons;
        for (const Detection& d : detections) {
            if (d.class_id == kPersonClassId) persons.push_back(d);
        }

        // 5. 추적 (TRK-01~04)
        tracker.setFrameSize(data.frame.size());
        const std::vector<TrackState> tracks = tracker.update(persons, data.frame_idx);
        if (track_log) {
            static const char* kStateNames[] = { "TENTATIVE", "CONFIRMED", "LOST" };
            for (const TrackState& t : tracks) {
                track_log << data.frame_idx << ',' << t.id << ',' << kStateNames[t.state] << ','
                    << t.box.x << ',' << t.box.y << ',' << t.box.width << ',' << t.box.height << '\n';
            }
        }
        const auto t_track = Clock::now();

        // 6. 착용 판단 자리
        // 7. 경고 자리

        // 8. 화면 출력 (UI-01). 디버그 표시는 --debug-draw일 때만 그림 (DET-06)
        //    CONFIRMED 트랙 ID는 확인용 임시 표시 (정식 표시는 UI-03)
        const bool draw = options_.display && options_.debug_draw && options_.detector != nullptr;
        const cv::Mat shown = draw
            ? drawTrackIds(drawDetections(data.frame, detections, options_.detector->classNames()), tracks)
            : data.frame;
        const bool keep_running = window.show(shown);
        const auto t_display = Clock::now();

        // 9. 단계별 소요 시간 기록
        read_ms.push_back(elapsedMs(t_start, t_read));
        preprocess_ms.push_back(elapsedMs(t_read, t_preprocess));
        detect_ms.push_back(elapsedMs(t_preprocess, t_detect));
        track_ms.push_back(elapsedMs(t_detect, t_track));
        display_ms.push_back(elapsedMs(t_track, t_display));
        ++processed_;

        if (!keep_running) break;
    }

    created_tracks_ = tracker.createdTracks();
    confirmed_tracks_ = tracker.confirmedTracks();

    // 일시정지한 시간은 처리 시간이 아니므로 평균 FPS에서 뺌
    const double total_sec = elapsedMs(t_begin, Clock::now()) / 1000.0 - window.pausedSeconds();

    // Log.h(SYS-04)가 준비되면 교체
    std::cout << "[INFO] Processed frames: " << processed_ << "\n"
        << "[INFO] Failed frames: " << failed_ << "\n"
        << "[INFO] Tracks: created " << created_tracks_ << ", confirmed " << confirmed_tracks_ << "\n"
        << "[INFO] Average FPS: " << (total_sec > 0.0 ? processed_ / total_sec : 0.0) << "\n"
        << "[INFO] Stage time (median ms): read " << medianMs(read_ms)
        << ", preprocess " << medianMs(preprocess_ms)
        << ", detect " << medianMs(detect_ms)
        << ", track " << medianMs(track_ms)
        << ", display " << medianMs(display_ms) << "\n";
    return exit_code;
}
