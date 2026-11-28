// SYS-05 파이프라인 (메인 루프)
//
// 순서: 영상 읽기 -> 전처리 -> 탐지 -> 추적 -> (판단·경고 자리) -> 화면 출력
// 빈 자리는 해당 기능의 이슈에서 채움

#pragma once

#include "track/Tracker.h"

#include <string>

class IDetector;

struct PipelineOptions {
    std::string input;              // 영상 파일 경로 (input.path)
    int model_size = 640;           // 모델 입력 크기 (Config 준비 전 임시 기본값)
    int pad_value = 114;            // Letterbox 패딩 값 (model.yaml)
    double scale = 1.0 / 255.0;     // 픽셀 정규화 배율 (model.yaml)
    bool swap_rb = true;            // BGR -> RGB 변환 여부 (model.yaml)
    bool display = true;            // false이면 창 없이 처리만 함
    IDetector* detector = nullptr;  // 로드가 끝난 탐지기. 없으면 탐지 단계를 건너뜀
    bool debug_draw = false;        // DET-06 탐지 결과 표시 (--debug-draw)
    int max_consecutive_fail = 30;  // pipeline.max_consecutive_fail
    TrackerOptions track;           // TRK-01~03 추적 설정 (track.*)
    std::string track_log;          // 비어 있지 않으면 프레임별 트랙을 CSV로 저장 (추적 확인용)
};

class Pipeline {
public:
    explicit Pipeline(const PipelineOptions& options);

    // 메인 루프를 실행하고 종료 코드를 반환함
    // 0 = 정상, 1 = 탐지 연속 실패, 2 = 입력 영상 오류
    int run();

    // 마지막 run()에서 끝까지 처리한 프레임 수
    int processedFrames() const { return processed_; }

    // 마지막 run()에서 탐지에 실패해 건너뛴 프레임 수
    int failedFrames() const { return failed_; }

    // 마지막 run()에서 만든 트랙 수, CONFIRMED가 된 적 있는 트랙 수 (EVL-04 확인용)
    int createdTracks() const { return created_tracks_; }
    int confirmedTracks() const { return confirmed_tracks_; }

private:
    PipelineOptions options_;
    int processed_ = 0;
    int failed_ = 0;
    int created_tracks_ = 0;
    int confirmed_tracks_ = 0;
};
