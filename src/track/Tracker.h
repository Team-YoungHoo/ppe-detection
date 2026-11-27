// TRK-01~03 작업자 추적 (SORT 방식: 칼만 필터 + 헝가리안 + IoU)
//
// 매 프레임 person 박스를 받아 추적 ID를 붙임
//   1. 모든 트랙 예측 (TRK-04)
//   2. 예측 박스와 탐지 박스의 IoU 행렬, 비용 1 - IoU로 헝가리안 매칭 (TRK-02)
//   3. IoU가 iou_threshold보다 낮은 짝은 매칭에서 뺌
//   4. 매칭된 트랙 보정, 남은 탐지로 새 트랙(TENTATIVE) 생성 (TRK-01)
//   5. 상태 전이와 삭제 (TRK-03)
//      TENTATIVE --hits >= min_hits--> CONFIRMED --미매칭--> LOST --재매칭--> CONFIRMED
//      TENTATIVE가 한 번이라도 미매칭이면 바로 삭제
//      LOST가 max_age 프레임을 넘게 미매칭이면 삭제
//      예측 박스가 NaN·넓이 0 이하·화면 밖으로 완전히 나가면 삭제
// 좌표는 모두 원본 프레임 픽셀

#pragma once

#include "common/Types.h"
#include "track/Track.h"

#include <opencv2/core.hpp>

#include <vector>

// 설정값 (configs/default.yaml의 track: 섹션과 같은 기본값)
struct TrackerOptions {
    int start_id = 1;            // track.start_id
    float iou_threshold = 0.3f;  // track.iou_threshold
    int min_hits = 3;            // track.min_hits
    int max_age = 30;            // track.max_age
};

class Tracker {
public:
    explicit Tracker(const TrackerOptions& options = TrackerOptions());

    // 화면 크기를 알려 주면 예측 박스가 화면 밖으로 완전히 나간 트랙을 지움
    // 설정하지 않으면(빈 크기) 화면 밖 검사를 하지 않음
    void setFrameSize(const cv::Size& size) { frame_size_ = size; }

    // person 박스(class_id 0)만 넘김. 다른 클래스가 섞여 있어도 그대로 박스로 씀
    // frame_idx가 이전 호출보다 2 이상 커지면(실패로 건너뛴 프레임) 그만큼 더 예측하고 나이를 셈
    // 반환: 남아 있는 모든 트랙 (TENTATIVE·CONFIRMED·LOST)
    std::vector<TrackState> update(const std::vector<Detection>& person_boxes, int frame_idx);

    // 모든 트랙을 지우고 ID를 start_id부터 다시 셈
    void reset();

    // 통계 (EVL-04 확인용): 지금까지 만든 트랙 수, CONFIRMED가 된 적 있는 트랙 수
    int createdTracks() const { return created_; }
    int confirmedTracks() const { return confirmed_; }

private:
    bool outOfFrame(const cv::Rect2f& box) const;

    TrackerOptions options_;
    std::vector<Track> tracks_;
    cv::Size frame_size_;
    int next_id_;
    int last_frame_idx_ = -1;
    int created_ = 0;
    int confirmed_ = 0;
};
