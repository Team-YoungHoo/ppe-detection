// TRK-03 추적 대상 한 명의 내부 값
//
// 바깥(판단·경고·화면)에는 common/Types.h의 TrackState만 넘기고
// 칼만 필터와 횟수 값은 Tracker 안에서만 씀

#pragma once

#include "common/Types.h"
#include "track/KalmanBoxTracker.h"

using TrackStatus = decltype(TrackState::state);

struct Track {
    int id;
    KalmanBoxTracker kf;
    int hits = 1;                // 매칭된 프레임 수 (생성한 프레임 포함)
    int time_since_update = 0;   // 마지막 매칭 뒤 지난 프레임 수
    TrackStatus state = TrackState::TENTATIVE;
    bool ever_confirmed = false; // CONFIRMED가 된 적이 있는지 (통계용)

    Track(int track_id, const cv::Rect2f& box) : id(track_id), kf(box) {}
};
