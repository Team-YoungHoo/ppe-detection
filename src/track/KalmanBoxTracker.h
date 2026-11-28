// TRK-04 칼만 필터 박스 추적기 (SORT 방식)
//
// 상태 7차원 [cx, cy, s, r, vx, vy, vs], 측정 4차원 [cx, cy, s, r]
//   cx, cy = 박스 중심, s = 면적, r = 가로/세로 비 (r은 일정하다고 가정해 속도 없음)
// 행렬과 노이즈 값은 SORT 원본(Bewley et al., 2016)과 같음
// 좌표는 모두 원본 프레임 픽셀

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

class KalmanBoxTracker {
public:
    // 첫 탐지 박스로 상태를 초기화함 (속도는 0)
    explicit KalmanBoxTracker(const cv::Rect2f& box);

    // 한 프레임 뒤의 상태를 예측하고 예측 박스를 반환함
    // s 또는 r이 0 이하(또는 NaN)로 예측되면 invalid()가 true가 됨
    cv::Rect2f predict();

    // 탐지 박스로 상태를 보정함
    void update(const cv::Rect2f& box);

    // 현재 상태의 박스 (predict 뒤에는 예측 박스, update 뒤에는 보정된 박스)
    cv::Rect2f box() const;

    // 삭제 대상 여부 (상태가 박스로 바꿀 수 없는 값이 됨)
    bool invalid() const { return invalid_; }

    // 박스 <-> 측정 벡터 [cx, cy, s, r] 변환 (테스트에서도 씀)
    static cv::Mat toMeasurement(const cv::Rect2f& box);
    static cv::Rect2f toBox(double cx, double cy, double s, double r);

private:
    void checkState();

    cv::KalmanFilter kf_;
    bool invalid_ = false;
};
