// TRK-04 단위 테스트 (src/track/KalmanBoxTracker)
//
// 사용법: test_kalman_box_tracker
// 일정 속도로 움직이는 가짜 박스로 예측 오차가 줄어드는지 검사함

#include "track/KalmanBoxTracker.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

// 왼쪽 위 꼭짓점 기준 거리 + 크기 차이
double boxError(const cv::Rect2f& a, const cv::Rect2f& b)
{
    return std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.width - b.width)
        + std::abs(a.height - b.height);
}

void testConversion()
{
    const cv::Rect2f box(10, 20, 50, 100);
    const cv::Mat z = KalmanBoxTracker::toMeasurement(box);
    check(std::abs(z.at<double>(0) - 35.0) < 1e-9 && std::abs(z.at<double>(1) - 70.0) < 1e-9
        && std::abs(z.at<double>(2) - 5000.0) < 1e-9 && std::abs(z.at<double>(3) - 0.5) < 1e-9,
        "box -> [cx, cy, s, r]");
    const cv::Rect2f back = KalmanBoxTracker::toBox(35.0, 70.0, 5000.0, 0.5);
    check(boxError(back, box) < 1e-3, "[cx, cy, s, r] -> box");
    check(std::isnan(KalmanBoxTracker::toBox(0, 0, 0, 1).width), "s <= 0 gives NaN box");
    check(std::isnan(KalmanBoxTracker::toBox(0, 0, 100, -1).width), "r <= 0 gives NaN box");

    KalmanBoxTracker kf(box);
    check(boxError(kf.box(), box) < 1e-3 && !kf.invalid(), "initial state is the first box");
}

void testConstantVelocity()
{
    // 프레임마다 x +5, y +2 로 움직이고 크기는 그대로
    auto truth = [](int k) { return cv::Rect2f(100.0f + 5.0f * k, 50.0f + 2.0f * k, 40.0f, 90.0f); };

    KalmanBoxTracker kf(truth(0));
    double early = 0.0, late = 0.0;
    for (int k = 1; k <= 30; ++k) {
        const double err = boxError(kf.predict(), truth(k));
        if (k <= 3) early += err;
        if (k > 27) late += err;
        kf.update(truth(k));
    }
    std::cout << "  prediction error: first 3 frames " << early << ", last 3 frames " << late << "\n";
    check(late < early, "constant velocity: prediction error goes down");
    check(late / 3.0 < 1.0, "constant velocity: late prediction error < 1 px");

    // 탐지 없이 예측만 해도 같은 속도로 계속 움직임
    cv::Rect2f p;
    for (int k = 31; k <= 35; ++k) p = kf.predict();
    check(boxError(p, truth(35)) < 5.0, "constant velocity: coasting keeps the motion");
    check(!kf.invalid(), "constant velocity: state stays valid");
}

void testInvalid()
{
    // 넓이가 빠르게 줄어드는 박스: 예측만 계속하면 s가 0 이하가 됨
    KalmanBoxTracker kf(cv::Rect2f(0, 0, 100, 100));
    for (float size : { 80.0f, 60.0f, 40.0f, 20.0f }) {
        kf.predict();
        kf.update(cv::Rect2f(0, 0, size, size));
    }
    bool became_invalid = false;
    for (int k = 0; k < 10 && !became_invalid; ++k) {
        kf.predict();
        became_invalid = kf.invalid();
    }
    check(became_invalid, "shrinking box: s <= 0 marks the tracker invalid");
}

}  // namespace 끝

int main()
{
    testConversion();
    testConstantVelocity();
    testInvalid();
    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
