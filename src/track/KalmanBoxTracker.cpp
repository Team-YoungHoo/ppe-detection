#include "track/KalmanBoxTracker.h"

#include <cmath>
#include <limits>

namespace {

const int kStateDim = 7;
const int kMeasureDim = 4;

// SORT 원본 노이즈 값
const double kMeasureNoisePos = 1.0;      // R: cx, cy
const double kMeasureNoiseShape = 10.0;   // R: s, r
const double kInitCovPos = 10.0;          // P: 위치·모양 (관측된 값)
const double kInitCovVel = 10000.0;       // P: 속도 (관측 안 된 값이라 크게 둠)
const double kProcessNoisePos = 1.0;      // Q: cx, cy, s, r
const double kProcessNoiseVel = 0.01;     // Q: vx, vy
const double kProcessNoiseArea = 0.0001;  // Q: vs

}  // namespace 끝

KalmanBoxTracker::KalmanBoxTracker(const cv::Rect2f& box)
    : kf_(kStateDim, kMeasureDim, 0, CV_64F)
{
    // 등속 모델: cx += vx, cy += vy, s += vs
    cv::setIdentity(kf_.transitionMatrix);
    kf_.transitionMatrix.at<double>(0, 4) = 1.0;
    kf_.transitionMatrix.at<double>(1, 5) = 1.0;
    kf_.transitionMatrix.at<double>(2, 6) = 1.0;

    kf_.measurementMatrix = cv::Mat::zeros(kMeasureDim, kStateDim, CV_64F);
    for (int i = 0; i < kMeasureDim; ++i) kf_.measurementMatrix.at<double>(i, i) = 1.0;

    kf_.measurementNoiseCov = cv::Mat::diag((cv::Mat_<double>(kMeasureDim, 1)
        << kMeasureNoisePos, kMeasureNoisePos, kMeasureNoiseShape, kMeasureNoiseShape));
    kf_.errorCovPost = cv::Mat::diag((cv::Mat_<double>(kStateDim, 1)
        << kInitCovPos, kInitCovPos, kInitCovPos, kInitCovPos, kInitCovVel, kInitCovVel, kInitCovVel));
    kf_.processNoiseCov = cv::Mat::diag((cv::Mat_<double>(kStateDim, 1)
        << kProcessNoisePos, kProcessNoisePos, kProcessNoisePos, kProcessNoisePos,
        kProcessNoiseVel, kProcessNoiseVel, kProcessNoiseArea));

    kf_.statePost = cv::Mat::zeros(kStateDim, 1, CV_64F);
    toMeasurement(box).copyTo(kf_.statePost.rowRange(0, kMeasureDim));
    checkState();
}

cv::Rect2f KalmanBoxTracker::predict()
{
    // OpenCV의 predict()는 예측값을 statePost에도 복사하므로 box()가 예측 박스를 돌려줌
    kf_.predict();
    checkState();
    return box();
}

void KalmanBoxTracker::update(const cv::Rect2f& box)
{
    kf_.correct(toMeasurement(box));
    checkState();
}

cv::Rect2f KalmanBoxTracker::box() const
{
    const cv::Mat& x = kf_.statePost;
    return toBox(x.at<double>(0), x.at<double>(1), x.at<double>(2), x.at<double>(3));
}

cv::Mat KalmanBoxTracker::toMeasurement(const cv::Rect2f& box)
{
    const double w = box.width;
    const double h = box.height;
    const double r = h > 0.0 ? w / h : 0.0;
    return (cv::Mat_<double>(kMeasureDim, 1) << box.x + w / 2.0, box.y + h / 2.0, w * h, r);
}

cv::Rect2f KalmanBoxTracker::toBox(double cx, double cy, double s, double r)
{
    if (!(s > 0.0) || !(r > 0.0)) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        return cv::Rect2f(nan, nan, nan, nan);
    }
    const double w = std::sqrt(s * r);
    const double h = s / w;
    return cv::Rect2f(static_cast<float>(cx - w / 2.0), static_cast<float>(cy - h / 2.0),
        static_cast<float>(w), static_cast<float>(h));
}

void KalmanBoxTracker::checkState()
{
    const cv::Mat& x = kf_.statePost;
    // NaN이면 비교가 모두 false라서 !(> 0)으로 같이 걸러짐
    const bool finite = cv::checkRange(x, true, nullptr, -1e12, 1e12);
    if (!finite || !(x.at<double>(2) > 0.0) || !(x.at<double>(3) > 0.0)) invalid_ = true;
}
