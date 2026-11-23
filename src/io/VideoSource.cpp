#include "io/VideoSource.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
const double kFallbackFps = 30.0;  // VIN-01: 파일에서 올바른 fps를 읽지 못했을 때 쓰는 값
}

bool VideoSource::open(const std::string& path, int start_frame, int frame_skip)
{
    cap_.release();
    fps_ = 0.0;
    frame_count_ = width_ = height_ = 0;

    if (!cap_.open(path) || !cap_.isOpened()) {
        // Log.h(SYS-04)가 준비되면 교체
        std::cerr << "[ERROR] Cannot open video: " << path << "\n";
        return false;
    }

    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps_) || fps_ <= 0.0) {
        std::cerr << "[WARN] Cannot read video fps, assuming " << kFallbackFps << "\n";
        fps_ = kFallbackFps;
    }
    frame_count_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "[INFO] Video: " << path << " (" << width_ << "x" << height_ << ", " << fps_
        << " fps, " << frame_count_ << " frames)\n";

    next_idx_ = std::max(0, start_frame);
    frame_skip_ = std::max(0, frame_skip);

    // grab()으로 한 장씩 넘김
    // 압축 영상에서 CAP_PROP_POS_FRAMES로 이동하면 엉뚱한 프레임에 도착해 이후 시간이 전부 밀릴 수 있음
    for (int i = 0; i < next_idx_; ++i) {
        if (!cap_.grab()) break;  // 영상 끝을 넘어가면 첫 next()가 false를 반환
    }
    return true;
}

bool VideoSource::next(FrameData& out)
{
    if (!cap_.isOpened()) return false;

    // 매번 새 Mat을 씀
    // read()가 받은 버퍼를 재사용하면 뒤 모듈이 아직 들고 있는 프레임을 덮어쓸 수 있음
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) return false;

    out.frame = frame;
    out.frame_idx = next_idx_;
    out.timestamp_sec = next_idx_ / fps_;

    for (int i = 0; i < frame_skip_; ++i) {
        if (!cap_.grab()) break;  // 영상 끝은 다음 read()가 알려 줌
    }
    next_idx_ += 1 + frame_skip_;
    return true;
}
