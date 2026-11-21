// SYS-01
// #9 컴파일을 위해 명세대로 임시 작성
// 팀 최종 버전이 나오면 교체

#pragma once

#include <opencv2/core.hpp>

#include <string>

struct FrameData { cv::Mat frame; int frame_idx; double timestamp_sec; };

// box: 원본 프레임 픽셀 좌표
struct Detection { int class_id; float score; cv::Rect2f box; };

enum class Wear { WEARING, NOT_WEARING, UNKNOWN };

struct TrackState { int id; cv::Rect2f box; enum { TENTATIVE, CONFIRMED, LOST } state; };

// item: "helmet" | "vest"
struct AlertEvent {
    int event_id;
    int track_id;
    std::string item;
    int frame_idx;
    double time_sec;
    float ratio;
};
