// SYS-01 Shared data types between modules (spec section 2-3).
//
// Temporary copy written from the spec so that #9 can compile.
// To be replaced by the team's final version without changing these fields.

#pragma once

#include <opencv2/core.hpp>

#include <string>

struct FrameData { cv::Mat frame; int frame_idx; double timestamp_sec; };

// box: original frame pixel coordinates.
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
