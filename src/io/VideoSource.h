// VIN-01 녹화된 영상 파일을 프레임 단위로 읽음
// 모든 프레임에 frame_idx와 영상 시간(timestamp_sec = frame_idx / fps)을 붙임
// 벽시계가 아니라 영상 시간을 쓰므로 처리 속도가 달라도 지속 시간 결과는 같음

#pragma once

#include "common/Types.h"

#include <opencv2/videoio.hpp>

#include <string>

class VideoSource {
public:
    // 파일을 열 수 없으면 false 반환 & 호출한 쪽에서 종료 코드 2로 끝냄
    // start_frame: 처음 읽을 프레임 번호 (input.start_frame, 기본값 0)
    // frame_skip : 한 장 읽을 때마다 건너뛸 프레임 수 (input.frame_skip, 0이면 전부 읽음)
    bool open(const std::string& path, int start_frame = 0, int frame_skip = 0);

    // 다음 프레임을 읽음. 영상이 끝났거나 빈 프레임이면 false를 반환
    // (손상된 영상과 정상 종료는 구분하지 않음)
    // 프레임마다 픽셀을 따로 가지므로, 보관한 FrameData는 next() 후에도 유효함
    bool next(FrameData& out);

    double fps() const { return fps_; }
    int frameCount() const { return frame_count_; }  // 파일 정보 기준이라 근사값일 수 있음
    int width() const { return width_; }
    int height() const { return height_; }

private:
    cv::VideoCapture cap_;
    double fps_ = 0.0;
    int frame_count_ = 0;
    int width_ = 0;
    int height_ = 0;
    int next_idx_ = 0;
    int frame_skip_ = 0;
};
