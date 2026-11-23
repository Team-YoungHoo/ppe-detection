// DET-06 개발·디버깅용 탐지 결과 그리기
//
// 클래스별 색으로 박스를 그리고 박스 위에 "클래스명 점수"를 씀
// --debug-draw 옵션이 켜졌을 때만 호출함
// 착용/미착용 색(초록·빨강)은 UI-02에서 따로 처리하므로 여기서는 쓰지 않음

#pragma once

#include "common/Types.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// class_id별 디버그 색 (BGR): person 회색, helmet 파랑, vest 주황, no-helmet 자홍, no-vest 보라
// 없는 class_id이면 false를 반환함
bool debugColor(int class_id, cv::Scalar& color);

// "클래스명 점수(소수 2자리)" 글자를 만듦. 예: "person 0.87"
std::string detectionLabel(const std::string& class_name, float score);

// 탐지 결과를 그린 복사본을 반환함 (입력 프레임은 바꾸지 않음)
// 박스가 화면 밖으로 나가면 화면 안으로 잘라서 그림
cv::Mat drawDetections(const cv::Mat& frame, const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names);
