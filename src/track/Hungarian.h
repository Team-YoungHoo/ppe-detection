// TRK-02 헝가리안 알고리즘과 IoU
//
// 외부 코드 없이 직접 구현함 (라이선스 문제 방지)
// 퍼텐셜(potential)을 쓰는 O(n^2 m) 방식이며 행과 열 수가 달라도 됨

#pragma once

#include <opencv2/core.hpp>

#include <vector>

// 두 박스의 IoU (교집합 넓이 / 합집합 넓이). 넓이가 0 이하이거나 NaN이면 0
float iou(const cv::Rect2f& a, const cv::Rect2f& b);

// 비용 합이 최소가 되도록 행과 열을 1:1로 짝지음
// cost[i][j] = i행을 j열에 짝지을 때의 비용. 모든 행의 길이가 같아야 함
// 반환: 행 i의 짝 열 번호. 짝이 없으면 -1 (행이 열보다 많으면 일부 행은 짝이 없음)
// 빈 행렬이면 빈 벡터를 반환함
std::vector<int> solveAssignment(const std::vector<std::vector<double>>& cost);
