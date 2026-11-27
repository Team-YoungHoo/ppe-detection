// TRK-02 단위 테스트 (src/track/Hungarian)
//
// 사용법: test_hungarian
// 손으로 계산한 IoU와 배정 예제로 검사함 (모델 파일 불필요)

#include "track/Hungarian.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

bool near(double a, double b) { return std::abs(a - b) < 1e-5; }

void testIou()
{
    const cv::Rect2f a(0, 0, 10, 10);
    check(near(iou(a, a), 1.0), "iou: same box is 1");
    // 교집합 5x5 = 25, 합집합 100 + 100 - 25 = 175
    check(near(iou(a, cv::Rect2f(5, 5, 10, 10)), 25.0 / 175.0), "iou: half shifted box is 1/7");
    check(near(iou(a, cv::Rect2f(20, 0, 10, 10)), 0.0), "iou: separate boxes are 0");
    check(near(iou(a, cv::Rect2f(10, 0, 10, 10)), 0.0), "iou: touching boxes are 0");
    check(near(iou(a, cv::Rect2f(2, 2, 0, 5)), 0.0), "iou: zero area box is 0");
    // 안에 들어간 박스: 교집합 4, 합집합 100
    check(near(iou(a, cv::Rect2f(1, 1, 2, 2)), 4.0 / 100.0), "iou: inner box is 4/100");
    const float nan = std::nanf("");
    check(near(iou(a, cv::Rect2f(nan, 0, 10, 10)), 0.0), "iou: NaN box is 0");
}

void testAssignment()
{
    // 예제 1 (3x3): 가장 싼 칸(1행 1열 = 0)을 먼저 고르는 탐욕 방식은 6이 나오지만 최적은 5
    //   행0->열1(1) + 행1->열0(2) + 행2->열2(2) = 5
    const auto r1 = solveAssignment({ { 4, 1, 3 }, { 2, 0, 5 }, { 3, 2, 2 } });
    check(r1 == std::vector<int>({ 1, 0, 2 }), "hungarian 3x3: total cost 5");

    // 예제 2 (2x3, 열이 더 많음): 행0->열1(2) + 행1->열2(1) = 3, 열0은 남음
    const auto r2 = solveAssignment({ { 10, 2, 8 }, { 3, 9, 1 } });
    check(r2 == std::vector<int>({ 1, 2 }), "hungarian 2x3: total cost 3");

    // 예제 3 (3x2, 행이 더 많음): 행0->열1(1) + 행1->열0(1) = 2, 행2는 짝 없음
    const auto r3 = solveAssignment({ { 5, 1 }, { 1, 5 }, { 4, 4 } });
    check(r3 == std::vector<int>({ 1, 0, -1 }), "hungarian 3x2: extra row is -1");

    check(solveAssignment({}).empty(), "hungarian: empty matrix");
    check(solveAssignment({ {}, {} }) == std::vector<int>({ -1, -1 }), "hungarian: no columns");
    check(solveAssignment({ { 0.7 } }) == std::vector<int>({ 0 }), "hungarian: 1x1");
}

void testAssociate()
{
    // 트랙 0은 탐지 1과 겹치고, 트랙 1은 탐지 0과 조금(IoU 1/7 < 0.3) 겹치며, 탐지 2는 혼자 있음
    const std::vector<cv::Rect2f> tracks = { cv::Rect2f(0, 0, 10, 10), cv::Rect2f(100, 100, 10, 10) };
    const std::vector<cv::Rect2f> dets = { cv::Rect2f(105, 105, 10, 10), cv::Rect2f(1, 0, 10, 10),
        cv::Rect2f(300, 300, 10, 10) };
    const MatchResult m = associate(tracks, dets, 0.3f);
    check(m.matches == std::vector<std::pair<int, int>>({ { 0, 1 } }), "associate: matched pair (0, 1)");
    check(m.unmatched_tracks == std::vector<int>({ 1 }), "associate: low IoU track is unmatched");
    check(m.unmatched_detections == std::vector<int>({ 0, 2 }), "associate: unmatched detections 0 and 2");

    const MatchResult no_tracks = associate({}, dets, 0.3f);
    check(no_tracks.matches.empty() && no_tracks.unmatched_tracks.empty()
        && no_tracks.unmatched_detections == std::vector<int>({ 0, 1, 2 }), "associate: no tracks");
    const MatchResult no_dets = associate(tracks, {}, 0.3f);
    check(no_dets.matches.empty() && no_dets.unmatched_tracks == std::vector<int>({ 0, 1 })
        && no_dets.unmatched_detections.empty(), "associate: no detections");
}

}  // namespace 끝

int main()
{
    testIou();
    testAssignment();
    testAssociate();
    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
