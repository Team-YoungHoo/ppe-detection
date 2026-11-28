// TRK-01~03 단위 테스트 (src/track/Tracker)
//
// 사용법: test_tracker
// 코드로 만든 person 박스로 ID 부여·상태 전이·삭제 규칙을 검사함 (모델 파일 불필요)

#include "track/Tracker.h"

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

Detection person(float x, float y, float w = 40.0f, float h = 100.0f)
{
    return { 0, 0.9f, cv::Rect2f(x, y, w, h) };
}

const TrackState* find(const std::vector<TrackState>& tracks, int id)
{
    for (const auto& t : tracks) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

void testIdsAndMinHits()
{
    Tracker tracker;
    std::vector<TrackState> out;
    for (int f = 0; f < 5; ++f) {
        out = tracker.update({ person(100.0f + 2.0f * f, 100.0f), person(400.0f - 2.0f * f, 120.0f) }, f);
        if (f < 2) {
            check(out.size() == 2 && out[0].state == TrackState::TENTATIVE && out[1].state == TrackState::TENTATIVE,
                "min_hits: frame " + std::to_string(f) + " is TENTATIVE");
        }
        if (f == 2) {
            check(out.size() == 2 && out[0].state == TrackState::CONFIRMED && out[1].state == TrackState::CONFIRMED,
                "min_hits: 3rd matched frame is CONFIRMED");
        }
    }
    check(find(out, 1) && find(out, 2) && out.size() == 2, "ids: two people get id 1 and 2");
    check(find(out, 1)->box.x < 200.0f && find(out, 2)->box.x > 300.0f, "ids: each id keeps its person");
    check(tracker.createdTracks() == 2 && tracker.confirmedTracks() == 2, "ids: stats count 2 tracks");
}

void testTentativeDeleted()
{
    Tracker tracker;
    tracker.update({ person(100, 100) }, 0);
    tracker.update({ person(101, 100) }, 1);
    const auto out = tracker.update({}, 2);
    check(out.empty(), "TENTATIVE: deleted after one miss");

    const auto next = tracker.update({ person(100, 100) }, 3);
    check(next.size() == 1 && next[0].id == 2, "ids: deleted id is not reused");
}

void testNoDetections()
{
    Tracker tracker;
    check(tracker.update({}, 0).empty(), "no detections: no tracks");
    check(tracker.update({}, 1).empty() && tracker.createdTracks() == 0, "no detections: still no new track");
}

void testMaxAgeAndOcclusion()
{
    TrackerOptions options;
    Tracker tracker(options);
    for (int f = 0; f < 3; ++f) tracker.update({ person(200, 100) }, f);

    // 가려짐: max_age(30) 프레임까지는 LOST로 예측 박스를 유지함
    std::vector<TrackState> out;
    bool all_lost = true;
    for (int f = 3; f < 3 + options.max_age; ++f) {
        out = tracker.update({}, f);
        all_lost = all_lost && out.size() == 1 && out[0].state == TrackState::LOST && out[0].id == 1;
    }
    check(all_lost, "max_age: LOST for 30 missed frames");
    check(out.size() == 1 && std::abs(out[0].box.x - 200.0f) < 1.0f, "LOST: keeps the predicted box");

    out = tracker.update({}, 3 + options.max_age);
    check(out.empty(), "max_age: deleted after 31 missed frames");

    // 다시 나타나면 같은 ID (max_age 안)
    Tracker again(options);
    for (int f = 0; f < 5; ++f) again.update({ person(200.0f + 3.0f * f, 100) }, f);
    for (int f = 5; f < 10; ++f) again.update({}, f);
    out = again.update({ person(200.0f + 3.0f * 10, 100) }, 10);
    check(out.size() == 1 && out[0].id == 1 && out[0].state == TrackState::CONFIRMED,
        "occlusion: same id comes back as CONFIRMED");
}

void testIouThreshold()
{
    Tracker tracker;
    for (int f = 0; f < 3; ++f) tracker.update({ person(100, 100) }, f);
    // 겹침이 거의 없는 곳으로 튀면 같은 사람으로 보지 않음
    const auto out = tracker.update({ person(160, 100) }, 3);
    check(out.size() == 2 && find(out, 1) && find(out, 1)->state == TrackState::LOST && find(out, 2)
        && find(out, 2)->state == TrackState::TENTATIVE, "iou_threshold: low IoU makes a new track");
}

void testOutOfFrame()
{
    Tracker tracker;
    tracker.setFrameSize(cv::Size(640, 480));
    // 프레임마다 왼쪽으로 20px씩 나감
    for (int f = 0; f < 5; ++f) tracker.update({ person(100.0f - 20.0f * f, 100) }, f);
    int frames_until_deleted = 0;
    for (int f = 5; f < 40; ++f) {
        if (tracker.update({}, f).empty()) break;
        ++frames_until_deleted;
    }
    check(frames_until_deleted < 30, "out of frame: deleted before max_age");
}

void testFrameGapAndReset()
{
    Tracker tracker;
    for (int f = 0; f < 3; ++f) tracker.update({ person(100, 100) }, f);
    // 실패로 31프레임을 건너뛰면 max_age를 넘으므로 삭제
    check(tracker.update({}, 2 + 31).empty(), "frame gap: skipped frames count toward max_age");

    tracker.reset();
    const auto out = tracker.update({ person(100, 100) }, 0);
    check(out.size() == 1 && out[0].id == 1 && tracker.createdTracks() == 1, "reset: id starts from start_id");

    TrackerOptions options;
    options.start_id = 100;
    options.min_hits = 1;
    Tracker custom(options);
    const auto first = custom.update({ person(100, 100) }, 0);
    check(first.size() == 1 && first[0].id == 100 && first[0].state == TrackState::CONFIRMED,
        "options: start_id and min_hits = 1");
}

}  // namespace 끝

int main()
{
    testIdsAndMinHits();
    testTentativeDeleted();
    testNoDetections();
    testMaxAgeAndOcclusion();
    testIouThreshold();
    testOutOfFrame();
    testFrameGapAndReset();
    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
