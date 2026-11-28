#include "track/Tracker.h"

#include "track/Hungarian.h"

#include <algorithm>
#include <cmath>

namespace {

bool isBadBox(const cv::Rect2f& box)
{
    return !std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.width)
        || !std::isfinite(box.height) || !(box.width * box.height > 0.0f);
}

}  // namespace 끝

Tracker::Tracker(const TrackerOptions& options) : options_(options), next_id_(options.start_id)
{
}

void Tracker::reset()
{
    tracks_.clear();
    next_id_ = options_.start_id;
    last_frame_idx_ = -1;
    created_ = 0;
    confirmed_ = 0;
}

bool Tracker::outOfFrame(const cv::Rect2f& box) const
{
    if (frame_size_.width <= 0 || frame_size_.height <= 0) return false;
    return box.x + box.width <= 0.0f || box.y + box.height <= 0.0f
        || box.x >= static_cast<float>(frame_size_.width) || box.y >= static_cast<float>(frame_size_.height);
}

std::vector<TrackState> Tracker::update(const std::vector<Detection>& person_boxes, int frame_idx)
{
    // 건너뛴 프레임이 있으면 그만큼 더 예측함 (첫 호출, 같은 번호, 되감기는 1번)
    const int steps = (last_frame_idx_ >= 0 && frame_idx > last_frame_idx_) ? frame_idx - last_frame_idx_ : 1;
    last_frame_idx_ = frame_idx;

    // 1. 예측. 상태가 깨졌거나 화면 밖으로 나간 트랙은 지움
    std::vector<cv::Rect2f> predicted;
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        cv::Rect2f box;
        for (int s = 0; s < steps; ++s) box = it->kf.predict();
        if (it->kf.invalid() || isBadBox(box) || outOfFrame(box)) {
            it = tracks_.erase(it);
            continue;
        }
        predicted.push_back(box);
        ++it;
    }

    // 2~3. 매칭 (트랙 또는 탐지가 없으면 헝가리안 없이 모두 미매칭)
    const size_t num_tracks = tracks_.size();
    const size_t num_dets = person_boxes.size();
    std::vector<cv::Rect2f> detected;
    detected.reserve(num_dets);
    for (const Detection& d : person_boxes) detected.push_back(d.box);
    const MatchResult match = associate(predicted, detected, options_.iou_threshold);

    std::vector<int> det_of_track(num_tracks, -1);
    std::vector<bool> det_matched(num_dets, false);
    for (const auto& [t, d] : match.matches) {
        det_of_track[t] = d;
        det_matched[d] = true;
    }

    // 4~5. 기존 트랙 보정과 상태 전이
    std::vector<Track> kept;
    kept.reserve(num_tracks + num_dets);
    for (size_t t = 0; t < num_tracks; ++t) {
        Track& track = tracks_[t];
        if (det_of_track[t] >= 0) {
            track.kf.update(person_boxes[det_of_track[t]].box);
            if (track.kf.invalid()) continue;
            ++track.hits;
            track.time_since_update = 0;
            if (track.state == TrackState::LOST || track.hits >= options_.min_hits) {
                track.state = TrackState::CONFIRMED;
            }
        } else {
            if (track.state == TrackState::TENTATIVE) continue;
            track.time_since_update += steps;
            if (track.time_since_update > options_.max_age) continue;
            track.state = TrackState::LOST;
        }
        if (track.state == TrackState::CONFIRMED && !track.ever_confirmed) {
            track.ever_confirmed = true;
            ++confirmed_;
        }
        kept.push_back(std::move(track));
    }

    // 4. 남은 탐지로 새 트랙 (ID는 재사용하지 않음)
    for (size_t d = 0; d < num_dets; ++d) {
        if (det_matched[d] || isBadBox(person_boxes[d].box)) continue;
        Track track(next_id_++, person_boxes[d].box);
        ++created_;
        if (track.hits >= options_.min_hits) {
            track.state = TrackState::CONFIRMED;
            track.ever_confirmed = true;
            ++confirmed_;
        }
        kept.push_back(std::move(track));
    }
    tracks_ = std::move(kept);

    std::vector<TrackState> result;
    result.reserve(tracks_.size());
    for (const Track& track : tracks_) {
        TrackState out{};
        out.id = track.id;
        out.box = track.kf.box();
        out.state = track.state;
        result.push_back(out);
    }
    return result;
}
