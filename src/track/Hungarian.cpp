#include "track/Hungarian.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool finiteBox(const cv::Rect2f& r)
{
    return std::isfinite(r.x) && std::isfinite(r.y) && std::isfinite(r.width) && std::isfinite(r.height);
}

}  // namespace 끝

float iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    // std::min/max는 NaN을 그냥 지나칠 수 있어서 먼저 거름
    if (!finiteBox(a) || !finiteBox(b)) return 0.0f;
    const float area_a = a.width * a.height;
    const float area_b = b.width * b.height;
    if (!(area_a > 0.0f) || !(area_b > 0.0f)) return 0.0f;

    const float ix = std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x);
    const float iy = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);
    if (!(ix > 0.0f) || !(iy > 0.0f)) return 0.0f;

    const float inter = ix * iy;
    return inter / (area_a + area_b - inter);
}

namespace {

// 행 수 n <= 열 수 m 인 경우의 풀이
// 각 행을 하나씩 추가하면서, 퍼텐셜 u(행)·v(열)를 고쳐 가며 최단 증가 경로를 찾음
// 1부터 세는 번호를 쓰고 0번 열은 "아직 짝 없음"을 나타내는 가상의 열
std::vector<int> solveWide(const std::vector<std::vector<double>>& a, int n, int m)
{
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> row_of(m + 1, 0);   // row_of[j] = 열 j에 짝지어진 행 (0 = 없음)
    std::vector<int> prev_col(m + 1, 0); // 증가 경로에서 열 j 직전의 열

    for (int i = 1; i <= n; ++i) {
        row_of[0] = i;
        int col = 0;
        std::vector<double> min_slack(m + 1, inf);
        std::vector<bool> used(m + 1, false);

        // 짝 없는 열에 닿을 때까지 경로를 늘림
        do {
            used[col] = true;
            const int row = row_of[col];
            double delta = inf;
            int next_col = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double slack = a[row - 1][j - 1] - u[row] - v[j];
                if (slack < min_slack[j]) {
                    min_slack[j] = slack;
                    prev_col[j] = col;
                }
                if (min_slack[j] < delta) {
                    delta = min_slack[j];
                    next_col = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[row_of[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_slack[j] -= delta;
                }
            }
            col = next_col;
        } while (row_of[col] != 0);

        // 찾은 경로를 따라 짝을 뒤집음
        do {
            const int before = prev_col[col];
            row_of[col] = row_of[before];
            col = before;
        } while (col != 0);
    }

    std::vector<int> result(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (row_of[j] != 0) result[row_of[j] - 1] = j - 1;
    }
    return result;
}

}  // namespace 끝

std::vector<int> solveAssignment(const std::vector<std::vector<double>>& cost)
{
    const int rows = static_cast<int>(cost.size());
    if (rows == 0) return {};
    const int cols = static_cast<int>(cost[0].size());
    if (cols == 0) return std::vector<int>(rows, -1);

    if (rows <= cols) return solveWide(cost, rows, cols);

    // 행이 더 많으면 전치해서 풀고 결과를 되돌림
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) transposed[j][i] = cost[i][j];
    }
    const std::vector<int> col_to_row = solveWide(transposed, cols, rows);

    std::vector<int> result(rows, -1);
    for (int j = 0; j < cols; ++j) {
        if (col_to_row[j] >= 0) result[col_to_row[j]] = j;
    }
    return result;
}

MatchResult associate(const std::vector<cv::Rect2f>& track_boxes,
    const std::vector<cv::Rect2f>& detection_boxes, float iou_threshold)
{
    const int num_tracks = static_cast<int>(track_boxes.size());
    const int num_dets = static_cast<int>(detection_boxes.size());
    std::vector<int> det_of_track(num_tracks, -1);
    std::vector<bool> det_matched(num_dets, false);

    if (num_tracks > 0 && num_dets > 0) {
        std::vector<std::vector<double>> cost(num_tracks, std::vector<double>(num_dets));
        std::vector<std::vector<float>> overlap(num_tracks, std::vector<float>(num_dets));
        for (int t = 0; t < num_tracks; ++t) {
            for (int d = 0; d < num_dets; ++d) {
                overlap[t][d] = iou(track_boxes[t], detection_boxes[d]);
                cost[t][d] = 1.0 - overlap[t][d];
            }
        }
        const std::vector<int> assignment = solveAssignment(cost);
        for (int t = 0; t < num_tracks; ++t) {
            const int d = assignment[t];
            if (d < 0 || overlap[t][d] < iou_threshold) continue;
            det_of_track[t] = d;
            det_matched[d] = true;
        }
    }

    MatchResult result;
    for (int t = 0; t < num_tracks; ++t) {
        if (det_of_track[t] >= 0) result.matches.emplace_back(t, det_of_track[t]);
        else result.unmatched_tracks.push_back(t);
    }
    for (int d = 0; d < num_dets; ++d) {
        if (!det_matched[d]) result.unmatched_detections.push_back(d);
    }
    return result;
}
