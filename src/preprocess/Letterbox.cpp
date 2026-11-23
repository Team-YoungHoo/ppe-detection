#include "preprocess/Letterbox.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

LetterboxResult letterbox(const cv::Mat& src, int size, int pad_value)
{
    LetterboxResult out;
    if (src.empty()) {
        // Log.h(SYS-04)가 준비되면 교체
        std::cerr << "[WARN] Empty frame, skipped\n";
        return out;
    }

    const int w = src.cols;
    const int h = src.rows;

    // 1. 배율 = min(S / h, S / w). S보다 작은 프레임도 같은 방식으로 확대함
    const double r = std::min(static_cast<double>(size) / h, static_cast<double>(size) / w);

    // 2. 새 크기를 반올림으로 계산함 (Ultralytics LetterBox와 동일)
    const int new_w = static_cast<int>(std::lround(w * r));
    const int new_h = static_cast<int>(std::lround(h * r));

    cv::Mat resized;
    if (new_w != w || new_h != h) {
        cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = src;
    }

    // 3. 이미지를 가운데에 놓고 나머지를 회색으로 채움
    //    패딩이 홀수일 때 -0.1 / +0.1로 나눠야 Ultralytics와 같아짐
    //    덕분에 Python에서 쓰는 텐서와 값이 동일함
    const double dw = (size - new_w) / 2.0;
    const double dh = (size - new_h) / 2.0;
    const int top = static_cast<int>(std::lround(dh - 0.1));
    const int bottom = static_cast<int>(std::lround(dh + 0.1));
    const int left = static_cast<int>(std::lround(dw - 0.1));
    const int right = static_cast<int>(std::lround(dw + 0.1));

    cv::copyMakeBorder(resized, out.image, top, bottom, left, right, cv::BORDER_CONSTANT,
        cv::Scalar(pad_value, pad_value, pad_value));

    out.info.scale = static_cast<float>(r);
    out.info.scale_x = static_cast<float>(new_w) / w;
    out.info.scale_y = static_cast<float>(new_h) / h;
    out.info.pad_x = left;
    out.info.pad_y = top;
    out.info.orig_w = w;
    out.info.orig_h = h;
    return out;
}

cv::Mat toBlob(const cv::Mat& letterboxed, double scale, bool swap_rb)
{
    // BGR -> RGB, scale 곱하기, HWC -> NCHW, 배치 1, float32
    // 이미 S x S이므로 blobFromImage가 다시 리사이즈하지 않음
    return cv::dnn::blobFromImage(letterboxed, scale, letterboxed.size(), cv::Scalar(), swap_rb,
        false, CV_32F);
}

cv::Point2f toOriginal(const cv::Point2f& p, const LetterboxInfo& info)
{
    // 축별 실제 배율을 사용함
    // 크기를 반올림하면 scale과 달라져서 좌표가 최대 0.5 / scale 픽셀 밀릴 수 있음
    return { (p.x - info.pad_x) / info.scale_x, (p.y - info.pad_y) / info.scale_y };
}
