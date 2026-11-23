// VIN-02 / VIN-03 단위 테스트 (src/preprocess/Letterbox)
//
// 사용법:
//   test_letterbox                         전체 검사, 모두 통과하면 종료 코드 0
//   test_letterbox --dump <이미지> <출력>  이미지의 blob을 float32 그대로 파일에 저장
//                                          (tests/check_blob.py 비교용)
//   test_letterbox --view <이미지> <폴더>  변환 전후 이미지를 폴더에 저장
//                                          두 초록 박스 안의 장면이 같아야 함
//
// 검사 항목:
//   1. 여러 프레임 크기의 출력 크기(S x S)와 패딩
//      (가로형, 세로형, 4:3, 정사각형, S보다 작은 영상, 4K)
//   2. VIN-02 명세: cv::resize로 실제 변환한 픽셀이 원래 위치로 1픽셀 이내로 돌아옴
//   3. 패딩 색이 114
//   4. 빈 프레임은 예외 없이 빈 이미지를 반환
//   5. VIN-03: blob 형태 {1, 3, S, S}, RGB 순서, 값 / 255

#include "preprocess/Letterbox.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& name)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failed;
}

// 각 픽셀에 자기 (x, y)를 저장한 프레임을 만듦
// 실제로 리사이즈한 뒤 픽셀 값을 보면 원본의 어느 위치에서 왔는지 알 수 있음
cv::Mat coordinateFrame(int w, int h)
{
    cv::Mat frame(h, w, CV_32FC3);
    for (int y = 0; y < h; ++y) {
        auto* row = frame.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            row[x] = cv::Vec3f(static_cast<float>(x), static_cast<float>(y), 0.0f);
        }
    }
    return frame;
}

// S x S 안에서 패딩을 뺀 실제 이미지 크기
cv::Size resizedSize(const LetterboxInfo& info)
{
    return { static_cast<int>(std::lround(info.orig_w * info.scale_x)),
        static_cast<int>(std::lround(info.orig_h * info.scale_y)) };
}

float maxMappingError(const LetterboxResult& r)
{
    float max_err = 0.0f;
    const cv::Rect area(cv::Point(r.info.pad_x, r.info.pad_y), resizedSize(r.info));
    for (int my = area.y; my < area.br().y; ++my) {
        const auto* row = r.image.ptr<cv::Vec3f>(my);
        for (int mx = area.x; mx < area.br().x; ++mx) {
            // +0.5: 픽셀 번호 -> 픽셀 중심의 연속 좌표
            const cv::Point2f back = toOriginal(cv::Point2f(mx + 0.5f, my + 0.5f), r.info);
            max_err = std::max(max_err, std::abs(back.x - (row[mx][0] + 0.5f)));
            max_err = std::max(max_err, std::abs(back.y - (row[mx][1] + 0.5f)));
        }
    }
    return max_err;
}

void testSize(int w, int h, int S)
{
    const std::string tag = std::to_string(w) + "x" + std::to_string(h);
    cv::Mat frame(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    const LetterboxResult r = letterbox(frame, S);

    check(r.image.cols == S && r.image.rows == S && r.image.type() == CV_8UC3,
        tag + ": output is " + std::to_string(S) + "x" + std::to_string(S));

    // 변환된 이미지는 S x S 안에 들어가고 한 축은 양쪽 끝에 닿아야 함
    const cv::Size resized = resizedSize(r.info);
    const int new_w = resized.width;
    const int new_h = resized.height;
    const bool fits = (new_w == S || new_h == S) && new_w <= S && new_h <= S;
    check(fits, tag + ": aspect ratio kept, long side = S (scale " + std::to_string(r.info.scale) + ")");

    const float max_err = maxMappingError(letterbox(coordinateFrame(w, h), S, 0));
    check(max_err <= 1.0f, tag + ": mapping error " + std::to_string(max_err) + " px <= 1 px");
}

void testPadColor()
{
    cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
    const LetterboxResult r = letterbox(frame, 640);
    // 1920x1080 -> 640x360, 위아래에 회색 140줄씩
    const cv::Vec3b top = r.image.at<cv::Vec3b>(0, 320);
    const cv::Vec3b bottom = r.image.at<cv::Vec3b>(639, 320);
    const cv::Vec3b center = r.image.at<cv::Vec3b>(320, 320);
    check(r.info.pad_x == 0 && r.info.pad_y == 140, "1920x1080: pad_x 0, pad_y 140");
    check(r.info.orig_w == 1920 && r.info.orig_h == 1080, "1920x1080: orig_w, orig_h are stored");
    check(top == cv::Vec3b(114, 114, 114) && bottom == cv::Vec3b(114, 114, 114),
        "padding color is (114, 114, 114)");
    check(center == cv::Vec3b(0, 0, 0), "image area is not padded");
}

void testEmpty()
{
    const LetterboxResult r = letterbox(cv::Mat(), 640);
    check(r.image.empty(), "empty frame returns empty image (no exception)");
}

void testBlob()
{
    // 단색 BGR (10, 20, 30): swapRB 후에는 R 채널이 먼저 나옴
    cv::Mat img(640, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat blob = toBlob(img);

    const bool shape = blob.dims == 4 && blob.size[0] == 1 && blob.size[1] == 3
        && blob.size[2] == 640 && blob.size[3] == 640 && blob.type() == CV_32F;
    check(shape, "blob shape {1, 3, 640, 640}, float32");

    const float* d = blob.ptr<float>();
    const size_t plane = 640 * 640;
    const bool rgb = std::abs(d[0] - 30 / 255.0f) < 1e-6f
        && std::abs(d[plane] - 20 / 255.0f) < 1e-6f
        && std::abs(d[2 * plane] - 10 / 255.0f) < 1e-6f;
    check(rgb, "blob is RGB order and scaled by 1/255");
}

int view(const std::string& image_path, const std::string& out_dir, int S)
{
    const cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << image_path << "\n";
        return 2;
    }
    LetterboxResult r = letterbox(img, S);

    // 이미지 영역 가운데에 놓은 가짜 박스 (모델 좌표)
    const cv::Size resized = resizedSize(r.info);
    const cv::Point2f model_tl(r.info.pad_x + resized.width * 0.25f,
        r.info.pad_y + resized.height * 0.25f);
    const cv::Point2f model_br(r.info.pad_x + resized.width * 0.75f,
        r.info.pad_y + resized.height * 0.75f);
    const cv::Rect2f model_box(model_tl, model_br);

    // 두 모서리를 점 역변환으로 되돌려 원본 위치의 박스를 만듦
    const cv::Rect2f original_box(toOriginal(model_tl, r.info), toOriginal(model_br, r.info));

    cv::Mat original = img.clone();
    const int thickness = std::max(2, static_cast<int>(std::lround(2 / r.info.scale)));
    cv::rectangle(original, cv::Rect(original_box), cv::Scalar(0, 255, 0), thickness);
    cv::rectangle(r.image, cv::Rect(model_box), cv::Scalar(0, 255, 0), 2);

    std::filesystem::create_directories(out_dir);
    const std::string before = (std::filesystem::path(out_dir) / "1_original.png").string();
    const std::string after = (std::filesystem::path(out_dir) / "2_letterbox.png").string();
    if (!cv::imwrite(before, original) || !cv::imwrite(after, r.image)) {
        std::cerr << "[ERROR] Cannot write images to: " << out_dir << "\n";
        return 2;
    }
    std::cout << "[INFO] " << img.cols << "x" << img.rows << " -> " << resized.width << "x"
        << resized.height << " + pad (" << r.info.pad_x << ", " << r.info.pad_y << ")\n"
        << "[INFO] Saved: " << before << "\n[INFO] Saved: " << after << "\n";
    return 0;
}

int dump(const std::string& image_path, const std::string& out_path, int S)
{
    const cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << image_path << "\n";
        return 2;
    }
    const LetterboxResult r = letterbox(img, S);
    const cv::Mat blob = toBlob(r.image);

    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(blob.ptr<float>()),
        static_cast<std::streamsize>(blob.total() * sizeof(float)));
    std::cout << "[INFO] " << image_path << " (" << img.cols << "x" << img.rows
        << ") -> blob {1, 3, " << S << ", " << S << "} scale " << r.info.scale
        << " pad (" << r.info.pad_x << ", " << r.info.pad_y << ") -> " << out_path << "\n";
    return 0;
}

}  // namespace 끝

int main(int argc, char** argv)
{
    if (argc >= 4 && std::string(argv[1]) == "--dump") {
        const int S = (argc >= 5) ? std::stoi(argv[4]) : 640;
        return dump(argv[2], argv[3], S);
    }

    if (argc >= 4 && std::string(argv[1]) == "--view") {
        const int S = (argc >= 5) ? std::stoi(argv[4]) : 640;
        return view(argv[2], argv[3], S);
    }

    testSize(1920, 1080, 640);  // 가로형 FHD
    testSize(1280, 720, 640);   // 가로형 HD
    testSize(3840, 2160, 640);  // 4K
    testSize(1080, 1920, 640);  // 세로형 (휴대폰)
    testSize(640, 480, 640);    // 4:3 CCTV
    testSize(320, 240, 640);    // S보다 작음 -> 확대
    testSize(800, 800, 640);    // 정사각형
    testSize(1001, 333, 640);   // 패딩이 홀수
    testSize(3840, 2158, 640);  // 반올림 경우: 수정 전 오차가 가장 컸음
    testSize(2560, 1439, 640);  // 반올림 경우
    testSize(1918, 1078, 640);  // 반올림 경우
    testSize(1920, 1080, 960);  // 960 입력 모델
    testPadColor();
    testEmpty();
    testBlob();

    std::cout << (g_failed == 0 ? "\nALL PASSED\n" : "\nFAILED: " + std::to_string(g_failed) + "\n");
    return g_failed == 0 ? 0 : 1;
}
