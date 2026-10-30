// Build check program (design page 4-5 completion criterion)
// Opens a video file, shows it in a window, and overlays the processing FPS.
//
// Usage : safety_app <video file path>
// Quit  : ESC or q key, closing the window, or end of video
// Exit codes (feature spec 2-4): 0 = normal, 2 = input video error
//
// Note: comments and log messages are kept in English on purpose.
//       MSVC may misread Korean text in UTF-8 source files unless the
//       /utf-8 compiler option is set, which can break the build check.

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    using Clock = std::chrono::steady_clock;

    // 1. Check arguments
    if (argc < 2) {
        std::cerr << "[ERROR] Usage: " << argv[0] << " <video file path>\n";
        return 2;
    }
    const std::string path = argv[1];

    // 2. Open the video file
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video: " << path << "\n";
        return 2;
    }

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double video_fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "[INFO] OpenCV " << CV_VERSION << "\n";
    std::cout << "[INFO] Video: " << path << " (" << width << "x" << height
        << ", " << video_fps << " fps)\n";

    // 3. Create the window (name from UI-01: ui.window_name)
    const std::string window = "Safety Monitor";
    cv::namedWindow(window, cv::WINDOW_NORMAL);

    // 4. Frame loop
    cv::Mat frame;
    int frames = 0;
    double fps_smooth = 0.0;
    const auto t_start = Clock::now();

    while (true) {
        const auto t0 = Clock::now();

        if (!cap.read(frame) || frame.empty()) {
            break;  // end of video
        }
        ++frames;

        // Draw FPS of the previous loop (smoothed)
        char text[64];
        std::snprintf(text, sizeof(text), "FPS: %.1f", fps_smooth);
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
            0.8, cv::Scalar(0, 0, 0), 4);        // outline
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
            0.8, cv::Scalar(0, 255, 0), 2);      // text

        cv::imshow(window, frame);
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;  // quit key
        }
        if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) {
            break;  // window closed with the X button
        }

        // Processing FPS = 1 / time of one loop (read + draw + show)
        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        if (dt > 0.0) {
            const double inst = 1.0 / dt;
            fps_smooth = (frames == 1) ? inst : 0.9 * fps_smooth + 0.1 * inst;
        }
    }

    // 5. Summary
    const double total = std::chrono::duration<double>(Clock::now() - t_start).count();
    std::cout << "[INFO] Processed frames: " << frames << "\n";
    if (total > 0.0 && frames > 0) {
        std::cout << "[INFO] Average FPS: " << frames / total << "\n";
    }

    cv::destroyAllWindows();
    return 0;
}