#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kChannels = 3;

constexpr const char* kOutputPath = "/dev/shm/argus_frame.jpg";
constexpr const char* kTempPath   = "/dev/shm/argus_frame.tmp.jpg";

std::atomic<bool> g_running{true};

void handle_signal(int) {
    g_running.store(false);
}
}  // namespace

int main() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    // 🔥 Use V4L2 backend (important for WSL)
    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open webcam device 0\n";
        return 1;
    }

    // 🔥 Force MJPEG (critical fix for WSL camera)
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, kWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, kHeight);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    cv::Mat frame;
    cv::Mat resized;

    std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 85};

    std::cout << "📷 Webcam started successfully...\n";

    while (g_running.load()) {

        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "⚠️ Frame capture failed, retrying...\n";
            continue;
        }

        // Resize frame
        cv::resize(frame, resized, cv::Size(kWidth, kHeight), 0.0, 0.0, cv::INTER_AREA);

        // Write safely (atomic swap)
        if (cv::imwrite(kTempPath, resized, jpeg_params)) {
            std::rename(kTempPath, kOutputPath);
        }
    }

    cap.release();
    return 0;
}