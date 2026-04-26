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
constexpr const char* kTempPath = "/dev/shm/argus_frame.tmp.jpg";

std::atomic<bool> g_running{true};

void handle_signal(int) {
    g_running.store(false);
}
}  // namespace

int main() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open webcam device 0\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, kWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, kHeight);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    cv::Mat frame;
    cv::Mat resized;
    std::vector<unsigned char> input(kWidth * kHeight * kChannels);
    std::vector<unsigned char> gray(kWidth * kHeight);
    std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 85};

    while (g_running.load()) {
        if (!cap.read(frame) || frame.empty()) {
            continue;
        }

        cv::resize(frame, resized, cv::Size(kWidth, kHeight), 0.0, 0.0, cv::INTER_AREA);
        if (!resized.isContinuous()) {
            resized = resized.clone();
        }

        const int pixels = kWidth * kHeight;
        const int input_len = pixels * kChannels;
        const int gray_len = pixels;

        std::memcpy(input.data(), resized.data, static_cast<std::size_t>(input_len));

        unsigned char* in = input.data();
        unsigned char* out = gray.data();

        #pragma acc data copyin(in[0:input_len]) copyout(out[0:gray_len])
        {
            #pragma acc parallel loop present(in[0:input_len], out[0:gray_len])
            for (int i = 0; i < pixels; ++i) {
                const int base = i * kChannels;
                const unsigned char b = in[base + 0];
                const unsigned char g = in[base + 1];
                const unsigned char r = in[base + 2];
                out[i] = static_cast<unsigned char>((static_cast<int>(r) + static_cast<int>(g) + static_cast<int>(b)) / 3);
            }
        }

        cv::Mat gray_mat(kHeight, kWidth, CV_8UC1, gray.data());
        if (cv::imwrite(kTempPath, gray_mat, jpeg_params)) {
            std::rename(kTempPath, kOutputPath);
        }
    }

    return 0;
}
