#pragma once

#include <opencv2/videoio.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace sima_test {

inline int probe_rtsp_source_fps(const std::string& url) {
  cv::VideoCapture capture(url);
  if (!capture.isOpened()) {
    throw std::runtime_error("failed to open RTSP source for FPS probe");
  }

  const int fps = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FPS)));
  capture.release();
  if (fps <= 0) {
    throw std::runtime_error("failed to probe RTSP source FPS");
  }
  return fps;
}

} // namespace sima_test
