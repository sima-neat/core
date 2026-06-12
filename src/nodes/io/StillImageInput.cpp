#include "nodes/io/StillImageInput.h"

#include "gst/GstInit.h"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace simaai::neat {
namespace {

std::vector<uint8_t> bgr_to_nv12_tight(const cv::Mat& bgr_in, int w, int h) {
  cv::Mat bgr;
  if (bgr_in.cols != w || bgr_in.rows != h) {
    cv::resize(bgr_in, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  } else {
    bgr = bgr_in;
  }

  cv::Mat yuv_i420;
  cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);

  const size_t y_sz = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t u_sz = y_sz / 4;
  const size_t v_sz = u_sz;

  if (static_cast<size_t>(yuv_i420.total()) != y_sz + u_sz + v_sz) {
    throw std::runtime_error("bgr_to_nv12_tight: unexpected I420 size from OpenCV");
  }

  const uint8_t* base = yuv_i420.data;
  const uint8_t* y = base;
  const uint8_t* u = base + y_sz;
  const uint8_t* v = base + y_sz + u_sz;

  std::vector<uint8_t> nv12;
  nv12.resize(y_sz + 2 * u_sz);
  std::memcpy(nv12.data(), y, y_sz);

  uint8_t* uv = nv12.data() + y_sz;
  for (size_t i = 0; i < u_sz; ++i) {
    uv[2 * i + 0] = u[i];
    uv[2 * i + 1] = v[i];
  }
  return nv12;
}

std::vector<uint8_t> nv12_pad_center(const std::vector<uint8_t>& src_nv12, int src_w, int src_h,
                                     int dst_w, int dst_h) {
  if (dst_w < src_w || dst_h < src_h) {
    throw std::runtime_error("nv12_pad_center: dst must be >= src");
  }
  if ((src_w & 1) || (src_h & 1) || (dst_w & 1) || (dst_h & 1)) {
    throw std::runtime_error("nv12_pad_center: NV12 requires even dimensions");
  }

  const size_t src_y_sz = static_cast<size_t>(src_w) * static_cast<size_t>(src_h);
  const size_t src_uv_sz = src_y_sz / 2;
  if (src_nv12.size() != src_y_sz + src_uv_sz) {
    throw std::runtime_error("nv12_pad_center: src size mismatch");
  }

  const size_t dst_y_sz = static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h);
  const size_t dst_uv_sz = dst_y_sz / 2;

  std::vector<uint8_t> dst(dst_y_sz + dst_uv_sz);
  std::memset(dst.data(), 16, dst_y_sz);
  std::memset(dst.data() + dst_y_sz, 128, dst_uv_sz);

  int off_x = (dst_w - src_w) / 2;
  int off_y = (dst_h - src_h) / 2;
  off_x &= ~1;
  off_y &= ~1;

  const uint8_t* src_y = src_nv12.data();
  const uint8_t* src_uv = src_nv12.data() + src_y_sz;

  uint8_t* dst_y = dst.data();
  uint8_t* dst_uv = dst.data() + dst_y_sz;

  for (int r = 0; r < src_h; ++r) {
    std::memcpy(dst_y + static_cast<size_t>(off_y + r) * static_cast<size_t>(dst_w) +
                    static_cast<size_t>(off_x),
                src_y + static_cast<size_t>(r) * static_cast<size_t>(src_w),
                static_cast<size_t>(src_w));
  }

  const int src_uv_h = src_h / 2;
  const int dst_uv_off_y = off_y / 2;

  for (int r = 0; r < src_uv_h; ++r) {
    std::memcpy(dst_uv + static_cast<size_t>(dst_uv_off_y + r) * static_cast<size_t>(dst_w) +
                    static_cast<size_t>(off_x),
                src_uv + static_cast<size_t>(r) * static_cast<size_t>(src_w),
                static_cast<size_t>(src_w));
  }

  return dst;
}

} // namespace

StillImageInput::StillImageInput(std::string image_path, ContentWidth content_w,
                                 ContentHeight content_h, EncodeWidth enc_w, EncodeHeight enc_h,
                                 FramesPerSecond fps)
    : image_path_(std::move(image_path)), content_w_(content_w.value), content_h_(content_h.value),
      enc_w_(enc_w.value), enc_h_(enc_h.value), fps_(fps.value) {
  gst_init_once();

  if (!fs::exists(image_path_)) {
    throw std::runtime_error("StillImageInput: file not found: " + image_path_);
  }
  if ((content_w_ & 1) || (content_h_ & 1) || (enc_w_ & 1) || (enc_h_ & 1)) {
    throw std::runtime_error("StillImageInput: widths/heights must be even for NV12");
  }
  if (enc_w_ < content_w_ || enc_h_ < content_h_) {
    throw std::runtime_error("StillImageInput: enc dims must be >= content dims");
  }
  if (fps_ <= 0) {
    throw std::runtime_error("StillImageInput: fps must be > 0");
  }

  cv::Mat bgr = cv::imread(image_path_, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("StillImageInput: OpenCV failed to read: " + image_path_);
  }

  auto nv12_content = bgr_to_nv12_tight(bgr, content_w_, content_h_);
  std::vector<uint8_t> nv12_enc =
      (enc_w_ == content_w_ && enc_h_ == content_h_)
          ? std::move(nv12_content)
          : nv12_pad_center(nv12_content, content_w_, content_h_, enc_w_, enc_h_);

  nv12_enc_ = std::make_shared<std::vector<uint8_t>>(std::move(nv12_enc));
}

std::string StillImageInput::backend_fragment(int node_index) const {
  (void)node_index;
  std::ostringstream ss;
  ss << "appsrc name=mysrc is-live=true format=time " << "! queue name=n" << node_index << "_queue "
     << "! video/x-raw,format=NV12,width=" << enc_w_ << ",height=" << enc_h_
     << ",framerate=" << fps_ << "/1";
  return ss.str();
}

std::vector<std::string> StillImageInput::element_names(int node_index) const {
  return {"mysrc", "n" + std::to_string(node_index) + "_queue"};
}

OutputSpec StillImageInput::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = "NV12";
  out.width = enc_w_;
  out.height = enc_h_;
  out.depth = 0;
  out.fps_num = fps_;
  out.fps_den = 1;
  out.layout = "Planar";
  out.dtype = "UInt8";
  out.memory = "SystemMemory";
  out.certainty = SpecCertainty::Derived;
  out.note = "StillImageInput (NV12)";
  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node>
StillImageInput(std::string image_path, simaai::neat::StillImageInput::ContentWidth content_w,
                simaai::neat::StillImageInput::ContentHeight content_h,
                simaai::neat::StillImageInput::EncodeWidth enc_w,
                simaai::neat::StillImageInput::EncodeHeight enc_h,
                simaai::neat::StillImageInput::FramesPerSecond fps) {
  return std::make_shared<simaai::neat::StillImageInput>(std::move(image_path), content_w,
                                                         content_h, enc_w, enc_h, fps);
}

} // namespace simaai::neat::nodes
