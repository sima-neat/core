/**
 * @file
 * @ingroup nodes_io
 * @brief OptiView JSON UDP sender helpers.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

struct OptiViewObject {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  float score = 0.0f;
  int class_id = -1;
};

struct OptiViewChannelOptions {
  std::string host = "127.0.0.1";
  int channel = 0;
  int video_port_base = 9000;
  int json_port_base = 9100;
};

std::vector<std::string> OptiViewDefaultLabels();

std::string OptiViewMakeJson(int64_t timestamp_ms, const std::string& frame_id,
                             const std::vector<OptiViewObject>& objects,
                             const std::vector<std::string>& labels);

class OptiViewJsonOutput {
public:
  explicit OptiViewJsonOutput(const OptiViewChannelOptions& opt, std::string* err = nullptr);
  ~OptiViewJsonOutput();
  OptiViewJsonOutput(const OptiViewJsonOutput&) = delete;
  OptiViewJsonOutput& operator=(const OptiViewJsonOutput&) = delete;
  OptiViewJsonOutput(OptiViewJsonOutput&&) noexcept;
  OptiViewJsonOutput& operator=(OptiViewJsonOutput&&) noexcept;

  bool ok() const;
  const std::string& host() const;
  int json_port() const;
  int video_port() const;

  bool send_json(const std::string& payload, std::string* err = nullptr) const;

  bool send_detection(int64_t timestamp_ms, const std::string& frame_id,
                      const std::vector<OptiViewObject>& objects,
                      const std::vector<std::string>& labels, std::string* err = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat
