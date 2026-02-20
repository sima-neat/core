#include "nodes/io/OptiViewJsonOutput.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace {

class UdpReceiver {
public:
  UdpReceiver() {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) {
      throw std::runtime_error("UdpReceiver socket() failed");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (::bind(fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
      ::close(fd_);
      throw std::runtime_error("UdpReceiver bind() failed");
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
      ::close(fd_);
      throw std::runtime_error("UdpReceiver getsockname() failed");
    }

    port_ = static_cast<int>(ntohs(addr.sin_port));
  }

  ~UdpReceiver() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  int port() const {
    return port_;
  }

  bool recv_one(std::string* payload, int timeout_ms = 2000) const {
    if (!payload)
      return false;
    payload->clear();

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd_, &rfds);

    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    const int ready = ::select(fd_ + 1, &rfds, nullptr, nullptr, &tv);
    if (ready <= 0)
      return false;

    std::array<char, 4096> buf{};
    const ssize_t n = ::recvfrom(fd_, buf.data(), buf.size(), 0, nullptr, nullptr);
    if (n <= 0)
      return false;

    payload->assign(buf.data(), static_cast<size_t>(n));
    return true;
  }

private:
  int fd_ = -1;
  int port_ = -1;
};

} // namespace

RUN_TEST(
    "unit_optiview_json_output_udp_test", ([] {
      using nlohmann::json;

      UdpReceiver rx;

      simaai::neat::OptiViewChannelOptions opt;
      opt.host = "127.0.0.1";
      opt.channel = 0;
      opt.json_port_base = rx.port();
      opt.video_port_base = 9200;

      std::string init_err;
      simaai::neat::OptiViewJsonOutput sender(opt, &init_err);
      require(sender.ok(), "OptiViewJsonOutput init failed: " + init_err);
      require(sender.json_port() == rx.port(), "OptiViewJsonOutput json_port mismatch");
      require(sender.video_port() == 9200, "OptiViewJsonOutput video_port mismatch");

      const std::string payload = R"({"type":"health","ok":true})";
      std::string send_err;
      require(sender.send_json(payload, &send_err),
              "OptiViewJsonOutput send_json failed: " + send_err);

      std::string received;
      require(rx.recv_one(&received, 2000),
              "OptiViewJsonOutput send_json payload not received over UDP");
      require(received == payload, "OptiViewJsonOutput send_json payload mismatch");

      std::vector<simaai::neat::OptiViewObject> objects = {
          simaai::neat::OptiViewObject{
              .x = 10, .y = 20, .w = 30, .h = 40, .score = 0.95f, .class_id = 0},
          simaai::neat::OptiViewObject{
              .x = 1, .y = 2, .w = 3, .h = 4, .score = 0.25f, .class_id = 42},
      };
      const std::vector<std::string> labels = {"person"};

      require(sender.send_detection(12345, "frame-7", objects, labels, &send_err),
              "OptiViewJsonOutput send_detection failed: " + send_err);
      require(rx.recv_one(&received, 2000),
              "OptiViewJsonOutput send_detection payload not received over UDP");

      const json parsed = json::parse(received);
      require(parsed["type"].get<std::string>() == "object-detection",
              "OptiView JSON type mismatch");
      require(parsed["timestamp"].get<int64_t>() == 12345, "OptiView JSON timestamp mismatch");
      require(parsed["frame_id"].get<std::string>() == "frame-7",
              "OptiView JSON frame_id mismatch");
      require(parsed["data"]["objects"].is_array(), "OptiView JSON objects should be an array");
      require(parsed["data"]["objects"].size() == 2, "OptiView JSON object count mismatch");
      require(parsed["data"]["objects"][0]["label"].get<std::string>() == "person",
              "OptiView JSON known label mismatch");
      require(parsed["data"]["objects"][1]["label"].get<std::string>() == "Unknown",
              "OptiView JSON unknown class should map to Unknown");

      const auto default_labels = simaai::neat::OptiViewDefaultLabels();
      require(default_labels.size() == 80, "OptiViewDefaultLabels should emit 80 labels");
      require(default_labels.front() == "label_0", "OptiViewDefaultLabels first label mismatch");

      simaai::neat::OptiViewChannelOptions bad_opt;
      bad_opt.host = "256.256.256.256";
      bad_opt.channel = 0;
      bad_opt.json_port_base = 9300;

      std::string bad_init_err;
      simaai::neat::OptiViewJsonOutput bad_sender(bad_opt, &bad_init_err);
      require(!bad_sender.ok(), "OptiViewJsonOutput should fail on invalid host");
      require_contains(bad_init_err, "getaddrinfo", "OptiViewJsonOutput bad-host error mismatch");

      std::string bad_send_err;
      require(!bad_sender.send_json("{}", &bad_send_err),
              "OptiViewJsonOutput should reject send_json when uninitialized");
      require_contains(bad_send_err, "not initialized",
                       "OptiViewJsonOutput uninitialized send error mismatch");
    }));
