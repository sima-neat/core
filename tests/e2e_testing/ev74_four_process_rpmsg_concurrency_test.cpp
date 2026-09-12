#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "gst/GstHelpers.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t kChildCount = 4;
constexpr int kPhaseTimeoutMs = 30000;

struct ChildMessage {
  char state = '\0';
  char detail[255] = {};
};

struct ChildProcess {
  pid_t pid = -1;
  int socket = -1;
};

std::string errno_message(const std::string& action) {
  return action + ": " + std::strerror(errno);
}

void close_fd(int& fd) {
  if (fd >= 0) {
    (void)::close(fd);
    fd = -1;
  }
}

bool write_all(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  while (size > 0) {
    const ssize_t written = ::write(fd, bytes, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    bytes += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool read_all(int fd, void* data, size_t size) {
  auto* bytes = static_cast<unsigned char*>(data);
  while (size > 0) {
    const ssize_t received = ::read(fd, bytes, size);
    if (received < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (received == 0) {
      return false;
    }
    bytes += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

void send_message(int fd, char state, const std::string& detail = {}) {
  ChildMessage message;
  message.state = state;
  std::snprintf(message.detail, sizeof(message.detail), "%s", detail.c_str());
  if (!write_all(fd, &message, sizeof(message))) {
    throw std::runtime_error(errno_message("write child status"));
  }
}

void send_command(int fd, char command) {
  if (!write_all(fd, &command, sizeof(command))) {
    throw std::runtime_error(errno_message("write child command"));
  }
}

char receive_command(int fd) {
  char command = '\0';
  if (!read_all(fd, &command, sizeof(command))) {
    throw std::runtime_error(errno_message("read parent command"));
  }
  return command;
}

std::array<uint8_t, 3> expected_rgb(size_t child_index, size_t round) {
  const auto base = static_cast<uint8_t>(16U + child_index * 8U + round * 32U);
  return {base, static_cast<uint8_t>(base + 16U), static_cast<uint8_t>(base + 32U)};
}

void validate_output(const simaai::neat::TensorList& outputs, size_t child_index, size_t round) {
  require(outputs.size() == 1, "Preproc output missing tensor");
  const simaai::neat::Tensor& tensor = outputs.front();
  require(tensor.shape.size() == 3, "Preproc output missing RGB shape");
  require(tensor.shape[0] == 640 && tensor.shape[1] == 640 && tensor.shape[2] == 3,
          "Preproc size mismatch");
  require(tensor.dtype == simaai::neat::TensorDType::UInt8 ||
              tensor.dtype == simaai::neat::TensorDType::Int8,
          "Preproc dtype mismatch");

  const auto map = tensor.map_read();
  constexpr size_t expected = 640U * 640U * 3U;
  require(map.data != nullptr && map.size_bytes >= expected, "Preproc bytes missing");
  std::array<size_t, 3> strides{640U * 3U, 3U, 1U};
  if (!tensor.strides_bytes.empty()) {
    require(tensor.strides_bytes.size() == strides.size(), "Preproc RGB strides missing");
    for (size_t axis = 0; axis < strides.size(); ++axis) {
      require(tensor.strides_bytes[axis] > 0, "Preproc RGB stride must be positive");
      strides[axis] = static_cast<size_t>(tensor.strides_bytes[axis]);
    }
  }
  size_t span = 1;
  for (size_t axis = 0; axis < strides.size(); ++axis) {
    const auto steps = static_cast<size_t>(tensor.shape[axis] - 1);
    require(strides[axis] <= (map.size_bytes - span) / steps,
            "Preproc RGB strides exceed mapped bytes");
    span += strides[axis] * steps;
  }
  const auto rgb = expected_rgb(child_index, round);
  const auto* pixels = static_cast<const uint8_t*>(map.data);
  // map_read() already applies the Tensor's byte offset. Only logical RGB bytes matter.
  for (size_t y = 0; y < 640U; ++y) {
    for (size_t x = 0; x < 640U; ++x) {
      for (size_t channel = 0; channel < rgb.size(); ++channel) {
        const auto value = pixels[y * strides[0] + x * strides[1] + channel * strides[2]];
        if (value != rgb[channel]) {
          throw std::runtime_error("Child " + std::to_string(child_index) + " round " +
                                   std::to_string(round) + " RGB mismatch at " + std::to_string(x) +
                                   "," + std::to_string(y) + " channel " + std::to_string(channel) +
                                   ": expected=" + std::to_string(rgb[channel]) +
                                   " actual=" + std::to_string(value));
        }
      }
    }
  }
}

int run_child(int socket, size_t child_index) {
  try {
    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing SIMA preproc plugin (neatprocesscvu)");

    const auto initial_rgb = expected_rgb(child_index, 0);
    cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(initial_rgb[0], initial_rgb[1], initial_rgb[2]));
    if (!image.isContinuous()) {
      image = image.clone();
    }
    simaai::neat::Tensor input = simaai::neat::Tensor::from_cv_mat(
        image, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);

    simaai::neat::InputOptions input_options;
    input_options.format = simaai::neat::FormatTag::RGB;
    input_options.width = image.cols;
    input_options.height = image.rows;
    input_options.depth = 3;
    input_options.is_live = true;
    input_options.do_timestamp = true;
    input_options.block = false;
    input_options.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
    input_options.pool_min_buffers = 4;
    input_options.pool_max_buffers = 4;
    input_options.buffer_name = "decoder";

    simaai::neat::PreprocOptions preproc_options;
    preproc_options.set_input_shape({image.rows, image.cols, 3});
    preproc_options.set_output_shape({640, 640, 3});
    preproc_options.scaled_width = 640;
    preproc_options.scaled_height = 640;
    preproc_options.input_img_type = "RGB";
    preproc_options.output_img_type = "RGB";
    preproc_options.normalize = false;
    preproc_options.aspect_ratio = false;
    preproc_options.output_dtype = "EVXX_INT8";
    preproc_options.scaling_type = "BILINEAR";
    preproc_options.padding_type = "CENTER";
    preproc_options.next_cpu = "APU";
    preproc_options.upstream_name = "decoder";
    preproc_options.num_buffers = input_options.pool_min_buffers;
    preproc_options.set_slice_shape({32, 128, 3});
    preproc_options.q_scale = 0.25;
    preproc_options.q_zp = 0;

    simaai::neat::OutputOptions output_options;
    output_options.sync = false;
    output_options.drop = true;
    output_options.max_buffers = 1;

    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input(input_options));
    graph.add(simaai::neat::nodes::Preproc(preproc_options));
    graph.add(simaai::neat::nodes::Output(output_options));

    simaai::neat::RunOptions run_options;
    run_options.queue_depth = 1;
    auto run = graph.build(simaai::neat::TensorList{input}, run_options);

    send_message(socket, 'R');
    for (size_t round = 0;; ++round) {
      const char command = receive_command(socket);
      if (command == 'X') {
        require(round > 0, "Expected a run before parent close command");
        break;
      }
      require(command == 'G' && round < 2, "Expected parent run or close command");
      if (round != 0) {
        const auto rgb = expected_rgb(child_index, round);
        image.setTo(cv::Scalar(rgb[0], rgb[1], rgb[2]));
        input = simaai::neat::Tensor::from_cv_mat(image, simaai::neat::ImageSpec::PixelFormat::RGB,
                                                  simaai::neat::TensorMemory::EV74);
      }
      validate_output(run.run(simaai::neat::TensorList{input}, kPhaseTimeoutMs), child_index,
                      round);
      send_message(socket, 'S');
    }

    run.close();
    send_message(socket, 'C');
    return 0;
  } catch (const std::exception& error) {
    try {
      send_message(socket, 'F', error.what());
    } catch (...) {
    }
    return 1;
  } catch (...) {
    try {
      send_message(socket, 'F', "unknown child failure");
    } catch (...) {
    }
    return 1;
  }
}

class ChildGuard {
public:
  explicit ChildGuard(std::array<ChildProcess, kChildCount>& children) : children_(children) {}

  ~ChildGuard() {
    for (ChildProcess& child : children_) {
      close_fd(child.socket);
      if (child.pid > 0) {
        (void)::kill(child.pid, SIGTERM);
      }
    }

    for (int attempt = 0; attempt < 50; ++attempt) {
      bool waiting = false;
      for (ChildProcess& child : children_) {
        if (child.pid <= 0) {
          continue;
        }
        const pid_t result = ::waitpid(child.pid, nullptr, WNOHANG);
        if (result == child.pid || (result < 0 && errno == ECHILD)) {
          child.pid = -1;
        } else {
          waiting = true;
        }
      }
      if (!waiting) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    for (ChildProcess& child : children_) {
      if (child.pid > 0) {
        (void)::kill(child.pid, SIGKILL);
        (void)::waitpid(child.pid, nullptr, 0);
        child.pid = -1;
      }
    }
  }

private:
  std::array<ChildProcess, kChildCount>& children_;
};

void await_state(std::span<const ChildProcess> children, char expected) {
  std::set<size_t> pending;
  for (size_t i = 0; i < children.size(); ++i) {
    pending.insert(i);
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kPhaseTimeoutMs);
  while (!pending.empty()) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      throw std::runtime_error(std::string("Timed out waiting for child state ") + expected);
    }

    std::vector<pollfd> poll_fds;
    std::vector<size_t> child_indexes;
    for (const size_t index : pending) {
      poll_fds.push_back({children[index].socket, POLLIN, 0});
      child_indexes.push_back(index);
    }
    const int remaining_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
    const int result = ::poll(poll_fds.data(), poll_fds.size(), remaining_ms);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(errno_message("poll child status"));
    }
    if (result == 0) {
      throw std::runtime_error(std::string("Timed out waiting for child state ") + expected);
    }

    for (size_t i = 0; i < poll_fds.size(); ++i) {
      if ((poll_fds[i].revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
        continue;
      }
      ChildMessage message;
      if (!read_all(poll_fds[i].fd, &message, sizeof(message))) {
        throw std::runtime_error("Child exited before reporting its state");
      }
      if (message.state == 'F') {
        throw std::runtime_error("Child PID " + std::to_string(children[child_indexes[i]].pid) +
                                 " failed: " + message.detail);
      }
      if (message.state != expected) {
        throw std::runtime_error("Child PID " + std::to_string(children[child_indexes[i]].pid) +
                                 " reported unexpected state");
      }
      pending.erase(child_indexes[i]);
    }
  }
}

void run_test() {
  std::array<std::array<int, 2>, kChildCount> sockets;
  for (auto& pair : sockets) {
    pair = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, pair.data()) != 0) {
      throw std::runtime_error(errno_message("create child socket"));
    }
  }

  std::array<ChildProcess, kChildCount> children;
  ChildGuard guard(children);
  for (size_t child_index = 0; child_index < kChildCount; ++child_index) {
    const pid_t pid = ::fork();
    if (pid < 0) {
      throw std::runtime_error(errno_message("fork child"));
    }
    if (pid == 0) {
      for (size_t socket_index = 0; socket_index < sockets.size(); ++socket_index) {
        close_fd(sockets[socket_index][0]);
        if (socket_index != child_index) {
          close_fd(sockets[socket_index][1]);
        }
      }
      const int child_socket = sockets[child_index][1];
      const int result = run_child(child_socket, child_index);
      (void)::close(child_socket);
      ::_exit(result);
    }
    children[child_index] = {pid, sockets[child_index][0]};
  }
  for (auto& pair : sockets) {
    pair[0] = -1;
    close_fd(pair[1]);
  }

  await_state(children, 'R');
  for (const ChildProcess& child : children) {
    send_command(child.socket, 'G');
  }
  await_state(children, 'S');
  const auto close_children = [](std::span<ChildProcess> retiring) {
    for (const ChildProcess& child : retiring) {
      send_command(child.socket, 'X');
    }
    await_state(retiring, 'C');
    for (ChildProcess& child : retiring) {
      close_fd(child.socket);
      int status = 0;
      if (::waitpid(child.pid, &status, 0) != child.pid) {
        throw std::runtime_error(errno_message("wait for child"));
      }
      child.pid = -1;
      require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "EV74 child exited unsuccessfully");
    }
  };

  // Retiring one completed client must not disrupt the other clients' next results.
  std::span<ChildProcess> active(children);
  close_children(active.first(1));
  active = active.subspan(1);
  for (const ChildProcess& child : active) {
    send_command(child.socket, 'G');
  }
  await_state(active, 'S');
  close_children(active);
}

} // namespace

int main() {
  (void)::signal(SIGPIPE, SIG_IGN);
  try {
    run_test();
    std::cout << "[OK] ev74_four_process_rpmsg_concurrency_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
