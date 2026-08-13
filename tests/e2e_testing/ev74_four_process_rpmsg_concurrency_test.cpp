#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "gst/GstHelpers.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <dirent.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
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

void validate_output(const simaai::neat::TensorList& outputs) {
  require(outputs.size() == 1, "Preproc output missing tensor");
  const simaai::neat::Tensor& tensor = outputs.front();
  require(tensor.shape.size() >= 2, "Preproc output missing shape");
  require(tensor.shape[0] == 640 && tensor.shape[1] == 640, "Preproc size mismatch");
  require(tensor.dtype == simaai::neat::TensorDType::UInt8 ||
              tensor.dtype == simaai::neat::TensorDType::Int8,
          "Preproc dtype mismatch");

  simaai::neat::Tensor cpu = tensor.clone();
  simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
  constexpr size_t expected = 640U * 640U * 3U;
  require(map.data != nullptr && map.size_bytes >= expected, "Preproc bytes missing");
}

int run_child(int socket) {
  try {
    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing SIMA preproc plugin (neatprocesscvu)");

    cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
    if (!image.isContinuous()) {
      image = image.clone();
    }
    const simaai::neat::Tensor input = simaai::neat::Tensor::from_cv_mat(
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
    run_options.output_memory = simaai::neat::OutputMemory::Owned;
    run_options.queue_depth = 1;
    auto run = graph.build(simaai::neat::TensorList{input}, run_options);

    send_message(socket, 'R');
    require(receive_command(socket) == 'G', "Expected parent run command");

    validate_output(run.run(simaai::neat::TensorList{input}, kPhaseTimeoutMs));
    send_message(socket, 'S');
    require(receive_command(socket) == 'X', "Expected parent close command");

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

void await_state(const std::array<ChildProcess, kChildCount>& children, char expected) {
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
        throw std::runtime_error("Child " + std::to_string(child_indexes[i]) +
                                 " failed: " + message.detail);
      }
      if (message.state != expected) {
        throw std::runtime_error("Child " + std::to_string(child_indexes[i]) +
                                 " reported unexpected state");
      }
      pending.erase(child_indexes[i]);
    }
  }
}

std::map<std::string, std::string> read_key_values(const std::string& path) {
  std::ifstream stream(path);
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(stream, line)) {
    const size_t separator = line.find('=');
    if (separator != std::string::npos) {
      values.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
  }
  return values;
}

void verify_one_unique_channel_per_child(const std::array<ChildProcess, kChildCount>& children) {
  std::set<pid_t> child_pids;
  for (const ChildProcess& child : children) {
    child_pids.insert(child.pid);
  }

  std::map<pid_t, std::set<std::string>> nodes_by_pid;
  DIR* directory = ::opendir("/tmp");
  if (directory == nullptr) {
    throw std::runtime_error(errno_message("open /tmp"));
  }
  while (dirent* entry = ::readdir(directory)) {
    const std::string name = entry->d_name;
    if (name.rfind("rpmsg_lock_rpmsg", 0) != 0 || name.size() < std::strlen(".owner") ||
        name.compare(name.size() - std::strlen(".owner"), std::strlen(".owner"), ".owner") != 0) {
      continue;
    }
    const auto values = read_key_values("/tmp/" + name);
    const auto pid_it = values.find("pid");
    const auto node_it = values.find("node");
    if (pid_it == values.end() || node_it == values.end()) {
      continue;
    }
    try {
      const pid_t pid = static_cast<pid_t>(std::stol(pid_it->second));
      if (child_pids.count(pid) != 0) {
        nodes_by_pid[pid].insert(node_it->second);
      }
    } catch (const std::exception&) {
    }
  }
  (void)::closedir(directory);

  std::set<std::string> all_nodes;
  for (const pid_t pid : child_pids) {
    const auto found = nodes_by_pid.find(pid);
    require(found != nodes_by_pid.end(),
            "Missing RPMsg owner metadata for child " + std::to_string(pid));
    require(found->second.size() == 1,
            "Child " + std::to_string(pid) + " does not own exactly one RPMsg channel");
    all_nodes.insert(*found->second.begin());
  }
  require(all_nodes.size() == kChildCount,
          "Four live EV74 processes did not own four distinct RPMsg channels");
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
      const int result = run_child(child_socket);
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
  verify_one_unique_channel_per_child(children);

  for (const ChildProcess& child : children) {
    send_command(child.socket, 'X');
  }
  await_state(children, 'C');

  for (ChildProcess& child : children) {
    close_fd(child.socket);
    int status = 0;
    if (::waitpid(child.pid, &status, 0) != child.pid) {
      throw std::runtime_error(errno_message("wait for child"));
    }
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "EV74 child exited unsuccessfully");
    child.pid = -1;
  }
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
