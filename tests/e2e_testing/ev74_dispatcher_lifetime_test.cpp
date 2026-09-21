#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"
#include "test_utils.h"

#include <opencv2/core.hpp>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> inject_fault{false};
std::atomic<bool> pause_write{false};
std::atomic<bool> write_gate_timed_out{false};
std::binary_semaphore write_entered{0};
std::binary_semaphore resume_write{0};

} // namespace

// Test-only transport fault: reject before sending anything to firmware.
extern "C" ssize_t write(int fd, const void* data, size_t size) {
  if (inject_fault.load() || pause_write.load()) {
    char link[64];
    char target[128];
    std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    const auto length = ::readlink(link, target, sizeof(target));
    if (length >= 10 && std::memcmp(target, "/dev/rpmsg", 10) == 0) {
      if (pause_write.exchange(false)) {
        write_entered.release();
        if (!resume_write.try_acquire_for(std::chrono::seconds(3))) {
          write_gate_timed_out.store(true);
          errno = EBADF;
          return -1;
        }
      }
      if (inject_fault.load()) {
        errno = EBADF;
        return -1;
      }
    }
  }
  return ::syscall(SYS_write, fd, data, size);
}

namespace {

simaai::neat::Run build_run() {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
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
  run_options.startup_preflight = false;
  return graph.build(simaai::neat::TensorList{input}, run_options);
}

// Probe actual locks, not advisory owner-metadata files. Separate opens in a
// thread must conflict too; process-owned lockf locks do not provide that.
bool channel_free(const std::string& path) {
  const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0660);
  require(fd >= 0, "Cannot open channel lock: " + path);
  const int rc = ::lockf(fd, F_TLOCK, 0);
  const int error = errno;
  ::close(fd);
  require(rc == 0 || error == EACCES || error == EAGAIN,
          "Unexpected channel lock failure: " + std::to_string(error));
  return rc == 0;
}

std::vector<std::string> channel_paths() {
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator("/sys/class/rpmsg")) {
    const auto name = entry.path().filename().string();
    if (name.starts_with("rpmsg") && name.size() > 5 &&
        name.find_first_not_of("0123456789", 5) == std::string::npos) {
      paths.push_back("/tmp/rpmsg_lock_" + name);
    }
  }
  std::sort(paths.begin(), paths.end());
  require(paths.size() >= 2, "Need at least two discovered RPMsg endpoints");
  return paths;
}

size_t free_channels(const std::vector<std::string>& paths) {
  return std::count_if(paths.begin(), paths.end(), channel_free);
}

void expect_reserved(const std::vector<std::string>& paths, size_t count) {
  const auto available = std::async(std::launch::async, [&] { return free_channels(paths); }).get();
  require(available == paths.size() - count, "Wrong reservation count before/after graph close");
  // A failed same-process probe must not drop the owner's lock.
  require(free_channels(paths) == available, "Closing a competing descriptor released ownership");
}

void expect_external_reservation(const std::vector<std::string>& paths, size_t count) {
  const pid_t child = ::fork();
  require(child >= 0, "Cannot fork lock probe");
  if (child == 0) {
    // Only async-signal-safe operations after fork from the live graph process.
    size_t available = 0;
    for (const auto& path : paths) {
      const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
      if (fd < 0)
        ::_exit(2);
      struct flock lock {};
      lock.l_type = F_WRLCK;
      lock.l_whence = SEEK_SET;
      const int rc = ::fcntl(fd, F_SETLK, &lock);
      const int error = errno;
      ::close(fd);
      if (rc == 0)
        ++available;
      else if (error != EACCES && error != EAGAIN)
        ::_exit(3);
    }
    ::_exit(available == paths.size() - count ? 0 : 1);
  }
  int status = 0;
  require(::waitpid(child, &status, 0) == child, "Cannot wait for lock probe");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "Competing process acquired a reserved channel");
}

class HoldChannels {
public:
  HoldChannels(const std::vector<std::string>& paths, size_t count) {
    try {
      for (size_t i = 0; i < count; ++i) {
        int fd = ::open(paths.at(i).c_str(), O_RDWR | O_CLOEXEC);
        require(fd >= 0, "Cannot open holder lock");
        fds_.push_back(fd);
        struct flock lock {};
        lock.l_type = F_WRLCK;
        lock.l_whence = SEEK_SET;
        require(::fcntl(fd, F_OFD_SETLK, &lock) == 0,
                "Endpoint already owned before capacity test");
      }
    } catch (...) {
      close();
      throw;
    }
  }
  ~HoldChannels() {
    close();
  }
  HoldChannels(const HoldChannels&) = delete;
  HoldChannels& operator=(const HoldChannels&) = delete;

private:
  void close() {
    for (int fd : fds_)
      ::close(fd);
    fds_.clear();
  }
  std::vector<int> fds_;
};

void require_neat_failure(const std::function<void()>& action, const std::string& message) {
  try {
    action();
  } catch (const simaai::neat::NeatError&) {
    return;
  }
  require(false, message);
}

void require_capacity_failure() {
  try {
    auto run = build_run();
  } catch (const simaai::neat::NeatError& error) {
    require(error.report().error_code == simaai::neat::error_codes::kDispatcherUnavailable,
            "Wrong startup error code");
    require_contains(error.report().to_json(), "RPMsg capacity exhausted",
                     "Missing resource-specific bus diagnostic");
    return;
  }
  require(false, "Build succeeded with exhausted channels");
}

void execute(simaai::neat::Run& run) {
  using namespace simaai::neat;
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
  auto input = Tensor::from_cv_mat(image, ImageSpec::PixelFormat::RGB, TensorMemory::EV74);
  auto output = run.run(TensorList{input}, 10000);
  require(output.size() == 1, "Missing EV74 output");
  require(output.front().shape[0] == 640 && output.front().shape[1] == 640,
          "Wrong EV74 output dimensions");
}

void test_concurrent_close(int old_clients) {
  ::setenv("SIMA_EVXX_DISPATCHER_WORKERS", "2", 1);
  ::setenv("SIMA_DISPATCHER_EAGER_RELEASE", "0", 1);
  ::setenv("SIMA_RPMSG_ACQUIRE_TIMEOUT_MS", "0", 1);
  ::setenv("SIMA_RPCEVXX_AUTO_REMOTEPROC_RESET", "0", 1);
  ::setenv("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0", 1);
  ::setenv("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0", 1);
  ::setenv("SIMA_GST_TEARDOWN_ASYNC", "0", 1);
  for (int cycle = 0; cycle < 50; ++cycle) {
    std::vector<simaai::neat::Run> old_runs;
    for (int client = 0; client < old_clients; ++client)
      old_runs.push_back(build_run());
    std::vector<std::future<void>> closing;
    for (auto& old : old_runs)
      closing.push_back(
          std::async(std::launch::async, [run = std::move(old)]() mutable { run.close(); }));
    // Submit before joining the old clients: their teardown must not stop this client.
    auto replacement = build_run();
    execute(replacement);
    for (auto& closed : closing)
      closed.get();
    std::cout << "Concurrent close: old_clients=" << old_clients << " cycle=" << cycle << '\n';
  }
}

void test_lifetime() {
  ::setenv("SIMA_EVXX_DISPATCHER_WORKERS", "1", 1);
  ::setenv("SIMA_DISPATCHER_EAGER_RELEASE", "0", 1);
  ::setenv("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0", 1);
  ::setenv("SIMA_RPMSG_ACQUIRE_TIMEOUT_MS", "0", 1);
  ::setenv("SIMA_RPCEVXX_AUTO_REMOTEPROC_RESET", "0", 1);
  for (const char* idle : {"0", "10"}) {
    ::setenv("SIMA_RPCEVXX_IDLE_RELEASE_MS", idle, 1);
    for (int cycle = 0; cycle < 3; ++cycle) {
      auto first = build_run();
      const auto paths = channel_paths();
      expect_reserved(paths, 1);
      auto second = build_run();
      expect_reserved(paths, 1);
      first.close();
      execute(second);
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      expect_reserved(paths, 1);
      expect_external_reservation(paths, 1);
      second.close();
      expect_reserved(paths, 0);
      expect_external_reservation(paths, 0);
    }
  }
  const auto paths = channel_paths();
  {
    HoldChannels holders(paths, paths.size());
    require_capacity_failure();
  }
  expect_reserved(paths, 0);
  {
    auto retry = build_run();
    expect_reserved(paths, 1);
    retry.close();
  }

  ::setenv("SIMA_EVXX_DISPATCHER_WORKERS", "2", 1);
  {
    HoldChannels holders(paths, paths.size() - 1);
    require_neat_error([] { auto run = build_run(); },
                       simaai::neat::error_codes::kDispatcherUnavailable);
    require(channel_free(paths.back()), "Partial reservation leaked its first channel");
  }
  {
    HoldChannels holders(paths, paths.size() - 2);
    auto run = build_run();
    expect_reserved(paths, paths.size());
    execute(run);
    run.close();
    require(channel_free(paths[paths.size() - 2]) && channel_free(paths.back()),
            "Two-worker teardown leaked channels");
  }
  expect_reserved(paths, 0);

  {
    auto failed = build_run();
    std::vector<simaai::neat::Run> shared;
    for (int i = 0; i < 4; ++i)
      shared.push_back(build_run());
    inject_fault.store(true);
    bool threw = false;
    try {
      execute(failed);
    } catch (const simaai::neat::NeatError&) {
      threw = true;
    }
    inject_fault.store(false);
    require(threw, "Injected transport failure did not reach Core");
    require_neat_failure([] { auto rejected = build_run(); },
                         "Faulted dispatcher accepted a graph");
    for (auto& run : shared) {
      require_neat_failure([&] { execute(run); },
                           "Shared dispatcher executed after a transport fault");
      run.close();
    }
    failed.close();
  }
  expect_reserved(paths, 0);
  auto rebuilt = build_run();
  execute(rebuilt);
  rebuilt.close();
  expect_reserved(paths, 0);
}

void test_default_timeout() {
  ::unsetenv("SIMA_RPMSG_ACQUIRE_TIMEOUT_MS");
  ::setenv("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0", 1);
  auto control = build_run();
  control.close();
  const auto paths = channel_paths();
  HoldChannels holders(paths, paths.size());
  const auto start = std::chrono::steady_clock::now();
  require_capacity_failure();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  require(elapsed >= std::chrono::seconds(14) && elapsed < std::chrono::seconds(50),
          "Default acquisition did not fail before Core's outer timeout");
}

size_t process_resources(const char* path) {
  return std::distance(std::filesystem::directory_iterator(path),
                       std::filesystem::directory_iterator{});
}

void close_with_work_in_flight(const std::vector<std::string>& paths) {
  auto run = build_run();
  pause_write.store(true);
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
  require(run.push(std::vector<cv::Mat>{image}), "Cannot enqueue teardown input");
  if (!write_entered.try_acquire_for(std::chrono::seconds(5))) {
    pause_write.store(false);
    // Also unblock a writer that reached the gate at the deadline.
    resume_write.release();
    require(false, "Dispatcher did not reach the teardown write gate");
  }
  std::promise<void> closing;
  auto started = closing.get_future();
  auto closed = std::async(std::launch::async, [run = std::move(run), &closing]() mutable {
    closing.set_value();
    run.close();
    run.close();
  });
  started.wait();
  const bool waited =
      closed.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
  // Always release the test gate before an assertion can unwind the future.
  const auto reserved = paths.size() - free_channels(paths);
  resume_write.release();
  closed.get();
  require(!write_gate_timed_out.load(), "Teardown test did not release its write gate in time");
  require(waited, "Teardown returned while a dispatcher worker was still active");
  require(reserved == 2, "Teardown released a channel while a worker was active");
  expect_reserved(paths, 0);
}

void test_destructor_reliability(int cycles) {
  ::setenv("SIMA_EVXX_DISPATCHER_WORKERS", "2", 1);
  ::setenv("SIMA_DISPATCHER_EAGER_RELEASE", "0", 1);
  ::setenv("SIMA_RPMSG_ACQUIRE_TIMEOUT_MS", "0", 1);
  ::setenv("SIMA_RPCEVXX_AUTO_REMOTEPROC_RESET", "0", 1);
  ::setenv("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0", 1);
  ::unsetenv("SIMA_RPCEVXX_IDLE_RELEASE_MS");
  // Wait for actual destruction instead of delegating it to Core's reaper.
  ::setenv("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0", 1);
  ::setenv("SIMA_GST_TEARDOWN_ASYNC", "0", 1);
  // Warm lazy runtime state before measuring retained descriptors and threads.
  for (int i = 0; i < 3; ++i) {
    auto warm = build_run();
    execute(warm);
  }
  const auto paths = channel_paths();
  close_with_work_in_flight(paths);
  size_t baseline_fds = 0;
  size_t baseline_threads = 0;
  const auto start = std::chrono::steady_clock::now();
  // The first iteration warms shared task pools at the measured concurrency.
  for (int cycle = -1; cycle < cycles; ++cycle) {
    {
      auto first = build_run();
      auto second = build_run();
      auto survivor = build_run();
      expect_reserved(paths, 2);
      if (cycle % 2 != 0) {
        execute(survivor);
        survivor.close();
      }
      auto first_close =
          std::async(std::launch::async, [run = std::move(first)]() mutable { run.close(); });
      auto second_close =
          std::async(std::launch::async, [run = std::move(second)]() mutable { run.close(); });
      if (survivor) {
        first_close.get();
        second_close.get();
        execute(survivor);
        expect_reserved(paths, 2);
      } else {
        // Race a new client against concurrent releases of the final two.
        auto acquire = std::async(std::launch::async, [] {
          auto replacement = build_run();
          execute(replacement);
        });
        first_close.get();
        second_close.get();
        acquire.get();
      }
      // The final client is released by Run's destructor, not explicit close.
    }
    expect_reserved(paths, 0);
    expect_external_reservation(paths, 0);
    {
      // Constructor rollback must join its threads and release its partial RPC.
      HoldChannels holders(paths, paths.size() - 1);
      require_capacity_failure();
    }
    expect_reserved(paths, 0);
    {
      auto unused = build_run();
      // Destruction without submitting any work must release both workers too.
    }
    expect_reserved(paths, 0);
    if (cycle % 5 == 0)
      close_with_work_in_flight(paths);
    if (cycle == -1) {
      baseline_fds = process_resources("/proc/self/fd");
      baseline_threads = process_resources("/proc/self/task");
    }
  }
  const auto settle = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (process_resources("/proc/self/task") > baseline_threads &&
         std::chrono::steady_clock::now() < settle)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const auto final_fds = process_resources("/proc/self/fd");
  const auto final_threads = process_resources("/proc/self/task");
  std::cout << "Destructor reliability: cycles=" << cycles << " fds=" << baseline_fds << "->"
            << final_fds << " threads=" << baseline_threads << "->" << final_threads
            << " elapsed_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start)
                   .count()
            << '\n';
  require(final_fds <= baseline_fds, "Dispatcher teardown leaked descriptors");
  require(final_threads <= baseline_threads, "Dispatcher teardown leaked threads");
  auto recovered = build_run();
  execute(recovered);
  recovered.close();
  expect_reserved(paths, 0);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--default-timeout")
      test_default_timeout();
    else if (argc == 2 && std::string(argv[1]) == "--destructor-reliability")
      test_destructor_reliability(100);
    else if (argc == 3 && std::string(argv[1]) == "--concurrent-close") {
      require(std::string(argv[2]) == "1" || std::string(argv[2]) == "2",
              "Concurrent close requires one or two old clients");
      test_concurrent_close(std::stoi(argv[2]));
    } else {
      test_lifetime();
      test_destructor_reliability(20);
    }
    std::cout << "[OK] ev74_dispatcher_lifetime_test passed\n";
    return 0;
  } catch (const simaai::neat::NeatError& error) {
    std::cerr << "[FAIL] " << error.report().to_json() << "\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
