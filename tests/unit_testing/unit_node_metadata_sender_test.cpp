#include "nodes/io/MetadataSender.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <netinet/in.h>
#include <string>
#include <thread>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

bool install_sendto_flag_probe() {
  // Turn sendto into a deterministic probe: a call carrying MSG_DONTWAIT gets
  // EAGAIN, while a call without it gets EPERM. All other syscalls are allowed.
  const sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_sendto, 0, 5),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[3])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, MSG_DONTWAIT),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, MSG_DONTWAIT, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EAGAIN),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  const sock_fprog program = {static_cast<unsigned short>(std::size(filter)),
                              const_cast<sock_filter*>(filter)};
  return ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
         ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}

bool install_sendto_error_probe(int error_number) {
  const sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_sendto, 0, 1),
      BPF_STMT(BPF_RET | BPF_K,
               SECCOMP_RET_ERRNO | (static_cast<uint32_t>(error_number) & SECCOMP_RET_DATA)),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  const sock_fprog program = {static_cast<unsigned short>(std::size(filter)),
                              const_cast<sock_filter*>(filter)};
  return ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
         ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}

int probe_send_mode_in_child(bool nonblocking) {
  simaai::neat::MetadataSenderOptions opt;
  opt.host = "127.0.0.1";
  opt.metadata_port_base = 9199;
  simaai::neat::MetadataSenderSendOptions send_opt;
  send_opt.nonblocking = nonblocking;

  std::string init_err;
  simaai::neat::MetadataSender sender(opt, send_opt, &init_err);
  if (!sender.ok() || sender.nonblocking() != nonblocking || !install_sendto_flag_probe()) {
    return 10;
  }

  std::string send_err = "stale error";
  if (sender.send_raw_json("{}", &send_err)) {
    return 11;
  }
  const auto stats = sender.stats();
  if (stats.send_attempts != 1 || stats.datagrams_sent != 0 || stats.send_failures != 1) {
    return 12;
  }
  const int expected_errno = nonblocking ? EAGAIN : EPERM;
  if (stats.last_errno != expected_errno) {
    return 13;
  }
  if (stats.would_block != (nonblocking ? 1U : 0U) || stats.no_buffer_space != 0) {
    return 14;
  }
  if (send_err.empty() != nonblocking) {
    return 15;
  }
  return 0;
}

int probe_default_send_mode_in_child() {
  simaai::neat::MetadataSenderOptions opt;
  opt.host = "127.0.0.1";
  opt.metadata_port_base = 9199;

  std::string init_err;
  simaai::neat::MetadataSender sender(opt, &init_err);
  if (!sender.ok() || !sender.nonblocking() || !install_sendto_flag_probe()) {
    return 30;
  }

  std::string send_err = "stale error";
  if (sender.send_raw_json("{}", &send_err)) {
    return 31;
  }
  const auto stats = sender.stats();
  if (stats.send_attempts != 1 || stats.datagrams_sent != 0 || stats.send_failures != 1 ||
      stats.last_errno != EAGAIN || stats.would_block != 1 || stats.no_buffer_space != 0) {
    return 32;
  }
  if (!send_err.empty()) {
    return 33;
  }
  return 0;
}

void require_default_send_mode_flag() {
  const pid_t pid = ::fork();
  require(pid >= 0, "fork failed for MetadataSender default send-mode probe");
  if (pid == 0) {
    ::_exit(probe_default_send_mode_in_child());
  }

  int status = 0;
  require(::waitpid(pid, &status, 0) == pid,
          "waitpid failed for MetadataSender default send-mode probe");
  require(WIFEXITED(status), "MetadataSender default send-mode probe terminated abnormally");
  require(WEXITSTATUS(status) == 0,
          "MetadataSender default send-mode probe failed with child status " +
              std::to_string(WEXITSTATUS(status)));
}

void require_send_mode_flag(bool nonblocking) {
  const pid_t pid = ::fork();
  require(pid >= 0, "fork failed for MetadataSender send-mode probe");
  if (pid == 0) {
    ::_exit(probe_send_mode_in_child(nonblocking));
  }

  int status = 0;
  require(::waitpid(pid, &status, 0) == pid, "waitpid failed for MetadataSender send-mode probe");
  require(WIFEXITED(status), "MetadataSender send-mode probe terminated abnormally");
  require(WEXITSTATUS(status) == 0, "MetadataSender send-mode probe failed with child status " +
                                        std::to_string(WEXITSTATUS(status)));
}

int probe_enobufs_diagnostic_in_child(bool nonblocking) {
  simaai::neat::MetadataSenderOptions opt;
  opt.host = "127.0.0.1";
  opt.metadata_port_base = 9199;
  simaai::neat::MetadataSenderSendOptions send_opt;
  send_opt.nonblocking = nonblocking;

  std::string init_err;
  simaai::neat::MetadataSender sender(opt, send_opt, &init_err);
  if (!sender.ok() || !install_sendto_error_probe(ENOBUFS)) {
    return 20;
  }

  std::string send_err = "stale error";
  if (sender.send_raw_json("{}", &send_err)) {
    return 21;
  }
  const auto stats = sender.stats();
  if (stats.send_attempts != 1 || stats.send_failures != 1 || stats.datagrams_sent != 0 ||
      stats.last_errno != ENOBUFS || stats.no_buffer_space != 1 || stats.would_block != 0) {
    return 22;
  }
  if (send_err.empty() != nonblocking) {
    return 23;
  }
  return 0;
}

void require_enobufs_diagnostic(bool nonblocking) {
  const pid_t pid = ::fork();
  require(pid >= 0, "fork failed for MetadataSender ENOBUFS probe");
  if (pid == 0) {
    ::_exit(probe_enobufs_diagnostic_in_child(nonblocking));
  }

  int status = 0;
  require(::waitpid(pid, &status, 0) == pid, "waitpid failed for MetadataSender ENOBUFS probe");
  require(WIFEXITED(status), "MetadataSender ENOBUFS probe terminated abnormally");
  require(WEXITSTATUS(status) == 0, "MetadataSender ENOBUFS probe failed with child status " +
                                        std::to_string(WEXITSTATUS(status)));
}

} // namespace

RUN_TEST(
    "unit_node_metadata_sender_test", ([] {
      using nlohmann::json;

      sima_test::UdpReceiver rx;

      simaai::neat::MetadataSenderOptions opt;
      opt.host = "127.0.0.1";
      opt.channel = 0;
      opt.metadata_port_base = rx.port();
      simaai::neat::MetadataSenderSendOptions send_opt;
      require(send_opt.nonblocking, "MetadataSenderSendOptions must default to nonblocking sends");

      std::string init_err;
      simaai::neat::MetadataSender sender(opt, &init_err);
      require(sender.ok(), "MetadataSender initialization failed: " + init_err);
      require(sender.metadata_port() == rx.port(), "MetadataSender metadata_port mismatch");
      require(sender.nonblocking(), "MetadataSender nonblocking option mismatch");

      std::string send_err;
      require(sender.send_raw_json(R"({"type":"raw","data":{"ok":true}})", &send_err),
              "MetadataSender send_raw_json failed: " + send_err);

      std::string received;
      require(rx.recv_one(&received, 2000), "MetadataSender send_raw_json payload not received");
      require(json::parse(received)["type"].get<std::string>() == "raw",
              "MetadataSender send_raw_json payload mismatch");

      const std::string mtu_sized_payload(1200, 'r');
      require(sender.send_raw_json(mtu_sized_payload, &send_err),
              "MetadataSender 1200-byte send_raw_json failed: " + send_err);
      require(rx.recv_one(&received, 2000), "MetadataSender 1200-byte payload not received");
      require(received == mtu_sized_payload,
              "MetadataSender must preserve 1200-byte raw payloads byte-for-byte");

      const std::string chunked_payload(1201, 'x');
      require(sender.send_raw_json(chunked_payload, &send_err),
              "MetadataSender chunked send_raw_json failed: " + send_err);

      std::string first_chunk;
      std::string second_chunk;
      require(rx.recv_one(&first_chunk, 2000), "MetadataSender first chunk not received");
      require(rx.recv_one(&second_chunk, 2000), "MetadataSender second chunk not received");
      require(first_chunk.size() == 1200, "MetadataSender first chunk must fit 1200-byte MTU");
      require(second_chunk.size() == 25, "MetadataSender second chunk size mismatch");
      require(static_cast<unsigned char>(first_chunk[0]) == 0x4e &&
                  static_cast<unsigned char>(first_chunk[1]) == 0x01,
              "MetadataSender chunk header magic/version mismatch");
      require(first_chunk.substr(2, 8) == second_chunk.substr(2, 8),
              "MetadataSender chunks must share one message id");
      require(static_cast<unsigned char>(first_chunk[10]) == 0 &&
                  static_cast<unsigned char>(second_chunk[10]) == 1,
              "MetadataSender chunk indexes mismatch");
      require(static_cast<unsigned char>(first_chunk[11]) == 2 &&
                  static_cast<unsigned char>(second_chunk[11]) == 2,
              "MetadataSender chunk counts mismatch");
      require(first_chunk.substr(12) + second_chunk.substr(12) == chunked_payload,
              "MetadataSender chunks must reconstruct the original payload");

      const std::string max_payload(65507, 'm');
      require(sender.send_raw_json(max_payload, &send_err),
              "MetadataSender 65507-byte send_raw_json failed: " + send_err);
      std::string reconstructed;
      reconstructed.reserve(max_payload.size());
      std::string max_message_id;
      for (int expected_index = 0; expected_index < 56; ++expected_index) {
        std::string chunk;
        require(rx.recv_one(&chunk, 2000), "MetadataSender maximum-size chunk not received");
        require(chunk.size() <= 1200, "MetadataSender chunk exceeds 1200-byte UDP payload");
        require(static_cast<unsigned char>(chunk[0]) == 0x4e &&
                    static_cast<unsigned char>(chunk[1]) == 0x01,
                "MetadataSender maximum-size chunk header mismatch");
        if (max_message_id.empty()) {
          max_message_id = chunk.substr(2, 8);
        }
        require(chunk.substr(2, 8) == max_message_id,
                "MetadataSender maximum-size chunks must share one message id");
        require(static_cast<unsigned char>(chunk[10]) == expected_index,
                "MetadataSender maximum-size chunk index mismatch");
        require(static_cast<unsigned char>(chunk[11]) == 56,
                "MetadataSender maximum-size chunk count mismatch");
        reconstructed.append(chunk.data() + 12, chunk.size() - 12);
      }
      require(max_message_id != first_chunk.substr(2, 8),
              "MetadataSender chunked messages must have distinct message ids");
      require(reconstructed == max_payload,
              "MetadataSender maximum-size chunks must reconstruct the original payload");

      const std::string oversized_payload(65508, 'x');
      require(!sender.send_raw_json(oversized_payload, &send_err),
              "MetadataSender must reject payloads larger than 65507 bytes");
      require_contains(send_err, "65507", "MetadataSender oversized payload error mismatch");

      const std::string data_json = R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})";
      require(sender.send_metadata("tracking", data_json, 12345, "frame-7", &send_err),
              "MetadataSender send_metadata failed: " + send_err);

      require(rx.recv_one(&received, 2000), "MetadataSender send_metadata payload not received");
      const json parsed = json::parse(received);
      require(parsed["type"].get<std::string>() == "tracking",
              "MetadataSender send_metadata type mismatch");
      require(parsed["timestamp"].get<int64_t>() == 12345,
              "MetadataSender send_metadata timestamp mismatch");
      require(parsed["frame_id"].get<std::string>() == "frame-7",
              "MetadataSender send_metadata frame_id mismatch");
      require(parsed["data"]["tracks"][0]["id"].get<std::string>() == "trk-1",
              "MetadataSender send_metadata data mismatch");

      const auto send_stats = sender.stats();
      require(send_stats.send_attempts == 61, "MetadataSender send-attempt stats mismatch");
      require(send_stats.datagrams_sent == 61, "MetadataSender sent-datagram stats mismatch");
      require(send_stats.send_failures == 0, "MetadataSender failure stats mismatch");
      require(send_stats.would_block == 0, "MetadataSender would-block stats mismatch");
      require(send_stats.no_buffer_space == 0, "MetadataSender ENOBUFS stats mismatch");
      require(send_stats.last_errno == 0, "MetadataSender last errno should clear on success");
      require(send_stats.max_send_duration_ns >= send_stats.last_send_duration_ns,
              "MetadataSender duration stats mismatch");

      require(!sender.send_metadata("", data_json, 1, "frame", &send_err),
              "MetadataSender should reject empty metadata type");
      require_contains(send_err, "type must not be empty",
                       "MetadataSender empty type error mismatch");

      require(!sender.send_metadata("tracking", "{bad-json", 1, "frame", &send_err),
              "MetadataSender should reject invalid data_json");
      require_contains(send_err, "parse failed", "MetadataSender invalid data_json error mismatch");

      require(!sender.send_metadata("tracking", "[]", 1, "frame", &send_err),
              "MetadataSender should reject non-object data_json");
      require_contains(send_err, "must be a JSON object",
                       "MetadataSender non-object data_json error mismatch");

      simaai::neat::MetadataSenderOptions bad_opt;
      bad_opt.host = "256.256.256.256";
      bad_opt.channel = 0;
      bad_opt.metadata_port_base = 9300;

      std::string bad_init_err;
      simaai::neat::MetadataSender bad_sender(bad_opt, &bad_init_err);
      require(!bad_sender.ok(), "MetadataSender should fail with invalid host");
      require_contains(bad_init_err, "getaddrinfo", "MetadataSender bad-host error mismatch");

      std::string bad_send_err;
      require(!bad_sender.send_raw_json("{}", &bad_send_err),
              "MetadataSender should reject send_raw_json when uninitialized");
      require_contains(bad_send_err, "not initialized",
                       "MetadataSender uninitialized send error mismatch");
      const auto bad_stats = bad_sender.stats();
      require(bad_stats.send_attempts == 0,
              "uninitialized MetadataSender must not count a kernel send attempt");

      require_default_send_mode_flag();
      require_send_mode_flag(true);
      require_send_mode_flag(false);
      require_enobufs_diagnostic(false);
      require_enobufs_diagnostic(true);

      // Multiple dispatch threads may share one sender. Counter increments must
      // not be lost, while readers are allowed to observe a non-transactional
      // point-in-time combination of fields.
      constexpr int kThreads = 4;
      constexpr int kSendsPerThread = 100;
      std::atomic<bool> keep_reading{true};
      std::thread stats_reader([&] {
        while (keep_reading.load(std::memory_order_relaxed)) {
          (void)sender.stats();
          std::this_thread::yield();
        }
      });
      std::vector<std::thread> senders;
      senders.reserve(kThreads);
      for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
        senders.emplace_back([&] {
          for (int send_index = 0; send_index < kSendsPerThread; ++send_index) {
            (void)sender.send_raw_json("{}");
          }
        });
      }
      for (auto& dispatch_thread : senders) {
        dispatch_thread.join();
      }
      keep_reading.store(false, std::memory_order_relaxed);
      stats_reader.join();

      const auto concurrent_stats = sender.stats();
      require(concurrent_stats.send_attempts ==
                  2 + static_cast<uint64_t>(kThreads * kSendsPerThread),
              "MetadataSender concurrent send-attempt increments were lost");
      require(concurrent_stats.datagrams_sent + concurrent_stats.send_failures ==
                  concurrent_stats.send_attempts,
              "MetadataSender final concurrent counters must account for every attempt");
    }));
