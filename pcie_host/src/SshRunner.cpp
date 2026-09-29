#include "SshRunner.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace simaai::neat::pcie::internal {

std::string SshRunner::shell_escape(const std::string& value) {
  std::string out = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out += "'";
  return out;
}

CommandResult SshRunner::run(const std::vector<std::string>& args, const int timeout_sec) {
  const int effective_timeout = timeout_sec > 0 ? timeout_sec : 1;
  return run_for(args, std::chrono::seconds(effective_timeout));
}

CommandResult SshRunner::run_for(const std::vector<std::string>& args,
                                 const std::chrono::milliseconds timeout) {
  if (args.empty()) {
    throw std::invalid_argument("SshRunner::run requires a command");
  }

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  int pipefd[2] = {-1, -1};
  if (::pipe(pipefd) != 0) {
    throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
  }

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
  }

  if (pid == 0) {
    // Own process group: a terminal Ctrl-C goes to the caller only (it handles
    // it itself), not to an in-flight ssh.
    ::setpgid(0, 0);
    // Never inherit the caller's stdin: ssh would read (and eat) piped input.
    // Example: `printf 'q1\nq2\n' | pcie-genai ...` lost its prompts to an
    // ssh call made before the first prompt. With /dev/null, ssh sees EOF
    // at once and the prompts stay for the caller.
    const int devnull = ::open("/dev/null", O_RDONLY);
    if (devnull >= 0) {
      ::dup2(devnull, STDIN_FILENO);
      if (devnull != STDIN_FILENO) {
        ::close(devnull);
      }
    }
    ::dup2(pipefd[1], STDOUT_FILENO);
    ::dup2(pipefd[1], STDERR_FILENO);
    ::close(pipefd[0]);
    ::close(pipefd[1]);

    ::execvp(argv[0], argv.data());
    _exit(127);
  }

  ::close(pipefd[1]);
  // Also set it here (no race with the child's own setpgid) so a timeout can
  // kill the whole group, including any grandchild still holding the pipe.
  ::setpgid(pid, pid);

  CommandResult result;
  const auto effective_timeout = std::max(timeout, std::chrono::milliseconds(1));
  const auto deadline = std::chrono::steady_clock::now() + effective_timeout;
  std::array<char, 4096> buffer{};
  bool child_done = false;

  while (!child_done) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      result.timed_out = true;
      // Negative pid = the whole process group: ssh and anything it started.
      // Killing only ssh could leave a child that keeps the pipe open.
      ::kill(-pid, SIGTERM);
      usleep(200000);
      ::kill(-pid, SIGKILL);
      break;
    }

    const auto ms_left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    pollfd pfd{pipefd[0], POLLIN, 0};
    const int pr = ::poll(&pfd, 1, static_cast<int>(std::min<long long>(ms_left, 200)));
    if (pr > 0 && (pfd.revents & POLLIN)) {
      const ssize_t n = ::read(pipefd[0], buffer.data(), buffer.size());
      if (n > 0) {
        result.output.append(buffer.data(), static_cast<std::size_t>(n));
      }
    }

    int status = 0;
    const pid_t wr = ::waitpid(pid, &status, WNOHANG);
    if (wr == pid) {
      child_done = true;
      if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
      } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
      }
    }
  }

  for (;;) {
    const ssize_t n = ::read(pipefd[0], buffer.data(), buffer.size());
    if (n > 0) {
      result.output.append(buffer.data(), static_cast<std::size_t>(n));
      continue;
    }
    break;
  }
  ::close(pipefd[0]);

  if (!child_done) {
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (result.exit_code < 0) {
      result.exit_code = result.timed_out ? 124 : 1;
    }
  }

  return result;
}

} // namespace simaai::neat::pcie::internal
