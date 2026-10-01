#include "genai/ImageStage.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

namespace simaai::neat::pcie::genai::internal {

namespace fs = std::filesystem;

namespace {
constexpr auto kMaxLeftoverAge = std::chrono::hours(1);

// True if `name` is a staged image ("h<pid>-...") of a process that still runs.
// Another session sharing the stage dir may still be using such a file, even
// if it is old (a long answer), so the sweep must keep it.
bool owner_alive(const std::string& name) {
  if (name.size() < 2 || name[0] != 'h')
    return false;
  char* end = nullptr;
  errno = 0;
  const long pid = std::strtol(name.c_str() + 1, &end, 10);
  if (errno != 0 || end == name.c_str() + 1 || *end != '-' || pid <= 0)
    return false;
  // kill(pid, 0) sends nothing: 0 = it runs, EPERM = it runs as another user.
  return ::kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
}

// Delete regular files in `dir` older than one hour whose owner process is
// gone. Never throws: a leftover we cannot remove is not worth failing a new
// request over.
void sweep_old(const fs::path& dir) {
  std::error_code ec;
  const auto now = fs::file_time_type::clock::now();
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    if (!it->is_regular_file(ec))
      continue;
    const auto mtime = it->last_write_time(ec);
    if (ec) {
      ec.clear();
      continue;
    }
    if (now - mtime > kMaxLeftoverAge && !owner_alive(it->path().filename().string())) {
      std::error_code rm;
      fs::remove(it->path(), rm);
    }
  }
}
} // namespace

StagedImage::StagedImage(const fs::path& stage_dir, const std::string& run_id,
                         const fs::path& source) {
  std::error_code ec;
  fs::create_directories(stage_dir, ec);
  if (ec) {
    throw std::runtime_error("cannot create image stage dir " + stage_dir.string() + ": " +
                             ec.message());
  }
  sweep_old(stage_dir);

  const std::string name = run_id + source.extension().string();
  const fs::path target = stage_dir / name;
  std::ifstream in(source, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot read image " + source.string());
  }
  // The name is predictable (h<pid>-<n>-<i>), and the folder may be writable by
  // other users. copy_file(overwrite_existing) would follow a symlink planted at
  // the name and overwrite its target with our rights. So remove whatever is
  // there (remove() deletes a symlink itself, not its target) and create the
  // file new: O_EXCL fails if something reappears, O_NOFOLLOW never follows.
  fs::remove(target, ec);
  const int fd = ::open(target.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd < 0) {
    throw std::runtime_error("cannot create staged image " + target.string() + ": " +
                             std::strerror(errno));
  }
  char buffer[64 * 1024];
  bool ok = true;
  while (ok && in) {
    in.read(buffer, sizeof(buffer));
    const std::streamsize got = in.gcount();
    for (std::streamsize done = 0; ok && done < got;) {
      const ssize_t n = ::write(fd, buffer + done, static_cast<std::size_t>(got - done));
      ok = n > 0;
      done += n > 0 ? n : 0;
    }
  }
  ok = ok && in.eof();
  if (::close(fd) != 0 || !ok) {
    fs::remove(target, ec);
    throw std::runtime_error("cannot copy image " + source.string() + " to " + target.string());
  }
  staged_path_ = target;
  relative_name_ = stage_dir.filename().string() + "/" + name;
}

StagedImage::~StagedImage() {
  std::error_code ec;
  fs::remove(staged_path_, ec);
}

} // namespace simaai::neat::pcie::genai::internal
