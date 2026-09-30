#include "genai/ImageStage.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

#include <signal.h>

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
  staged_path_ = stage_dir / name;
  fs::copy_file(source, staged_path_, fs::copy_options::overwrite_existing, ec);
  if (ec) {
    throw std::runtime_error("cannot copy image " + source.string() + " to " +
                             staged_path_.string() + ": " + ec.message());
  }
  relative_name_ = stage_dir.filename().string() + "/" + name;
}

StagedImage::~StagedImage() {
  std::error_code ec;
  fs::remove(staged_path_, ec);
}

} // namespace simaai::neat::pcie::genai::internal
