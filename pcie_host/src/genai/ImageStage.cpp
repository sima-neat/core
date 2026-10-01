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
#include <sys/stat.h>
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

// The card reads a staged file by its path, later. If another user can change
// the stage folder or its parent (unlink our file, rename the folder away,
// put a symlink in its place), the card could read their file instead. So
// both folders must be real directories (not symlinks), owned by us or root,
// and either not writable by group/others or sticky (+t, like /tmp: then
// others cannot delete or rename our files). Throws with the chmod to run.
void require_protected_dir(const fs::path& dir) {
  struct stat st {};
  if (::lstat(dir.c_str(), &st) != 0) {
    throw std::runtime_error("cannot check image stage folder " + dir.string() + ": " +
                             std::strerror(errno));
  }
  if (!S_ISDIR(st.st_mode)) {
    throw std::runtime_error("image stage folder " + dir.string() +
                             " is not a real directory (a symlink or a file); refusing images");
  }
  if (st.st_uid != ::getuid() && st.st_uid != 0) {
    throw std::runtime_error("image stage folder " + dir.string() +
                             " is owned by another user; refusing images");
  }
  if ((st.st_mode & (S_IWGRP | S_IWOTH)) != 0 && (st.st_mode & S_ISVTX) == 0) {
    throw std::runtime_error(
        "image stage folder " + dir.string() +
        " can be changed by other users, so a staged image could be swapped before the card "
        "reads it; refusing images. Fix it with: chmod +t " +
        dir.string() + "  (or: chmod go-w " + dir.string() + ")");
  }
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
  require_protected_dir(stage_dir.parent_path());
  // New folder: owner-only writable. An existing one is checked as it is.
  if (::mkdir(stage_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    throw std::runtime_error("cannot create image stage dir " + stage_dir.string() + ": " +
                             std::strerror(errno));
  }
  require_protected_dir(stage_dir);
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
