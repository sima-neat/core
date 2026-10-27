#include "model/internal/ModelPackProcessRoot.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <system_error>
#include <sys/file.h>
#include <unistd.h>

namespace simaai::neat::internal {
namespace {
namespace fs = std::filesystem;
constexpr const char* kOwnerLock = ".sima_modelpack_owner.lock";
constexpr const char* kKeepMarker = ".sima_modelpack_keep";
constexpr const char* kRootPrefix = "proc_run_";

class ScopedFd {
public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() {
    if (fd_ >= 0)
      ::close(fd_);
  }
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  int get() const {
    return fd_;
  }
  int release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }

private:
  int fd_;
};

[[noreturn]] void storage_error(const fs::path& path) {
  throw std::system_error(errno, std::generic_category(),
                          "ModelPack: output_storage_unavailable: " + path.string());
}

// The owner file is published only after locking it, so an initializing root cannot look stale.
// Legacy proc_<pid> directories have no ownership proof and remain untouched.
void cleanup_stale_roots(const fs::path& base) {
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(base, ec)) {
    if (ec)
      break;
    if (entry.path().filename().string().rfind(kRootPrefix, 0) != 0)
      continue;
    const ScopedFd owner(
        ::open((entry.path() / kOwnerLock).c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW));
    if (owner.get() < 0 || ::flock(owner.get(), LOCK_EX | LOCK_NB) != 0)
      continue;
    if (!fs::exists(entry.path() / kKeepMarker, ec) && !ec)
      fs::remove_all(entry.path(), ec);
    ec.clear();
  }
}
} // namespace

ModelPackProcessRoot::ModelPackProcessRoot(const fs::path& base, bool collect_stale) {
  fs::create_directories(base);
  if (collect_stale)
    cleanup_stale_roots(base);

  std::string pattern = (base / "proc_run_XXXXXX").string();
  if (!::mkdtemp(pattern.data()))
    storage_error(base);
  path_ = pattern;
  const fs::path pending_lock = path_ / ".sima_modelpack_owner.pending";
  ScopedFd owner(
      ::open(pending_lock.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600));
  if (owner.get() < 0 || ::flock(owner.get(), LOCK_EX | LOCK_NB) != 0 ||
      ::rename(pending_lock.c_str(), (path_ / kOwnerLock).c_str()) != 0) {
    const int error = errno;
    std::error_code ec;
    fs::remove_all(path_, ec);
    errno = error;
    storage_error(path_);
  }
  lock_fd_ = owner.release();
}

ModelPackProcessRoot::~ModelPackProcessRoot() {
  if (!preserved_) {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  ::close(lock_fd_);
}

void ModelPackProcessRoot::preserve() {
  if (preserved_)
    return;
  preserved_ = true;
  std::ofstream(path_ / kKeepMarker) << "keep\n";
}

} // namespace simaai::neat::internal
