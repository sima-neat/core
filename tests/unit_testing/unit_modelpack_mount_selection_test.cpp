// Automatic NVMe selection reads /proc/mounts, so its filtering is exercised here against a
// synthetic mount table rather than against whatever the test machine happens to have mounted.
#include "model/internal/ModelPack.h"

#include "test_main.h"
#include "test_utils.h"

#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> bases_for(const std::string& mounts_content) {
  std::istringstream mounts(mounts_content);
  return simaai::neat::internal::nvme_model_bases_from_mounts(mounts);
}

} // namespace

RUN_TEST("unit_modelpack_mount_selection_test", ([] {
           // A plain data mount is the case automatic selection exists for.
           require(bases_for("/dev/nvme0n1p1 /media/nvme ext4 rw,noatime 0 0\n") ==
                       std::vector<std::string>{"/media/nvme/simaai/coprocessing/models"},
                   "a writable NVMe data mount should be a candidate");

           // noexec cannot load a .so extracted into lib/, so the mount is ineligible rather than
           // merely full: selection must fall through to the next candidate.
           require(bases_for("/dev/nvme0n1p1 /media/nvme ext4 rw,noatime,noexec 0 0\n").empty(),
                   "a noexec NVMe mount should not be a candidate");
           require(bases_for("/dev/nvme0n1p1 /media/nvme ext4 noexec,rw 0 0\n").empty(),
                   "noexec should be rejected wherever it appears in the option list");

           // Read-only was already rejected, but only when listed first. Both orders must fail now.
           require(bases_for("/dev/nvme0n1p1 /media/nvme ext4 ro,noatime 0 0\n").empty(),
                   "a read-only NVMe mount should not be a candidate");
           require(bases_for("/dev/nvme0n1p1 /media/nvme ext4 noatime,ro 0 0\n").empty(),
                   "read-only should be rejected wherever it appears in the option list");

           // Exact tokens only: these contain "ro" and "noexec" as substrings and must still
           // qualify.
           require(bases_for("/dev/nvme0n1p1 /media/nvme ext4 rw,relatime,errors=remount-ro 0 0\n")
                           .size() == 1,
                   "an option merely containing 'ro' should not reject the mount");

           // Pre-existing filters must survive the option-parsing change.
           require(bases_for("/dev/nvme0n1p1 / ext4 rw,noatime 0 0\n").empty(),
                   "a root mount should not be a model store");
           require(bases_for("/dev/nvme0n1p1 /boot/efi vfat rw,noatime 0 0\n").empty(),
                   "a vfat EFI partition should not be a model store");
           require(bases_for("/dev/mmcblk0p4 /media/data ext4 rw,noatime 0 0\n").empty(),
                   "only NVMe block devices are auto-selected");
           require(bases_for("192.168.0.1:/export /media/nfs nfs rw,noatime 0 0\n").empty(),
                   "a network filesystem should never be auto-selected");

           // The first eligible mount wins, and an ineligible one ahead of it must not mask it.
           require(bases_for("/dev/nvme0n1p1 /media/a ext4 rw,noexec 0 0\n"
                             "/dev/nvme0n1p2 /media/b ext4 rw,noatime 0 0\n") ==
                       std::vector<std::string>{"/media/b/simaai/coprocessing/models"},
                   "an ineligible mount should not hide a later eligible one");
         }));
