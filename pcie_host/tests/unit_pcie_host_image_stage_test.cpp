#include "genai/ImageStage.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
using simaai::neat::pcie::genai::internal::StagedImage;

namespace {
void require(bool c, const std::string& m) {
  if (!c)
    throw std::runtime_error(m);
}

fs::path make_temp_dir() {
  fs::path d = fs::temp_directory_path() /
               ("image_stage_test_" + std::to_string(::getpid()) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(d);
  // 0755 whatever the umask: the stage folder's parent must not be writable
  // by group or others (StagedImage refuses it then).
  fs::permissions(d, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                         fs::perms::others_read | fs::perms::others_exec);
  return d;
}

void write_file(const fs::path& p, const std::string& text) {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << text;
}
} // namespace

int main() {
  try {
    const fs::path root = make_temp_dir();
    const fs::path stage = root / "pcie-genai";
    const fs::path src = root / "photo.jpg";
    write_file(src, "JPEGDATA");

    // copy made, extension kept, name relative to the serve root
    {
      StagedImage s(stage, "h9-1", src);
      require(s.relative_name() == "pcie-genai/h9-1.jpg", "relative name");
      require(fs::exists(s.staged_path()), "staged copy exists");
      require(fs::exists(stage / "h9-1.jpg"), "copy lands in the stage dir");
    }
    require(!fs::exists(stage / "h9-1.jpg"), "copy deleted on scope exit");

    // A symlink planted at the predictable staged name must not be followed:
    // its target keeps its content, and the staged file is a new regular file.
    {
      const fs::path victim = root / "victim.txt";
      write_file(victim, "KEEP");
      fs::create_directories(stage);
      fs::create_symlink(victim, stage / "h9-5.jpg");
      {
        StagedImage s(stage, "h9-5", src);
        require(!fs::is_symlink(s.staged_path()) && fs::is_regular_file(s.staged_path()),
                "the staged file is a regular file, not the planted symlink");
        std::ifstream staged(s.staged_path());
        const std::string body((std::istreambuf_iterator<char>(staged)),
                               std::istreambuf_iterator<char>());
        require(body == "JPEGDATA", "the staged file holds the image");
      }
      std::ifstream v(victim);
      const std::string kept((std::istreambuf_iterator<char>(v)), std::istreambuf_iterator<char>());
      require(kept == "KEEP", "the symlink target must not be overwritten");
      fs::remove(victim);
    }

    // The stage folder and its parent must be safe from other users: writable
    // by others only with the sticky bit, and never a symlink.
    {
      const auto refused = [&](const fs::path& dir) {
        try {
          StagedImage s(dir, "h9-6", src);
        } catch (const std::runtime_error&) {
          return true;
        }
        return false;
      };
      const fs::path open_parent = root / "open";
      fs::create_directories(open_parent);
      fs::permissions(open_parent, fs::perms::all); // 0777, no sticky bit
      require(refused(open_parent / "pcie-genai"), "a parent writable by others is refused");
      fs::permissions(open_parent, fs::perms::sticky_bit, fs::perm_options::add);
      require(!refused(open_parent / "pcie-genai"), "a sticky parent (like /tmp) is accepted");
      require((fs::status(open_parent / "pcie-genai").permissions() & fs::perms::others_write) ==
                  fs::perms::none,
              "a new stage folder is not writable by others");
      fs::permissions(open_parent / "pcie-genai", fs::perms::group_write, fs::perm_options::add);
      require(refused(open_parent / "pcie-genai"), "a group-writable stage folder is refused");
      const fs::path linked = root / "linked";
      fs::create_directories(root / "elsewhere");
      fs::create_directories(linked);
      fs::create_directory_symlink(root / "elsewhere", linked / "pcie-genai");
      require(refused(linked / "pcie-genai"), "a symlinked stage folder is refused");
    }

    // missing source throws
    {
      bool threw = false;
      try {
        StagedImage s(stage, "h9-2", root / "does-not-exist.png");
      } catch (const std::runtime_error&) {
        threw = true;
      }
      require(threw, "missing source throws");
    }

    // 1-hour sweep: an old file goes, a fresh one stays
    {
      write_file(stage / "old.jpg", "x");
      const auto two_hours_ago = fs::file_time_type::clock::now() - std::chrono::hours(2);
      fs::last_write_time(stage / "old.jpg", two_hours_ago);
      write_file(stage / "young.jpg", "y");
      StagedImage s(stage, "h9-3", src);
      require(!fs::exists(stage / "old.jpg"), "old file swept");
      require(fs::exists(stage / "young.jpg"), "young file kept");
    }

    // The sweep keeps an old file whose owner process is still running: another
    // session sharing this folder may still be using it (a long answer). A file of a
    // process that is gone is swept as before.
    {
      const auto two_hours_ago = fs::file_time_type::clock::now() - std::chrono::hours(2);
      const std::string live = "h" + std::to_string(::getpid()) + "-7-0.jpg";
      const std::string dead = "h999999999-7-0.jpg"; // above any Linux pid_max
      write_file(stage / live, "x");
      write_file(stage / dead, "x");
      fs::last_write_time(stage / live, two_hours_ago);
      fs::last_write_time(stage / dead, two_hours_ago);
      StagedImage s(stage, "h9-4", src);
      require(fs::exists(stage / live), "an old file of a live process is kept");
      require(!fs::exists(stage / dead), "an old file of a dead process is swept");
    }

    fs::remove_all(root);
    std::cout << "[PASS] pcie-genai image stage\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
