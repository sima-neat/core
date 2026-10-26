#include "nodes/groups/internal/UvcCamera.h"

#include "nodes/sima/SimaDecode.h"

#include <linux/videodev2.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace simaai::neat::internal::uvc {
namespace {

std::string read_line(const fs::path& path) {
  std::ifstream file(path);
  std::string line;
  std::getline(file, line);
  return line;
}

// libcamera's sysfs::firmwareNodePath(): the device-tree node below
// /sys/firmware/devicetree, else an ACPI firmware node path.
std::string firmware_node(const fs::path& dir, const fs::path& root) {
  std::error_code ec;
  if (fs::exists(dir / "of_node", ec)) {
    const std::string node = fs::canonical(dir / "of_node", ec).string();
    const std::string prefix = (root / "firmware" / "devicetree").string();
    if (ec)
      return {};
    return node.rfind(prefix, 0) == 0 ? node.substr(prefix.size()) : node;
  }
  if (fs::exists(dir / "firmware_node" / "path", ec))
    return read_line(dir / "firmware_node" / "path");
  return {};
}

int xioctl(int fd, unsigned long request, void* arg) {
  int result;
  do {
    result = ::ioctl(fd, request, arg);
  } while (result == -1 && errno == EINTR);
  return result;
}

std::string errno_text(int error) {
  return std::strerror(error);
}

struct Fd {
  int fd = -1;
  explicit Fd(int value) : fd(value) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() {
    if (fd >= 0)
      ::close(fd);
  }
};

bool fraction_equal(std::uint32_t a_num, std::uint32_t a_den, std::uint32_t b_num,
                    std::uint32_t b_den) {
  return std::uint64_t(a_num) * b_den == std::uint64_t(b_num) * a_den;
}

// True when the camera lists `width`x`height` at num/den fps for MJPEG; fills
// `modes` with what it does list, for the error message.
bool mjpeg_mode_listed(int fd, std::uint32_t width, std::uint32_t height, std::uint32_t num,
                       std::uint32_t den, std::vector<std::string>* modes) {
  bool found = false;
  v4l2_frmsizeenum size{};
  size.pixel_format = V4L2_PIX_FMT_MJPEG;
  for (; xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0; ++size.index) {
    std::uint32_t w = 0, h = 0;
    bool size_ok = false;
    if (size.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
      w = size.discrete.width;
      h = size.discrete.height;
      size_ok = w == width && h == height;
    } else {
      const auto& s = size.stepwise;
      w = width;
      h = height;
      size_ok = width >= s.min_width && width <= s.max_width && height >= s.min_height &&
                height <= s.max_height && (width - s.min_width) % std::max(1U, s.step_width) == 0 &&
                (height - s.min_height) % std::max(1U, s.step_height) == 0;
    }
    std::ostringstream listed;
    listed << w << 'x' << h << '@';
    bool first = true;
    v4l2_frmivalenum ival{};
    ival.pixel_format = V4L2_PIX_FMT_MJPEG;
    ival.width = w;
    ival.height = h;
    for (; xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &ival) == 0; ++ival.index) {
      // A frame interval is the reciprocal of the frame rate.
      if (ival.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
        const auto& iv = ival.discrete;
        listed << (first ? "" : ",") << iv.denominator << '/' << iv.numerator;
        found = found || (size_ok && fraction_equal(num, den, iv.denominator, iv.numerator));
      } else {
        const auto& s = ival.stepwise;
        listed << (first ? "" : ",") << s.max.denominator << '/' << s.max.numerator << "-"
               << s.min.denominator << '/' << s.min.numerator;
        // Interval within [min, max]: compare den/num against each bound.
        found = found ||
                (size_ok &&
                 std::uint64_t(den) * s.min.denominator >= std::uint64_t(num) * s.min.numerator &&
                 std::uint64_t(den) * s.max.denominator <= std::uint64_t(num) * s.max.numerator);
      }
      first = false;
      if (ival.type != V4L2_FRMIVAL_TYPE_DISCRETE)
        break;
    }
    if (modes && size.type == V4L2_FRMSIZE_TYPE_DISCRETE)
      modes->push_back(listed.str());
    if (size.type != V4L2_FRMSIZE_TYPE_DISCRETE)
      break;
  }
  return found;
}

bool captures_mjpeg(int fd) {
  v4l2_capability cap{};
  if (xioctl(fd, VIDIOC_QUERYCAP, &cap) != 0)
    return false;
  const std::uint32_t caps =
      (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
  if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING))
    return false; // e.g. the interface's metadata node
  v4l2_fmtdesc fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  for (; xioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0; ++fmt.index) {
    if (fmt.pixelformat == V4L2_PIX_FMT_MJPEG)
      return true;
  }
  return false;
}

// Capture until one frame parses, and return its decoded chroma layout. The
// mode must already be known to be listed; the driver may still adjust it.
FormatTag read_sampling(int fd, const std::string& id, std::uint32_t width, std::uint32_t height,
                        std::uint32_t num, std::uint32_t den) {
  const auto fail = [&id](const std::string& what) {
    return std::runtime_error("UsbCameraInput: USB camera '" + id + "' " + what);
  };
  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = width;
  fmt.fmt.pix.height = height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;
  if (xioctl(fd, VIDIOC_S_FMT, &fmt) != 0)
    throw fail(errno == EBUSY ? "is in use by another process"
                              : "rejected MJPEG format: " + errno_text(errno));
  if (fmt.fmt.pix.width != width || fmt.fmt.pix.height != height ||
      fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
    std::ostringstream m;
    m << "adjusted MJPEG " << width << 'x' << height << " to " << fmt.fmt.pix.width << 'x'
      << fmt.fmt.pix.height;
    throw fail(m.str());
  }
  v4l2_streamparm parm{};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe = {den, num};
  if (xioctl(fd, VIDIOC_S_PARM, &parm) != 0 ||
      !fraction_equal(parm.parm.capture.timeperframe.denominator,
                      parm.parm.capture.timeperframe.numerator, num, den)) {
    std::ostringstream m;
    m << "did not accept " << num << '/' << den << " fps (got "
      << parm.parm.capture.timeperframe.denominator << '/'
      << parm.parm.capture.timeperframe.numerator << ')';
    throw fail(m.str());
  }

  v4l2_requestbuffers req{};
  req.count = 2;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd, VIDIOC_REQBUFS, &req) != 0 || req.count == 0)
    throw fail("could not allocate capture buffers: " + errno_text(errno));
  std::vector<std::pair<void*, std::size_t>> maps;
  const auto release = [&]() {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    (void)xioctl(fd, VIDIOC_STREAMOFF, &type);
    for (auto& [ptr, len] : maps)
      ::munmap(ptr, len);
    v4l2_requestbuffers none{};
    none.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    none.memory = V4L2_MEMORY_MMAP;
    (void)xioctl(fd, VIDIOC_REQBUFS, &none);
  };
  try {
    for (std::uint32_t i = 0; i < req.count; ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (xioctl(fd, VIDIOC_QUERYBUF, &buf) != 0)
        throw fail("could not query capture buffer: " + errno_text(errno));
      void* ptr = ::mmap(nullptr, buf.length, PROT_READ, MAP_SHARED, fd, buf.m.offset);
      if (ptr == MAP_FAILED)
        throw fail("could not map capture buffer: " + errno_text(errno));
      maps.emplace_back(ptr, buf.length);
      if (xioctl(fd, VIDIOC_QBUF, &buf) != 0)
        throw fail("could not queue capture buffer: " + errno_text(errno));
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) != 0)
      throw fail(errno == EBUSY ? "is in use by another process"
                                : "could not start streaming: " + errno_text(errno));
    // A live camera can send the odd corrupt frame; try a few before giving up.
    std::string last_error = "sent no frames";
    for (int attempt = 0; attempt < 15; ++attempt) {
      pollfd p{fd, POLLIN, 0};
      if (::poll(&p, 1, 2000) <= 0)
        throw fail("sent no frame within 2 s");
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      if (xioctl(fd, VIDIOC_DQBUF, &buf) != 0)
        throw fail("capture failed: " + errno_text(errno));
      if (!(buf.flags & V4L2_BUF_FLAG_ERROR) && buf.bytesused > 0 && buf.index < maps.size()) {
        try {
          const auto* data = static_cast<const std::uint8_t*>(maps[buf.index].first);
          const FormatTag sampling = jpeg_output_format(std::span<const std::uint8_t>(
              data, std::min<std::size_t>(buf.bytesused, maps[buf.index].second)));
          release();
          return sampling;
        } catch (const std::invalid_argument& e) {
          last_error = e.what();
        }
      }
      (void)xioctl(fd, VIDIOC_QBUF, &buf);
    }
    throw fail("sent no usable MJPEG frame (" + last_error + ")");
  } catch (...) {
    release();
    throw;
  }
}

} // namespace

std::string camera_id(const std::string& interface_dir, const std::string& sysfs_root) {
  const fs::path root(sysfs_root);
  const fs::path iface(interface_dir);
  // Controller ID: the first ancestor described in firmware.
  std::string controller;
  for (fs::path dir = iface.parent_path(); dir.has_relative_path() && dir != root;
       dir = dir.parent_path()) {
    controller = firmware_node(dir, root);
    if (!controller.empty())
      break;
  }
  // USB ID: "<bus>-<ports>:<config>.<interface>" without the unstable bus number.
  const std::string name = iface.filename().string();
  const auto dash = name.find('-');
  const std::string vendor = read_line(iface.parent_path() / "idVendor");
  const std::string product = read_line(iface.parent_path() / "idProduct");
  if (controller.empty() || dash == std::string::npos || vendor.empty() || product.empty())
    return {};
  return controller + "-" + name.substr(dash + 1) + "-" + vendor + ":" + product;
}

std::vector<CaptureNode> list_nodes(const std::string& sysfs_root) {
  std::vector<CaptureNode> nodes;
  std::error_code ec;
  const fs::path dir = fs::path(sysfs_root) / "class" / "video4linux";
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    const fs::path driver = fs::canonical(entry.path() / "device" / "driver", ec);
    if (ec || driver.filename() != "uvcvideo") {
      ec.clear();
      continue;
    }
    const fs::path iface = fs::canonical(entry.path() / "device", ec);
    const std::string id = ec ? std::string() : camera_id(iface.string(), sysfs_root);
    ec.clear();
    if (!id.empty())
      nodes.push_back({id, "/dev/" + entry.path().filename().string()});
  }
  // /dev/video2 before /dev/video10.
  std::sort(nodes.begin(), nodes.end(), [](const CaptureNode& a, const CaptureNode& b) {
    return a.device.size() != b.device.size() ? a.device.size() < b.device.size()
                                              : a.device < b.device;
  });
  return nodes;
}

ResolvedCamera resolve_mjpeg_camera(const std::string& id, std::uint32_t width,
                                    std::uint32_t height, std::uint32_t framerate_num,
                                    std::uint32_t framerate_den, const std::string& sysfs_root) {
  const auto nodes = list_nodes(sysfs_root);
  bool matched = false;
  for (const auto& node : nodes) {
    if (node.id != id)
      continue;
    matched = true;
    Fd fd(::open(node.device.c_str(), O_RDWR | O_CLOEXEC));
    if (fd.fd < 0)
      throw std::runtime_error("UsbCameraInput: cannot open USB camera '" + id + "' (" +
                               node.device + "): " + errno_text(errno));
    if (!captures_mjpeg(fd.fd))
      continue;
    std::vector<std::string> modes;
    if (!mjpeg_mode_listed(fd.fd, width, height, framerate_num, framerate_den, &modes)) {
      std::ostringstream m;
      m << "UsbCameraInput: USB camera '" << id << "' has no MJPEG mode " << width << 'x' << height
        << " at " << framerate_num << '/' << framerate_den << " fps; its MJPEG modes are:";
      for (const auto& mode : modes)
        m << ' ' << mode;
      throw std::runtime_error(m.str());
    }
    return {node.device, read_sampling(fd.fd, id, width, height, framerate_num, framerate_den)};
  }
  if (matched)
    throw std::runtime_error("UsbCameraInput: USB camera '" + id + "' has no MJPEG capture node");
  std::ostringstream m;
  m << "UsbCameraInput: no USB camera with ID '" << id << "'";
  std::vector<std::string> ids;
  for (const auto& node : nodes)
    if (std::find(ids.begin(), ids.end(), node.id) == ids.end())
      ids.push_back(node.id);
  if (ids.empty())
    m << "; no USB cameras are connected";
  for (std::size_t i = 0; i < ids.size(); ++i)
    m << (i ? ", " : "; connected USB camera IDs: ") << ids[i];
  throw std::runtime_error(m.str());
}

} // namespace simaai::neat::internal::uvc
