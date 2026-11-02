#include "nodes/io/internal/UvcCamera.h"
#include "nodes/io/internal/UvcMode.h"
#include <gst/video/video.h>

#include "nodes/sima/internal/JpegOutputFormat.h"

#include <linux/videodev2.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
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

// The node's sysfs index within its USB interface (capture 0, metadata 1); -1 when unreadable.
int node_index(const std::string& sysfs_root, const std::string& device) {
  const std::string text = read_line(fs::path(sysfs_root) / "class" / "video4linux" /
                                     fs::path(device).filename() / "index");
  int index = -1;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), index);
  return ec == std::errc() && end == text.data() + text.size() ? index : -1;
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
bool mode_listed(int fd, std::uint32_t fourcc, std::uint32_t width, std::uint32_t height,
                 std::uint32_t num, std::uint32_t den, std::vector<std::string>* modes) {
  bool found = false;
  v4l2_frmsizeenum size{};
  size.pixel_format = fourcc;
  for (; xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0; ++size.index) {
    const bool discrete = size.type == V4L2_FRMSIZE_TYPE_DISCRETE;
    std::uint32_t w = 0, h = 0;
    bool size_ok = false;
    std::ostringstream listed;
    if (discrete) {
      w = size.discrete.width;
      h = size.discrete.height;
      size_ok = w == width && h == height;
      listed << w << 'x' << h << '@';
    } else {
      const auto& s = size.stepwise;
      w = width;
      h = height;
      size_ok = width >= s.min_width && width <= s.max_width && height >= s.min_height &&
                height <= s.max_height && (width - s.min_width) % std::max(1U, s.step_width) == 0 &&
                (height - s.min_height) % std::max(1U, s.step_height) == 0;
      // One range covers every size (continuous ranges step 1x1); rates only for the request.
      listed << s.min_width << 'x' << s.min_height << '-' << s.max_width << 'x' << s.max_height
             << " step " << s.step_width << 'x' << s.step_height;
      if (size_ok)
        listed << " (" << w << 'x' << h << '@';
    }
    bool first = true;
    v4l2_frmivalenum ival{};
    ival.pixel_format = fourcc;
    ival.width = w;
    ival.height = h;
    for (; (discrete || size_ok) && xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &ival) == 0;
         ++ival.index) {
      // A frame interval is the reciprocal of the frame rate.
      if (ival.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
        const auto& iv = ival.discrete;
        listed << (first ? "" : ",") << iv.denominator << '/' << iv.numerator;
        found = found || (size_ok && interval_contains(ival, num, den));
      } else {
        const auto& s = ival.stepwise;
        listed << (first ? "" : ",") << s.max.denominator << '/' << s.max.numerator << "-"
               << s.min.denominator << '/' << s.min.numerator;
        found = found || (size_ok && interval_contains(ival, num, den));
      }
      first = false;
      if (ival.type != V4L2_FRMIVAL_TYPE_DISCRETE)
        break;
    }
    if (!discrete && size_ok)
      listed << ')';
    if (modes)
      modes->push_back(listed.str());
    if (!discrete)
      break;
  }
  return found;
}

bool captures_format(int fd, std::uint32_t fourcc) {
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
    if (fmt.pixelformat == fourcc)
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
      // One 2 s budget per attempt; a signal resumes the wait with the time left.
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      pollfd p{fd, POLLIN, 0};
      int ready;
      do {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        ready = ::poll(&p, 1, static_cast<int>(std::max<std::int64_t>(0, left.count())));
      } while (ready == -1 && errno == EINTR);
      if (ready <= 0)
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

std::vector<std::string> camera_ids(const std::vector<CaptureNode>& nodes) {
  std::vector<std::string> ids;
  for (const auto& node : nodes)
    ids.push_back(node.id);
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

namespace {

// True when `camera.device` still belongs to `camera.id` and is still the node at `camera.index`.
bool still_connected(const ResolvedCamera& camera) {
  std::error_code ec;
  const fs::path node = fs::path(camera.sysfs_root) / "class" / "video4linux" /
                        fs::path(camera.device).filename() / "device";
  const fs::path iface = fs::canonical(node, ec);
  // An unknown index (no sysfs file) does not decide; a changed one does.
  const int index = node_index(camera.sysfs_root, camera.device);
  return !ec && camera_id(iface.string(), camera.sysfs_root) == camera.id &&
         (camera.index < 0 || index < 0 || index == camera.index);
}

} // namespace

void require_still_connected(const ResolvedCamera& camera) {
  if (!camera.id.empty() && !still_connected(camera))
    throw std::runtime_error("UsbCameraInput: USB camera '" + camera.id + "' is no longer " +
                             camera.device +
                             "; it was unplugged, moved or renumbered. Call UsbCameraInput again "
                             "to open it.");
}

ResolvedCamera camera_from_info(const std::string& id, const UsbCameraInfo& info,
                                const std::string& sysfs_root) {
  ResolvedCamera camera{info.device, info.decoded_format, id, sysfs_root, info.node_index};
  if (!still_connected(camera))
    throw std::runtime_error("UsbCameraInput: USB camera '" + id + "' is not at '" + info.device +
                             "' as its UsbCameraInfo says; it was unplugged, moved or renumbered "
                             "since, or the info is for another camera. Call usb_camera_info "
                             "again and pass the new info.");
  return camera;
}

ResolvedCamera resolve_mjpeg_camera(const std::string& id, std::uint32_t width,
                                    std::uint32_t height, std::uint32_t framerate_num,
                                    std::uint32_t framerate_den, const std::string& sysfs_root) {
  const auto nodes = list_nodes(sysfs_root);
  bool matched = false;
  bool mjpeg_node = false;
  std::vector<std::string> modes;
  for (const auto& node : nodes) {
    if (node.id != id)
      continue;
    matched = true;
    Fd fd(::open(node.device.c_str(), O_RDWR | O_CLOEXEC));
    if (fd.fd < 0)
      throw std::runtime_error("UsbCameraInput: cannot open USB camera '" + id + "' (" +
                               node.device + "): " + errno_text(errno));
    if (!captures_format(fd.fd, V4L2_PIX_FMT_MJPEG))
      continue;
    mjpeg_node = true;
    // A camera can expose several MJPEG nodes; use the first that lists the mode.
    if (!mode_listed(fd.fd, V4L2_PIX_FMT_MJPEG, width, height, framerate_num, framerate_den,
                     &modes))
      continue;
    return {node.device, read_sampling(fd.fd, id, width, height, framerate_num, framerate_den), id,
            sysfs_root, node_index(sysfs_root, node.device)};
  }
  if (mjpeg_node) {
    std::ostringstream m;
    m << "UsbCameraInput: USB camera '" << id << "' has no MJPEG mode " << width << 'x' << height
      << " at " << framerate_num << '/' << framerate_den << " fps; its MJPEG modes are:";
    for (const auto& mode : modes)
      m << ' ' << mode;
    throw std::runtime_error(m.str());
  }
  if (matched)
    throw std::runtime_error("UsbCameraInput: USB camera '" + id + "' has no MJPEG capture node");
  std::ostringstream m;
  m << "UsbCameraInput: no USB camera with ID '" << id << "'";
  const auto ids = camera_ids(nodes);
  if (ids.empty())
    m << "; no USB cameras are connected";
  for (std::size_t i = 0; i < ids.size(); ++i)
    m << (i ? ", " : "; connected USB camera IDs: ") << ids[i];
  throw std::runtime_error(m.str());
}

ResolvedCamera resolve_camera(const UsbCameraInputOptions& opt, UsbCameraFormat format,
                              const std::string& sysfs_root) {
  if (format == UsbCameraFormat::MJPEG)
    return resolve_mjpeg_camera(opt.camera_name, opt.width, opt.height, opt.framerate_num,
                                opt.framerate_den, sysfs_root);
  if (format != UsbCameraFormat::YUYV || !opt.width || (opt.width & 1U) || !opt.height ||
      !opt.framerate_num || !opt.framerate_den)
    throw std::invalid_argument("UsbCameraInput: invalid YUYV mode");
  const auto fail = [&](const std::string& reason) {
    return std::runtime_error("UsbCameraInput: USB camera '" + opt.camera_name + "' " + reason);
  };
  bool matched = false, capture = false;
  std::vector<std::string> modes;
  const auto nodes = list_nodes(sysfs_root);
  for (const auto& node : nodes) {
    if (node.id != opt.camera_name)
      continue;
    matched = true;
    Fd fd(::open(node.device.c_str(), O_RDWR | O_CLOEXEC));
    if (fd.fd < 0)
      throw fail("cannot open " + node.device + ": " + errno_text(errno));
    if (!captures_format(fd.fd, V4L2_PIX_FMT_YUYV))
      continue;
    capture = true;
    if (!mode_listed(fd.fd, V4L2_PIX_FMT_YUYV, opt.width, opt.height, opt.framerate_num,
                     opt.framerate_den, &modes))
      continue;
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = opt.width;
    fmt.fmt.pix.height = opt.height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(fd.fd, VIDIOC_S_FMT, &fmt) != 0)
      throw fail("cannot configure YUYV: " + errno_text(errno));
    const auto& pix = fmt.fmt.pix;
    if (pix.width != opt.width || pix.height != opt.height ||
        pix.pixelformat != V4L2_PIX_FMT_YUYV || pix.field != V4L2_FIELD_NONE)
      throw fail("adjusted the requested progressive YUYV mode; select an advertised mode");
    const std::uint64_t active = std::uint64_t(opt.width) * 2;
    if (pix.bytesperline < active ||
        pix.sizeimage < std::uint64_t(opt.height - 1) * pix.bytesperline + active)
      throw fail("returned an invalid YUYV stride or storage capacity");
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe = {opt.framerate_den, opt.framerate_num};
    if (xioctl(fd.fd, VIDIOC_S_PARM, &parm) != 0 || !parm.parm.capture.timeperframe.numerator ||
        !parm.parm.capture.timeperframe.denominator ||
        !fraction_equal(parm.parm.capture.timeperframe.denominator,
                        parm.parm.capture.timeperframe.numerator, opt.framerate_num,
                        opt.framerate_den))
      throw fail("did not accept " + std::to_string(opt.framerate_num) + "/" +
                 std::to_string(opt.framerate_den) + " fps");
    // V4L2 defaults describe matrix/range; never equate DEFAULT to full-range 601.
    const auto colorspace =
        pix.colorspace == V4L2_COLORSPACE_DEFAULT
            ? V4L2_MAP_COLORSPACE_DEFAULT(opt.width <= 720 && opt.height <= 576,
                                          opt.width <= 1920 && opt.height <= 1080)
            : pix.colorspace;
    const auto encoding = pix.ycbcr_enc == V4L2_YCBCR_ENC_DEFAULT
                              ? V4L2_MAP_YCBCR_ENC_DEFAULT(colorspace)
                              : pix.ycbcr_enc;
    const auto range = pix.quantization == V4L2_QUANTIZATION_DEFAULT
                           ? V4L2_MAP_QUANTIZATION_DEFAULT(false, colorspace, encoding)
                           : pix.quantization;
    const auto transfer = pix.xfer_func == V4L2_XFER_FUNC_DEFAULT
                              ? V4L2_MAP_XFER_FUNC_DEFAULT(colorspace)
                              : pix.xfer_func;
    if ((encoding != V4L2_YCBCR_ENC_601 && encoding != V4L2_YCBCR_ENC_709) ||
        (range != V4L2_QUANTIZATION_FULL_RANGE && range != V4L2_QUANTIZATION_LIM_RANGE) ||
        (transfer != V4L2_XFER_FUNC_709 && transfer != V4L2_XFER_FUNC_SRGB))
      throw fail("has unsupported YUYV matrix/range/transfer metadata");
    GstVideoColorimetry color{};
    color.matrix = encoding == V4L2_YCBCR_ENC_601 ? GST_VIDEO_COLOR_MATRIX_BT601
                                                  : GST_VIDEO_COLOR_MATRIX_BT709;
    color.range = range == V4L2_QUANTIZATION_FULL_RANGE ? GST_VIDEO_COLOR_RANGE_0_255
                                                        : GST_VIDEO_COLOR_RANGE_16_235;
    color.transfer =
        transfer == V4L2_XFER_FUNC_709 ? GST_VIDEO_TRANSFER_BT709 : GST_VIDEO_TRANSFER_SRGB;
    switch (colorspace) {
    case V4L2_COLORSPACE_SMPTE170M:
      color.primaries = GST_VIDEO_COLOR_PRIMARIES_SMPTE170M;
      break;
    case V4L2_COLORSPACE_REC709:
    case V4L2_COLORSPACE_SRGB:
    case V4L2_COLORSPACE_JPEG:
      color.primaries = GST_VIDEO_COLOR_PRIMARIES_BT709;
      break;
    default:
      throw fail("has unsupported YUYV color primaries");
    }
    ResolvedCamera camera{node.device, FormatTag::YUYV, opt.camera_name, sysfs_root,
                          node_index(sysfs_root, node.device)};
    camera.format = format;
    camera.row_stride = pix.bytesperline;
    camera.storage_bytes = pix.sizeimage;
    camera.color.matrix =
        encoding == V4L2_YCBCR_ENC_601 ? ImageColorMatrix::Bt601 : ImageColorMatrix::Bt709;
    camera.color.range =
        range == V4L2_QUANTIZATION_FULL_RANGE ? ImageYuvRange::Full : ImageYuvRange::Limited;
    gchar* text = gst_video_colorimetry_to_string(&color);
    if (!text)
      throw fail("cannot describe YUYV color metadata");
    camera.colorimetry = text;
    g_free(text);
    return camera;
  }
  if (capture) {
    std::ostringstream message;
    message << "has no YUYV mode " << opt.width << 'x' << opt.height << " at " << opt.framerate_num
            << '/' << opt.framerate_den << " fps; YUYV modes:";
    for (const auto& mode : modes)
      message << ' ' << mode;
    throw fail(message.str());
  }
  if (matched)
    throw fail("has no YUYV capture node");
  std::string available;
  for (const auto& id : camera_ids(nodes))
    available += " " + id;
  throw fail("is not connected; connected USB camera IDs:" + available);
}

} // namespace simaai::neat::internal::uvc
