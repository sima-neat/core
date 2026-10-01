#include "CameraProvider.h"

#include "CameraProviderInternal.h"
#include "gst/GstInit.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphReport.h"
#include "pipeline/NeatError.h"

#include <gst/gst.h>

#include <nlohmann/json.hpp>

#include <linux/videodev2.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

namespace simaai::neat::peripherals_internal {
namespace {

constexpr const char* kLibcameraProvider = "libcameraprovider";
constexpr const char* kIspSysfsName = "isp_v4l2-vid-cap-out";
constexpr const char* kIspCardName = "arm-isp-out";
constexpr const char* kSupportedFormat = "NV12";
constexpr std::uint32_t kDefaultFramerateNum = 30;
constexpr std::uint32_t kDefaultFramerateDen = 1;

auto mode_key(const CameraMode& mode) {
  const CameraSizeRange range = mode.size_range.value_or(CameraSizeRange{});
  return std::tuple(mode.format, mode.is_range(), mode.width, mode.height, range.min_width,
                    range.min_height, range.max_width, range.max_height, range.step_width,
                    range.step_height, mode.framerate_num, mode.framerate_den, mode.supported,
                    mode.reason);
}

bool same_mode(const CameraMode& left, const CameraMode& right) {
  return mode_key(left) == mode_key(right);
}

struct Dimension {
  std::uint32_t min = 0;
  std::uint32_t max = 0;
  std::uint32_t step = 1;
};

std::string trim(std::string value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
    value.pop_back();
  const auto begin = std::find_if_not(value.begin(), value.end(),
                                      [](unsigned char ch) { return std::isspace(ch) != 0; });
  value.erase(value.begin(), begin);
  return value;
}

std::string uppercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return value;
}

std::optional<std::string> read_text_file(const std::string& path) {
  std::ifstream input(path);
  if (!input)
    return std::nullopt;
  std::ostringstream text;
  text << input.rdbuf();
  return trim(text.str());
}

std::vector<std::string> directory_entries(const std::string& path, int* error = nullptr) {
  if (error)
    *error = 0;
  DIR* directory = ::opendir(path.c_str());
  if (!directory) {
    if (error)
      *error = errno;
    return {};
  }
  std::vector<std::string> entries;
  errno = 0;
  while (dirent* entry = ::readdir(directory)) {
    if (entry->d_name[0] != '.')
      entries.emplace_back(entry->d_name);
  }
  if (error && errno != 0)
    *error = errno;
  ::closedir(directory);
  std::sort(entries.begin(), entries.end());
  return entries;
}

std::optional<std::string> inaccessible_camera_device() {
  int directory_error = 0;
  for (const std::string& entry : directory_entries("/dev", &directory_error)) {
    if (entry.rfind("media", 0) != 0)
      continue;
    const std::string path = "/dev/" + entry;
    errno = 0;
    if (::access(path.c_str(), R_OK | W_OK) != 0 && errno == EACCES)
      return path;
  }
  if (directory_error == EACCES)
    return std::string("/dev");
  return std::nullopt;
}

bool value_to_dimension(const GValue* value, Dimension& out) {
  if (!value)
    return false;
  if (G_VALUE_HOLDS_INT(value)) {
    const int number = g_value_get_int(value);
    if (number <= 0)
      return false;
    out = {static_cast<std::uint32_t>(number), static_cast<std::uint32_t>(number), 1};
    return true;
  }
  if (GST_VALUE_HOLDS_INT_RANGE(value)) {
    const int minimum = gst_value_get_int_range_min(value);
    const int maximum = gst_value_get_int_range_max(value);
    const int step = gst_value_get_int_range_step(value);
    if (minimum <= 0 || maximum < minimum || step <= 0)
      return false;
    out = {static_cast<std::uint32_t>(minimum), static_cast<std::uint32_t>(maximum),
           static_cast<std::uint32_t>(step)};
    return true;
  }
  return false;
}

std::vector<std::string> structure_formats(const GstStructure* structure,
                                           const std::string& media_type) {
  const GValue* value = gst_structure_get_value(structure, "format");
  std::vector<std::string> formats;
  if (value && G_VALUE_HOLDS_STRING(value)) {
    if (const char* format = g_value_get_string(value); format && *format)
      formats.push_back(uppercase_ascii(format));
  } else if (value && GST_VALUE_HOLDS_LIST(value)) {
    const std::size_t count = gst_value_list_get_size(value);
    for (std::size_t index = 0; index < count; ++index) {
      const GValue* item = gst_value_list_get_value(value, index);
      if (item && G_VALUE_HOLDS_STRING(item)) {
        if (const char* format = g_value_get_string(item); format && *format)
          formats.push_back(uppercase_ascii(format));
      }
    }
  } else if (media_type == "image/jpeg") {
    formats.emplace_back("MJPEG");
  }
  return formats;
}

bool same_probe_mode(const ProbeMode& left, const ProbeMode& right) {
  return std::tie(left.media_type, left.format, left.min_width, left.min_height, left.max_width,
                  left.max_height, left.step_width, left.step_height, left.size_is_range) ==
         std::tie(right.media_type, right.format, right.min_width, right.min_height,
                  right.max_width, right.max_height, right.step_width, right.step_height,
                  right.size_is_range);
}

std::vector<ProbeMode> modes_from_caps(GstCaps* caps) {
  std::vector<ProbeMode> modes;
  if (!caps || gst_caps_is_any(caps) || gst_caps_is_empty(caps))
    return modes;
  const std::size_t count = gst_caps_get_size(caps);
  for (std::size_t index = 0; index < count; ++index) {
    const GstStructure* structure = gst_caps_get_structure(caps, index);
    if (!structure)
      continue;
    const char* structure_name = gst_structure_get_name(structure);
    const std::string media_type = structure_name ? structure_name : "";
    Dimension width;
    Dimension height;
    if (!value_to_dimension(gst_structure_get_value(structure, "width"), width) ||
        !value_to_dimension(gst_structure_get_value(structure, "height"), height))
      continue;

    for (const std::string& format : structure_formats(structure, media_type)) {
      ProbeMode mode;
      if (width.min == width.max && height.min == height.max) {
        mode = ProbeMode::discrete(media_type, format, width.min, height.min);
      } else {
        mode = ProbeMode::range(media_type, format, width.min, height.min, width.max, height.max,
                                width.step, height.step);
      }
      if (std::none_of(modes.begin(), modes.end(),
                       [&](const ProbeMode& current) { return same_probe_mode(current, mode); })) {
        modes.push_back(std::move(mode));
      }
    }
  }
  return modes;
}

std::string camera_model(GstDevice* device) {
  GstStructure* properties = gst_device_get_properties(device);
  if (!properties)
    return {};
  std::string model;
  const int fields = gst_structure_n_fields(properties);
  for (int index = 0; index < fields; ++index) {
    const char* name = gst_structure_nth_field_name(properties, index);
    if (!name || uppercase_ascii(name) != "MODEL")
      continue;
    const GValue* value = gst_structure_get_value(properties, name);
    if (value && G_VALUE_HOLDS_STRING(value)) {
      if (const char* text = g_value_get_string(value); text)
        model = text;
    }
    break;
  }
  gst_structure_free(properties);
  return model;
}

std::vector<ProbeCamera> probe_gstreamer_devices(ProbeFailure& failure, std::string& detail) {
  GstDeviceProvider* provider = gst_device_provider_factory_get_by_name(kLibcameraProvider);
  if (!provider) {
    failure = ProbeFailure::BackendUnavailable;
    detail = "GStreamer device provider 'libcameraprovider' is not installed or could not load";
    return {};
  }

  GList* devices = gst_device_provider_get_devices(provider);
  std::vector<ProbeCamera> cameras;
  for (GList* item = devices; item; item = item->next) {
    auto* device = GST_DEVICE(item->data);
    const char* display_name = gst_device_get_display_name(device);
    if (!display_name || !*display_name)
      continue;
    ProbeCamera camera;
    camera.name = display_name;
    camera.model = camera_model(device);
    GstCaps* caps = gst_device_get_caps(device);
    camera.modes = modes_from_caps(caps);
    if (caps)
      gst_caps_unref(caps);
    cameras.push_back(std::move(camera));
  }
  g_list_free_full(devices, reinterpret_cast<GDestroyNotify>(gst_object_unref));
  gst_object_unref(provider);

  if (cameras.empty()) {
    if (const auto path = inaccessible_camera_device()) {
      failure = ProbeFailure::PermissionDenied;
      detail = "permission denied accessing " + *path;
    }
  }
  return cameras;
}

using IspModeKey = std::tuple<std::string, std::uint32_t, std::uint32_t>;

std::string v4l2_format_token(std::uint32_t format) {
  if (format == V4L2_PIX_FMT_NV12 || format == V4L2_PIX_FMT_NV12M)
    return kSupportedFormat;
  const char value[] = {
      static_cast<char>(format & 0xff),
      static_cast<char>((format >> 8) & 0xff),
      static_cast<char>((format >> 16) & 0xff),
      static_cast<char>((format >> 24) & 0xff),
      '\0',
  };
  return value;
}

std::set<IspModeKey> enumerate_node_sizes(const std::string& device, std::string& error,
                                          bool& permission_denied) {
  std::set<IspModeKey> sizes;
  const int descriptor = ::open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
  if (descriptor < 0) {
    permission_denied = errno == EACCES;
    error = "could not open " + device + ": " + std::strerror(errno);
    return sizes;
  }

  v4l2_capability capability{};
  if (::ioctl(descriptor, VIDIOC_QUERYCAP, &capability) < 0) {
    permission_denied = errno == EACCES;
    error = "VIDIOC_QUERYCAP failed for " + device + ": " + std::strerror(errno);
    ::close(descriptor);
    return sizes;
  }
  if (std::string(reinterpret_cast<const char*>(capability.card)) != kIspCardName) {
    ::close(descriptor);
    return sizes;
  }

  const std::uint32_t caps =
      capability.device_caps ? capability.device_caps : capability.capabilities;
  const v4l2_buf_type type = (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
                                 ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
                                 : V4L2_BUF_TYPE_VIDEO_CAPTURE;
  for (std::uint32_t format_index = 0;; ++format_index) {
    v4l2_fmtdesc format{};
    format.index = format_index;
    format.type = type;
    errno = 0;
    if (::ioctl(descriptor, VIDIOC_ENUM_FMT, &format) < 0) {
      if (errno != EINVAL)
        error = "VIDIOC_ENUM_FMT failed for " + device + ": " + std::strerror(errno);
      break;
    }
    for (std::uint32_t size_index = 0;; ++size_index) {
      v4l2_frmsizeenum frame_size{};
      frame_size.index = size_index;
      frame_size.pixel_format = format.pixelformat;
      errno = 0;
      if (::ioctl(descriptor, VIDIOC_ENUM_FRAMESIZES, &frame_size) < 0) {
        if (errno != EINVAL)
          error = "VIDIOC_ENUM_FRAMESIZES failed for " + device + ": " + std::strerror(errno);
        break;
      }
      if (frame_size.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
        sizes.emplace(v4l2_format_token(format.pixelformat), frame_size.discrete.width,
                      frame_size.discrete.height);
      }
    }
  }
  ::close(descriptor);
  return sizes;
}

std::optional<std::vector<ProbeSize>> probe_isp_sizes(std::string& error, bool& permission_denied) {
  int directory_error = 0;
  const auto entries = directory_entries("/sys/class/video4linux", &directory_error);
  if (directory_error != 0) {
    permission_denied = directory_error == EACCES;
    error = "could not read /sys/class/video4linux: " + std::string(std::strerror(directory_error));
    return std::nullopt;
  }

  std::optional<std::set<IspModeKey>> common_sizes;
  std::size_t isp_nodes = 0;
  for (const std::string& entry : entries) {
    const auto name = read_text_file("/sys/class/video4linux/" + entry + "/name");
    if (!name || *name != kIspSysfsName)
      continue;
    const std::string device = "/dev/" + entry;
    std::string node_error;
    auto sizes = enumerate_node_sizes(device, node_error, permission_denied);
    if (!node_error.empty()) {
      error = std::move(node_error);
      return std::nullopt;
    }
    if (sizes.empty())
      continue;
    ++isp_nodes;
    if (!common_sizes) {
      common_sizes = std::move(sizes);
    } else {
      std::set<IspModeKey> intersection;
      std::set_intersection(common_sizes->begin(), common_sizes->end(), sizes.begin(), sizes.end(),
                            std::inserter(intersection, intersection.end()));
      common_sizes = std::move(intersection);
    }
  }

  if (isp_nodes == 0) {
    error = "no Modalix ISP output node was found";
    return std::nullopt;
  }
  if (!common_sizes || common_sizes->empty()) {
    error = "Modalix ISP output nodes reported no common discrete sizes";
    return std::nullopt;
  }
  std::vector<ProbeSize> result;
  result.reserve(common_sizes->size());
  for (const auto& [format, width, height] : *common_sizes)
    result.push_back({width, height, format});
  return result;
}

[[noreturn]] void throw_discovery_error(const char* code, const std::string& message) {
  GraphReport report;
  report.error_code = code;
  report.repro_note = message;
  throw NeatError("[" + report.error_code + "] " + message, std::move(report));
}

bool isp_supports(const std::vector<ProbeSize>& sizes, const std::string& format,
                  std::uint32_t width, std::uint32_t height) {
  return std::any_of(sizes.begin(), sizes.end(), [&](const ProbeSize& size) {
    return size.format == format && size.width == width && size.height == height;
  });
}

CameraMode classify_mode(const ProbeMode& candidate, const CameraProbe& probe) {
  CameraMode mode;
  mode.format = candidate.format;
  mode.framerate_num = kDefaultFramerateNum;
  mode.framerate_den = kDefaultFramerateDen;
  if (candidate.size_is_range) {
    mode.size_range =
        CameraSizeRange{candidate.min_width,  candidate.min_height, candidate.max_width,
                        candidate.max_height, candidate.step_width, candidate.step_height};
  } else {
    mode.width = candidate.min_width;
    mode.height = candidate.min_height;
  }

  if (candidate.media_type != "video/x-raw") {
    mode.reason =
        "CameraInput requests video/x-raw; this is advertised as " + candidate.media_type + ".";
    return mode;
  }
  if (candidate.format != kSupportedFormat) {
    mode.reason = "CameraInput's current camera-memory path supports NV12 output only.";
    return mode;
  }
  if (candidate.size_is_range) {
    mode.reason = "The continuous size range is advisory; CameraInput support is reported only "
                  "for discrete ISP output sizes.";
    return mode;
  }
  if (!probe.isp_sizes) {
    mode.reason = "CameraInput support could not be verified because " +
                  (probe.isp_error.empty() ? std::string("ISP output sizes are unavailable")
                                           : probe.isp_error) +
                  ".";
    return mode;
  }
  if (!isp_supports(*probe.isp_sizes, mode.format, mode.width, mode.height)) {
    mode.reason = "This resolution is not an ISP output size on this board.";
    return mode;
  }
  mode.supported = true;
  return mode;
}

} // namespace

void canonicalize_camera_catalog(std::vector<CameraInfo>& cameras) {
  for (CameraInfo& camera : cameras) {
    std::sort(camera.modes.begin(), camera.modes.end(),
              [](const CameraMode& left, const CameraMode& right) {
                return mode_key(left) < mode_key(right);
              });
    camera.modes.erase(std::unique(camera.modes.begin(), camera.modes.end(), same_mode),
                       camera.modes.end());
  }
  std::sort(cameras.begin(), cameras.end(), [](const CameraInfo& left, const CameraInfo& right) {
    return std::tie(left.name, left.model) < std::tie(right.name, right.model);
  });
}

ProbeMode ProbeMode::discrete(std::string media_type_value, std::string format_value,
                              std::uint32_t width, std::uint32_t height) {
  ProbeMode mode;
  mode.media_type = std::move(media_type_value);
  mode.format = std::move(format_value);
  mode.min_width = mode.max_width = width;
  mode.min_height = mode.max_height = height;
  return mode;
}

ProbeMode ProbeMode::range(std::string media_type_value, std::string format_value,
                           std::uint32_t min_width_value, std::uint32_t min_height_value,
                           std::uint32_t max_width_value, std::uint32_t max_height_value,
                           std::uint32_t step_width_value, std::uint32_t step_height_value) {
  ProbeMode mode;
  mode.media_type = std::move(media_type_value);
  mode.format = std::move(format_value);
  mode.min_width = min_width_value;
  mode.min_height = min_height_value;
  mode.max_width = max_width_value;
  mode.max_height = max_height_value;
  mode.step_width = step_width_value;
  mode.step_height = step_height_value;
  mode.size_is_range = true;
  return mode;
}

CameraProbe probe_camera_backend() {
  CameraProbe probe;
  try {
    gst_init_once();
  } catch (const std::exception& error) {
    probe.failure = ProbeFailure::BackendUnavailable;
    probe.failure_detail = error.what();
    return probe;
  }

  probe.cameras = probe_gstreamer_devices(probe.failure, probe.failure_detail);
  if (probe.failure != ProbeFailure::None || probe.cameras.empty())
    return probe;

  bool permission_denied = false;
  probe.isp_sizes = probe_isp_sizes(probe.isp_error, permission_denied);
  if (permission_denied) {
    probe.failure = ProbeFailure::PermissionDenied;
    probe.failure_detail = probe.isp_error;
  }
  return probe;
}

std::vector<CameraInfo> build_camera_catalog(const CameraProbe& probe, bool allow_no_cameras) {
  if (probe.failure == ProbeFailure::BackendUnavailable) {
    throw_discovery_error(
        error_codes::kPluginMissing,
        "Camera discovery backend is unavailable: " + probe.failure_detail +
            ". Install the GStreamer libcamera plugin used by CameraInput, then try again.");
  }
  if (probe.failure == ProbeFailure::PermissionDenied) {
    throw_discovery_error(
        error_codes::kPermissionDenied,
        "Camera discovery was denied access: " + probe.failure_detail +
            ". Add this account to the board's video group, reconnect, and try again.");
  }
  if (probe.cameras.empty()) {
    if (allow_no_cameras)
      return {};
    throw_discovery_error(
        error_codes::kCameraNotFound,
        "No CameraInput cameras were found. Check the camera ribbon cable and device-tree "
        "overlay, then try again.");
  }

  std::vector<CameraInfo> cameras;
  cameras.reserve(probe.cameras.size());
  for (const ProbeCamera& candidate : probe.cameras) {
    CameraInfo camera;
    camera.name = candidate.name;
    camera.model = candidate.model;
    camera.modes.reserve(candidate.modes.size());
    for (const ProbeMode& mode : candidate.modes)
      camera.modes.push_back(classify_mode(mode, probe));
    cameras.push_back(std::move(camera));
  }
  canonicalize_camera_catalog(cameras);
  return cameras;
}

std::vector<PeripheralRecord> camera_records(const std::vector<CameraInfo>& cameras) {
  std::vector<PeripheralRecord> devices;
  devices.reserve(cameras.size());
  for (const auto& camera : cameras) {
    nlohmann::json modes = nlohmann::json::array();
    for (const auto& mode : camera.modes) {
      nlohmann::json serialized = {
          {"format", mode.format},
          {"framerate_num", mode.framerate_num},
          {"framerate_den", mode.framerate_den},
          {"supported", mode.supported},
          {"reason", mode.reason},
      };
      if (mode.size_range) {
        serialized["size_range"] = {
            {"min_width", mode.size_range->min_width},
            {"min_height", mode.size_range->min_height},
            {"max_width", mode.size_range->max_width},
            {"max_height", mode.size_range->max_height},
            {"step_width", mode.size_range->step_width},
            {"step_height", mode.size_range->step_height},
        };
      } else {
        serialized["width"] = mode.width;
        serialized["height"] = mode.height;
      }
      modes.push_back(std::move(serialized));
    }

    PeripheralRecord record;
    record.id = "camera:" + camera.name;
    record.type = "camera";
    record.provider = "daemon.camera.libcamera";
    record.details = {
        {"camera_name", camera.name},
        {"backend", "libcamera"},
        {"modes", std::move(modes)},
    };
    if (!camera.model.empty())
      record.details["model"] = camera.model;
    devices.push_back(std::move(record));
  }
  return devices;
}

std::vector<PeripheralRecord> discover_camera_peripherals() {
  return camera_records(build_camera_catalog(probe_camera_backend(), true));
}

} // namespace simaai::neat::peripherals_internal
