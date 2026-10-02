#include "nodes/io/CameraDiscovery.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#if defined(__linux__)
#include <fstream>
#include <fcntl.h>
#include <linux/media.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace simaai::neat::camera_discovery {
namespace {
bool is_media_node_name(const std::string& name) {
  return name.size() > 5 && name.rfind("media", 0) == 0 &&
         std::all_of(name.begin() + 5, name.end(), [](char c) { return c >= '0' && c <= '9'; });
}
} // namespace

std::vector<std::string> media_nodes_for_video_device(const std::filesystem::path& video_sysfs) {
  std::error_code error;
  const auto owner = std::filesystem::canonical(video_sysfs / "device", error);
  if (error)
    throw std::runtime_error("CameraInput cannot resolve the selected device's media controller; "
                             "specify device and profile explicitly to bypass discovery");
  std::vector<std::string> nodes;
  for (const auto& entry : std::filesystem::directory_iterator(owner)) {
    const auto name = entry.path().filename().string();
    if (is_media_node_name(name) && entry.is_directory())
      nodes.push_back("/dev/" + name);
  }
  if (nodes.empty())
    throw std::runtime_error("CameraInput selected device has no discoverable media controller; "
                             "specify device and profile explicitly to bypass discovery");
  std::sort(nodes.begin(), nodes.end());
  return nodes;
}

std::string select_simor_device(const std::vector<Topology>& graphs, const std::string& requested) {
  std::unordered_set<std::string> matches;
  for (const auto& graph : graphs) {
    for (const auto& endpoint : graph.endpoints) {
      if (!requested.empty() && endpoint.device != requested)
        continue;
      std::vector<std::uint32_t> pending{endpoint.entity};
      std::unordered_set<std::uint32_t> visited;
      std::vector<const Entity*> sensors;
      while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (!visited.insert(id).second)
          continue;
        const auto entity = std::find_if(graph.entities.begin(), graph.entities.end(),
                                         [id](const auto& value) { return value.id == id; });
        if (entity == graph.entities.end())
          throw std::runtime_error("CameraInput media topology contains a missing entity");
        if (entity->sensor)
          sensors.push_back(&*entity);
        for (const auto& link : graph.links) {
          if (link.enabled && link.sink == id)
            pending.push_back(link.source);
        }
      }
      // An entity-level walk deliberately rejects mixed-sensor routing. It must
      // not guess which sensor feeds a mux's active stream from entity names.
      if (sensors.size() == 1 &&
          (sensors[0]->name == "simor_metoak" || sensors[0]->name.rfind("simor_metoak ", 0) == 0))
        matches.insert(endpoint.device);
    }
  }
  if (matches.size() != 1)
    throw std::runtime_error(
        "CameraInput cannot uniquely identify a Metoak SIMOR capture endpoint; "
        "specify device and profile explicitly and verify the media topology");
  return *matches.begin();
}

#if defined(__linux__)
namespace {
struct MediaFd {
  int value;
  ~MediaFd() {
    if (value >= 0)
      ::close(value);
  }
};

Topology read_topology(int fd) {
  constexpr std::uint32_t limit = 4096;
  for (unsigned attempt = 0; attempt < 3; ++attempt) {
    media_v2_topology header{};
    if (::ioctl(fd, MEDIA_IOC_G_TOPOLOGY, &header) < 0)
      throw std::runtime_error("CameraInput cannot read media topology");
    if (header.num_entities > limit || header.num_interfaces > limit || header.num_pads > limit ||
        header.num_links > limit)
      throw std::runtime_error("CameraInput media topology exceeds discovery limits");
    std::vector<media_v2_entity> entities(header.num_entities);
    std::vector<media_v2_interface> interfaces(header.num_interfaces);
    std::vector<media_v2_pad> pads(header.num_pads);
    std::vector<media_v2_link> links(header.num_links);
    const auto version = header.topology_version;
    header.ptr_entities = reinterpret_cast<std::uintptr_t>(entities.data());
    header.ptr_interfaces = reinterpret_cast<std::uintptr_t>(interfaces.data());
    header.ptr_pads = reinterpret_cast<std::uintptr_t>(pads.data());
    header.ptr_links = reinterpret_cast<std::uintptr_t>(links.data());
    if (::ioctl(fd, MEDIA_IOC_G_TOPOLOGY, &header) < 0 || header.topology_version != version)
      continue;
    if (header.num_entities > entities.size() || header.num_interfaces > interfaces.size() ||
        header.num_pads > pads.size() || header.num_links > links.size())
      throw std::runtime_error("CameraInput inconsistent media topology counts");
    entities.resize(header.num_entities);
    interfaces.resize(header.num_interfaces);
    pads.resize(header.num_pads);
    links.resize(header.num_links);
    Topology result;
    for (const auto& entity : entities) {
      const auto end = std::find(std::begin(entity.name), std::end(entity.name), '\0');
      result.entities.push_back(
          {entity.id, std::string(entity.name, end), entity.function == MEDIA_ENT_F_CAM_SENSOR});
    }
    for (const auto& link : links) {
      const auto type = link.flags & MEDIA_LNK_FL_LINK_TYPE;
      if (type == MEDIA_LNK_FL_DATA_LINK) {
        const auto source = std::find_if(pads.begin(), pads.end(),
                                         [&](const auto& p) { return p.id == link.source_id; });
        const auto sink = std::find_if(pads.begin(), pads.end(),
                                       [&](const auto& p) { return p.id == link.sink_id; });
        if (source == pads.end() || sink == pads.end())
          throw std::runtime_error("CameraInput invalid media pad link");
        result.links.push_back(
            {source->entity_id, sink->entity_id, (link.flags & MEDIA_LNK_FL_ENABLED) != 0});
      } else if (type == MEDIA_LNK_FL_INTERFACE_LINK) {
        const auto interface = std::find_if(interfaces.begin(), interfaces.end(),
                                            [&](const auto& i) { return i.id == link.source_id; });
        if (interface == interfaces.end() || interface->intf_type != MEDIA_INTF_T_V4L_VIDEO)
          continue;
        // Resolve by device number, not enumeration order. Reading sysfs does
        // not run the capture driver's open/release callbacks.
        const auto devno = std::to_string(interface->devnode.major) + ":" +
                           std::to_string(interface->devnode.minor);
        std::ifstream uevent("/sys/dev/char/" + devno + "/uevent");
        std::string line;
        while (std::getline(uevent, line)) {
          if (line.rfind("DEVNAME=", 0) != 0)
            continue;
          const std::string name = line.substr(8);
          if (name.empty() || name.find('/') != std::string::npos)
            throw std::runtime_error("CameraInput invalid capture device name");
          const std::string device = "/dev/" + name;
          struct stat st {};
          if (::stat(device.c_str(), &st) == 0 && S_ISCHR(st.st_mode) &&
              major(st.st_rdev) == interface->devnode.major &&
              minor(st.st_rdev) == interface->devnode.minor)
            result.endpoints.push_back({link.sink_id, device});
        }
      }
    }
    return result;
  }
  throw std::runtime_error("CameraInput media topology changed during discovery; retry selection");
}
} // namespace
#endif

std::string find_simor_device(const std::string& requested) {
#if defined(__linux__)
  std::string canonical;
  if (!requested.empty()) {
    std::error_code error;
    canonical = std::filesystem::canonical(requested, error).string();
    if (error)
      throw std::runtime_error("CameraInput selected device does not exist: " + requested);
  }
  std::vector<std::string> media_nodes;
  if (!canonical.empty()) {
    struct stat st {};
    if (::stat(canonical.c_str(), &st) < 0 || !S_ISCHR(st.st_mode))
      throw std::runtime_error("CameraInput selected device is not a character device: " +
                               canonical);
    const auto devno = std::to_string(major(st.st_rdev)) + ":" + std::to_string(minor(st.st_rdev));
    // Filter before opening/querying metadata. Even G_TOPOLOGY takes the
    // kernel's graph mutex: an unrelated wedged controller must not block
    // selection of this device. Never fall back to a system-wide scan.
    media_nodes = media_nodes_for_video_device(std::filesystem::path("/sys/dev/char") / devno);
  } else {
    // With no selected endpoint all controllers are candidates. Callers that
    // cannot inspect system-wide metadata must supply both device and profile.
    for (const auto& entry : std::filesystem::directory_iterator("/dev")) {
      if (is_media_node_name(entry.path().filename().string()))
        media_nodes.push_back(entry.path().string());
    }
    std::sort(media_nodes.begin(), media_nodes.end());
  }
  std::vector<Topology> graphs;
  for (const auto& media_node : media_nodes) {
    MediaFd fd{::open(media_node.c_str(), O_RDONLY | O_CLOEXEC)};
    if (fd.value < 0)
      throw std::runtime_error("CameraInput cannot open media metadata: " + media_node);
    graphs.push_back(read_topology(fd.value));
  }
  return select_simor_device(graphs, canonical);
#else
  (void)requested;
  throw std::runtime_error("CameraInput V4L2 discovery requires Linux");
#endif
}
} // namespace simaai::neat::camera_discovery
