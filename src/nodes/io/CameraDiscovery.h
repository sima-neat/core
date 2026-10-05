#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace simaai::neat::camera_discovery {
// Private, portable snapshot: matching can be tested without opening devices.
struct Entity {
  std::uint32_t id;
  std::string name;
  bool sensor;
};
struct Link {
  std::uint32_t source;
  std::uint32_t sink;
  bool enabled;
};
struct Endpoint {
  std::uint32_t entity;
  std::string device;
};
struct Topology {
  std::vector<Entity> entities;
  std::vector<Link> links;
  std::vector<Endpoint> endpoints;
};
std::string select_simor_device(const std::vector<Topology>& graphs, const std::string& requested);
// Resolve only controllers belonging to the selected video device through sysfs.
// Kept separate from ioctl discovery so isolation is testable without hardware.
std::vector<std::string> media_nodes_for_video_device(const std::filesystem::path& video_sysfs);
// Read media-controller metadata only. Never open a capture/subdevice node,
// call libcamera, change formats/links, or allocate/queue camera buffers.
std::string find_simor_device(const std::string& requested);
} // namespace simaai::neat::camera_discovery
