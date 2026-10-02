#pragma once

#include <cstdint>
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
// Read media-controller metadata only. Never open a capture/subdevice node,
// call libcamera, change formats/links, or allocate/queue camera buffers.
std::string find_simor_device(const std::string& requested);
} // namespace simaai::neat::camera_discovery
