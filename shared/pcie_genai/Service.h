#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace simaai::neat::pcie::genai::wire {
// A handle belongs to one thread. Use separate handles for control and file IO.
class Service {
public:
  explicit Service(int card, bool endpoint = false);
  ~Service();
  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;
  void subscribe(const std::string& tag);
  void send(const std::string& tag, const std::string& payload);
  std::optional<std::string> receive(int timeout_ms);
  bool fetch(const std::string& root, const std::string& source, const std::string& destination,
             bool optional = false);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace simaai::neat::pcie::genai::wire
