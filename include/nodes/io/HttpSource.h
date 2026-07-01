/**
 * @file
 * @ingroup nodes_io
 * @brief `HttpSource` Node — source that reads from an HTTP or HTTPS URL.
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Construction options for an `HttpSource` Node.
 *
 * Empty strings leave optional `souphttpsrc` string properties at their element defaults.
 */
struct HttpSourceOptions {
  std::string location;      ///< HTTP/HTTPS URL to read from.
  int timeout_seconds = 15;  ///< Blocking I/O timeout in seconds; `0` disables timeout.
  int retries = 3;           ///< Maximum retries before failing; `-1` means infinite.
  bool is_live = false;      ///< If true, mark the source as live.
  bool do_timestamp = false; ///< If true, timestamp outgoing buffers with stream time.
  std::string user_agent;    ///< Optional HTTP User-Agent override.
};

/**
 * @brief HTTP source Node backed by GStreamer's `souphttpsrc`.
 *
 * Use this at the head of graphs that consume HTTP/HTTPS media streams such as
 * multipart MJPEG. The node is source-owned, so build the graph without pushed input.
 *
 * @ingroup nodes_io
 */
class HttpSource final : public Node {
public:
  explicit HttpSource(HttpSourceOptions opt);

  std::string kind() const override {
    return "HttpSource";
  }
  std::string user_label() const override {
    return opt_.location;
  }
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const HttpSourceOptions& options() const {
    return opt_;
  }

private:
  HttpSourceOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for an `HttpSource` Node.
std::shared_ptr<simaai::neat::Node> HttpSource(HttpSourceOptions opt);
} // namespace simaai::neat::nodes
