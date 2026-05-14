/**
 * @file
 * @ingroup nodes_common
 * @brief `FileInput` Node — file-system source. Reads encoded media from a file path.
 *
 * Source-role Node (`InputRole::Source`), so a Session that starts with a `FileInput`
 * uses `Session::run()` (no `push()`).
 */
#pragma once

#include "builder/Node.h"

#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Wraps GStreamer's `filesrc` element — reads encoded media from a path on disk.
 *
 * Typical placement: first Node in a Session that processes a video / image file.
 *
 * Usage:
 * @code
 * sess.add(sima::nodes::FileInput("input.mp4"));
 * @endcode
 *
 * @ingroup nodes_common
 */
class FileInput final : public Node {
public:
  /// Construct with the filesystem path to read from.
  explicit FileInput(std::string path);

  /// Type label for this Node kind.
  std::string kind() const override {
    return "FileInput";
  }
  /// User-facing label for this Node.
  std::string user_label() const override {
    return path_;
  }
  /// Role this Node plays as a stream source.
  InputRole input_role() const override {
    return InputRole::Source;
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;

  /// The path this `FileInput` was constructed with.
  const std::string& path() const {
    return path_;
  }

private:
  std::string path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `FileInput` Node.
std::shared_ptr<simaai::neat::Node> FileInput(std::string path);
} // namespace simaai::neat::nodes
