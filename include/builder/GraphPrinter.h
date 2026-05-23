/**
 * @file
 * @ingroup builder
 * @brief Builder-layer node-list pretty printer.
 *
 * `GraphPrinter` produces deterministic, human-readable views of linear Node
 * lists before public `Graph` converts them into a gst-launch string. It is
 * strictly STL-only — no GStreamer dependencies — so it can be invoked from CI,
 * dump tools, and Markdown report generators.
 */
// include/builder/GraphPrinter.h
#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "builder/Node.h"

namespace simaai::neat {

/**
 * @brief STL-only pretty printer for builder node lists.
 *
 * Purpose (aligned with the architecture):
 * - Give humans a deterministic, readable view of a Node list *before*
 *   Graph turns it into a gst-launch string (gst_parse_launch).
 * - Keep this strictly in builder/ (no GStreamer, no pipeline/ dependencies).
 *
 * Output formats:
 * - Text list (for node lists / chains)
 *
 * @ingroup builder
 */
class GraphPrinter final {
public:
  /**
   * @brief Knobs controlling text-printer verbosity.
   */
  struct Options {
    // Common
    bool show_index = true;      ///< Prepend deterministic Node index (0..N-1).
    bool show_kind = true;       ///< Print `Node::kind()` for each Node.
    bool show_user_label = true; ///< Print user-supplied label when non-empty.

    // Extra node info (still builder-only, calls Node::backend_fragment / element_names)
    bool show_backend_fragment = false; ///< Include each Node's gst fragment.
    bool show_element_names = false;    ///< Include each Node's deterministic element names.

    // Truncation controls (avoid huge debug dumps)
    std::size_t max_user_label_chars = 200; ///< Truncate user labels to this many chars.
    std::size_t max_fragment_chars = 220;   ///< Truncate gst fragments to this many chars.
    std::size_t max_elements_per_node = 16; ///< Cap per-Node element list before "...(N more)".
  };

  // --------------------------
  // Text (linear lists)
  // --------------------------

  /// @brief Print a linear node list in deterministic index order (0..N-1) with default options.
  static std::string to_text(std::span<const std::shared_ptr<Node>> nodes) {
    return to_text(nodes, Options{});
  }

  /// @brief Print a linear node list with explicit formatting options.
  static std::string to_text(std::span<const std::shared_ptr<Node>> nodes, const Options& opt) {
    std::ostringstream oss;

    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const auto& n = nodes[i];
      if (!n)
        continue;

      oss << format_node_line_(n, i, opt);

      if (opt.show_backend_fragment) {
        const std::string frag = n->backend_fragment(static_cast<int>(i));
        oss << "\n    gst: " << truncate_(frag, opt.max_fragment_chars);
      }
      if (opt.show_element_names) {
        const auto elems = n->element_names(static_cast<int>(i));
        oss << "\n    elems: " << join_trunc_(elems, opt.max_elements_per_node);
      }

      if (i + 1 < nodes.size())
        oss << "\n";
    }

    return oss.str();
  }

private:
  static std::string format_node_line_(const std::shared_ptr<Node>& n, std::size_t index,
                                       const Options& opt) {
    std::ostringstream oss;

    if (opt.show_index) {
      oss << index << ") ";
    }

    if (opt.show_kind) {
      oss << n->kind();
    } else {
      oss << "Node";
    }

    if (opt.show_user_label) {
      const std::string ul = n->user_label();
      if (!ul.empty()) {
        oss << "  [" << truncate_(ul, opt.max_user_label_chars) << "]";
      }
    }

    return oss.str();
  }

  static std::string truncate_(const std::string& s, std::size_t max_chars) {
    if (max_chars == 0)
      return "";
    if (s.size() <= max_chars)
      return s;
    if (max_chars <= 3)
      return s.substr(0, max_chars);
    // Keep it simple + deterministic.
    return s.substr(0, max_chars - 3) + "...";
  }

  static std::string join_trunc_(const std::vector<std::string>& items, std::size_t max_items) {
    std::ostringstream oss;
    const std::size_t n = items.size();
    const std::size_t m = (max_items == 0) ? 0 : (n < max_items ? n : max_items);

    for (std::size_t i = 0; i < m; ++i) {
      if (i)
        oss << ", ";
      oss << items[i];
    }
    if (n > m) {
      if (m)
        oss << ", ";
      oss << "...(" << (n - m) << " more)";
    }
    return oss.str();
  }
};

} // namespace simaai::neat
