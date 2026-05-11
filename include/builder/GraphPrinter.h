/**
 * @file
 * @ingroup builder
 * @brief Builder/Graph pretty printers (text, DOT, Mermaid).
 *
 * `GraphPrinter` produces deterministic, human-readable views of a `NodeGroup`
 * or `Graph` *before* the Session converts it into a gst-launch string. It
 * is strictly STL-only — no GStreamer dependencies — so it can be invoked
 * from CI, dump tools, and Markdown report generators.
 *
 * @see NodeGroup
 * @see Graph
 */
// include/builder/GraphPrinter.h
#pragma once

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "builder/Graph.h"
#include "builder/Node.h"
#include "builder/NodeGroup.h"

namespace simaai::neat {

/**
 * @brief STL-only pretty printers for Builder/Graph artifacts.
 *
 * Purpose (aligned with the architecture):
 * - Give humans a deterministic, readable view of the Node graph/list *before*
 *   Session turns it into a gst-launch string (gst_parse_launch).
 * - Keep this strictly in builder/ (no GStreamer, no pipeline/ dependencies).
 *
 * Output formats:
 * - Text list (for NodeGroup / chain)
 * - DOT (Graphviz) for Graph
 * - Mermaid flowchart for Graph
 *
 * @ingroup builder
 * @see NodeGroup
 * @see Graph
 */
class GraphPrinter final {
public:
  /**
   * @brief Knobs controlling the printers' verbosity and per-format behavior.
   */
  struct Options {
    // Common
    bool show_index = true;       ///< Prepend deterministic Node index (0..N-1).
    bool show_kind = true;        ///< Print `Node::kind()` for each Node.
    bool show_user_label = true;  ///< Print user-supplied label when non-empty.

    // Extra node info (still builder-only, calls Node::backend_fragment / element_names)
    bool show_backend_fragment = false; ///< Include each Node's gst fragment.
    bool show_element_names = false;    ///< Include each Node's deterministic element names.

    // Truncation controls (avoid huge debug dumps)
    std::size_t max_user_label_chars = 200; ///< Truncate user labels to this many chars.
    std::size_t max_fragment_chars = 220;   ///< Truncate gst fragments to this many chars.
    std::size_t max_elements_per_node = 16; ///< Cap per-Node element list before "...(N more)".

    // DOT controls
    bool dot_rankdir_lr = true;                  ///< Left-to-right layout (vs top-down).
    std::string dot_graph_name = "sima_graph";   ///< `digraph <name>` identifier.

    // Mermaid controls
    bool mermaid_lr = true;                ///< Left-to-right (LR) vs top-down (TD).
    std::string mermaid_id_prefix = "n";   ///< Mermaid node id prefix (must be identifier-like).
  };

  // --------------------------
  // Text (linear lists)
  // --------------------------

  /// @brief Print a linear NodeGroup in deterministic index order (0..N-1) with default options.
  static std::string to_text(const NodeGroup& group) {
    return to_text(group, Options{});
  }

  /// @brief Print a linear NodeGroup with explicit formatting options.
  static std::string to_text(const NodeGroup& group, const Options& opt) {
    std::ostringstream oss;
    const auto& nodes = group.nodes();

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

  // --------------------------
  // DOT (Graphviz)
  // --------------------------

  /// @brief Produce a DOT digraph for Graphviz with default options.
  static std::string to_dot(const Graph& g) {
    return to_dot(g, Options{});
  }

  /// @brief Produce a DOT digraph for Graphviz with explicit formatting options.
  static std::string to_dot(const Graph& g, const Options& opt) {
    std::ostringstream oss;

    oss << "digraph " << dot_id_(opt.dot_graph_name) << " {\n";
    if (opt.dot_rankdir_lr)
      oss << "  rankdir=LR;\n";
    oss << "  node [shape=box];\n";

    // Nodes
    for (Graph::NodeId id = 0; id < g.node_count(); ++id) {
      const auto& n = g.node(id);
      if (!n)
        continue;

      std::string label = node_label_(n, id, opt);

      // Optional heavy builder-only details
      if (opt.show_backend_fragment) {
        std::string frag =
            truncate_(n->backend_fragment(static_cast<int>(id)), opt.max_fragment_chars);
        label += "\\n" + dot_escape_("gst: " + frag);
      }
      if (opt.show_element_names) {
        const auto elems = n->element_names(static_cast<int>(id));
        label += "\\n" + dot_escape_("elems: " + join_trunc_(elems, opt.max_elements_per_node));
      }

      oss << "  " << dot_node_id_(id) << " [label=\"" << dot_escape_(label) << "\"];\n";
    }

    // Edges
    for (const auto& e : g.edges()) {
      oss << "  " << dot_node_id_(e.from) << " -> " << dot_node_id_(e.to) << ";\n";
    }

    oss << "}\n";
    return oss.str();
  }

  // --------------------------
  // Mermaid
  // --------------------------

  /// @brief Produce a Mermaid flowchart (useful in Markdown/Confluence) with default options.
  static std::string to_mermaid(const Graph& g) {
    return to_mermaid(g, Options{});
  }

  /// @brief Produce a Mermaid flowchart with explicit formatting options.
  static std::string to_mermaid(const Graph& g, const Options& opt) {
    std::ostringstream oss;
    oss << "flowchart " << (opt.mermaid_lr ? "LR" : "TD") << "\n";

    // Nodes
    for (Graph::NodeId id = 0; id < g.node_count(); ++id) {
      const auto& n = g.node(id);
      if (!n)
        continue;

      // Mermaid node id (must be identifier-like)
      const std::string mid = mermaid_node_id_(id, opt);

      // Mermaid label (quoted in brackets)
      std::string label = node_label_(n, id, opt);

      if (opt.show_backend_fragment) {
        std::string frag =
            truncate_(n->backend_fragment(static_cast<int>(id)), opt.max_fragment_chars);
        label += "\n" + ("gst: " + frag);
      }
      if (opt.show_element_names) {
        const auto elems = n->element_names(static_cast<int>(id));
        label += "\n" + ("elems: " + join_trunc_(elems, opt.max_elements_per_node));
      }

      oss << "  " << mid << "[\"" << mermaid_escape_(label) << "\"]\n";
    }

    // Edges
    for (const auto& e : g.edges()) {
      oss << "  " << mermaid_node_id_(e.from, opt) << " --> " << mermaid_node_id_(e.to, opt)
          << "\n";
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

  static std::string node_label_(const std::shared_ptr<Node>& n, Graph::NodeId id,
                                 const Options& opt) {
    std::ostringstream oss;

    bool first = true;

    if (opt.show_index) {
      oss << id;
      first = false;
    }

    if (opt.show_kind) {
      if (!first)
        oss << ": ";
      oss << n->kind();
      first = false;
    }

    if (opt.show_user_label) {
      const std::string ul = n->user_label();
      if (!ul.empty()) {
        if (!first)
          oss << "\\n";
        oss << truncate_(ul, opt.max_user_label_chars);
      }
    }

    return oss.str();
  }

  static std::string truncate_(const std::string& s, std::size_t max_chars) {
    if (max_chars == 0)
      return "";
    if (s.size() <= max_chars)
      return s;
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

  // DOT helpers
  static std::string dot_escape_(std::string s) {
    // Escape for DOT label="...".
    // - backslash, quotes
    // - preserve \n sequences (we pass real \n as \\n in label strings)
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
      switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        break;
      default:
        out += c;
        break;
      }
    }
    return out;
  }

  static std::string dot_id_(const std::string& s) {
    if (s.empty())
      return "sima_graph";
    // Keep it conservative: if it looks like an identifier, use it; else quote.
    for (char c : s) {
      const bool ok =
          (c == '_' || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'));
      if (!ok) {
        return "\"" + dot_escape_(s) + "\"";
      }
    }
    return s;
  }

  static std::string dot_node_id_(Graph::NodeId id) {
    // Deterministic node ids.
    return "n" + std::to_string(id);
  }

  // Mermaid helpers
  static std::string mermaid_node_id_(Graph::NodeId id, const Options& opt) {
    return opt.mermaid_id_prefix + std::to_string(id);
  }

  static std::string mermaid_escape_(std::string s) {
    // Mermaid label inside ["..."].
    // Escape quotes and backslashes; keep newlines (Mermaid supports them).
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
      switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\r':
        break;
      default:
        out += c;
        break;
      }
    }
    return out;
  }
};

} // namespace simaai::neat
