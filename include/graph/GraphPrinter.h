/**
 * @file
 * @ingroup graph
 * @brief STL-only pretty printers for hybrid graphs.
 */
#pragma once

#include "graph/Graph.h"

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::graph {

class GraphPrinter final {
public:
  struct Options {
    bool show_index = true;
    bool show_kind = true;
    bool show_user_label = true;
    bool show_backend = true;
    bool show_ports = true;

    std::size_t max_label_chars = 200;
    std::size_t max_ports = 16;

    bool dot_rankdir_lr = true;
    std::string dot_graph_name = "sima_hybrid_graph";

    bool mermaid_lr = true;
    std::string mermaid_id_prefix = "n";
  };

  static std::string to_text(const Graph& g) {
    return to_text(g, Options{});
  }
  static std::string to_text(const Graph& g, const Options& opt) {
    std::ostringstream oss;
    for (NodeId id = 0; id < g.node_count(); ++id) {
      const auto& n = g.node(id);
      if (!n)
        continue;

      if (opt.show_index)
        oss << id << ") ";
      if (opt.show_kind)
        oss << n->kind();
      if (opt.show_backend) {
        oss << " [" << backend_name_(n->backend()) << "]";
      }
      if (opt.show_user_label) {
        const std::string label = n->user_label();
        if (!label.empty())
          oss << " '" << truncate_(label, opt.max_label_chars) << "'";
      }

      if (opt.show_ports) {
        oss << "\n    in: " << join_ports_(n->input_ports(), opt.max_ports);
        oss << "\n    out: " << join_ports_(n->output_ports(), opt.max_ports);
      }

      if (id + 1 < g.node_count())
        oss << "\n";
    }
    return oss.str();
  }

  static std::string to_dot(const Graph& g) {
    return to_dot(g, Options{});
  }
  static std::string to_dot(const Graph& g, const Options& opt) {
    std::ostringstream oss;
    oss << "digraph " << dot_id_(opt.dot_graph_name) << " {\n";
    if (opt.dot_rankdir_lr)
      oss << "  rankdir=LR;\n";
    oss << "  node [shape=box];\n";

    for (NodeId id = 0; id < g.node_count(); ++id) {
      const auto& n = g.node(id);
      if (!n)
        continue;

      std::string label = node_label_(n, id, opt);
      if (opt.show_ports) {
        label += "\\n" + dot_escape_("in: " + join_ports_(n->input_ports(), opt.max_ports));
        label += "\\n" + dot_escape_("out: " + join_ports_(n->output_ports(), opt.max_ports));
      }

      oss << "  " << dot_node_id_(id) << " [label=\"" << dot_escape_(label) << "\"];\n";
    }

    for (const auto& e : g.edges()) {
      const std::string label = g.port_name(e.from_port) + "->" + g.port_name(e.to_port);
      oss << "  " << dot_node_id_(e.from) << " -> " << dot_node_id_(e.to) << " [label=\""
          << dot_escape_(label) << "\"];\n";
    }

    oss << "}\n";
    return oss.str();
  }

  static std::string to_mermaid(const Graph& g) {
    return to_mermaid(g, Options{});
  }
  static std::string to_mermaid(const Graph& g, const Options& opt) {
    std::ostringstream oss;
    oss << "flowchart " << (opt.mermaid_lr ? "LR" : "TD") << "\n";

    for (NodeId id = 0; id < g.node_count(); ++id) {
      const auto& n = g.node(id);
      if (!n)
        continue;

      std::string label = node_label_(n, id, opt);
      if (opt.show_ports) {
        label += "\n" + ("in: " + join_ports_(n->input_ports(), opt.max_ports));
        label += "\n" + ("out: " + join_ports_(n->output_ports(), opt.max_ports));
      }

      oss << "  " << mermaid_node_id_(id, opt) << "[\"" << mermaid_escape_(label) << "\"]\n";
    }

    for (const auto& e : g.edges()) {
      const std::string label = g.port_name(e.from_port) + "->" + g.port_name(e.to_port);
      oss << "  " << mermaid_node_id_(e.from, opt) << " -->|" << mermaid_escape_(label) << "| "
          << mermaid_node_id_(e.to, opt) << "\n";
    }

    return oss.str();
  }

private:
  static const char* backend_name_(Backend b) {
    switch (b) {
    case Backend::Pipeline:
      return "pipeline";
    case Backend::Stage:
      return "stage";
    }
    return "unknown";
  }

  static std::string node_label_(const std::shared_ptr<Node>& n, std::size_t id,
                                 const Options& opt) {
    std::ostringstream oss;
    if (opt.show_index)
      oss << id << ") ";
    if (opt.show_kind)
      oss << n->kind();
    if (opt.show_backend)
      oss << " [" << backend_name_(n->backend()) << "]";
    if (opt.show_user_label) {
      const std::string label = n->user_label();
      if (!label.empty())
        oss << " '" << truncate_(label, opt.max_label_chars) << "'";
    }
    return oss.str();
  }

  static std::string join_ports_(const std::vector<PortDesc>& ports, std::size_t max_ports) {
    if (ports.empty())
      return "<none>";
    std::ostringstream oss;
    std::size_t count = 0;
    for (const auto& p : ports) {
      if (count++ > 0)
        oss << ", ";
      oss << p.name;
      if (p.optional)
        oss << "?";
      if (count >= max_ports) {
        if (ports.size() > max_ports)
          oss << ", ...";
        break;
      }
    }
    return oss.str();
  }

  static std::string truncate_(const std::string& s, std::size_t max_len) {
    if (s.size() <= max_len)
      return s;
    if (max_len < 3)
      return s.substr(0, max_len);
    return s.substr(0, max_len - 3) + "...";
  }

  static std::string dot_escape_(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if (c == '"')
        out += "\\\"";
      else if (c == '\\')
        out += "\\\\";
      else
        out += c;
    }
    return out;
  }

  static std::string mermaid_escape_(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if (c == '"')
        out += "\\\"";
      else
        out += c;
    }
    return out;
  }

  static std::string dot_id_(const std::string& s) {
    if (s.empty())
      return "graph";
    return s;
  }

  static std::string dot_node_id_(std::size_t id) {
    return "n" + std::to_string(id);
  }

  static std::string mermaid_node_id_(std::size_t id, const Options& opt) {
    return opt.mermaid_id_prefix + std::to_string(id);
  }
};

} // namespace simaai::neat::graph
