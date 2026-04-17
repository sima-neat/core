// tutorial_0023_builder_graph_and_contracts.cpp
// Story: visualize builder graphs and validate contracts before GStreamer.
// What you learn:
// - GraphPrinter emits text, DOT, and Mermaid.
// - ContractRegistry validates structural rules (no sink last, RTSP source, etc.).
// - Session::describe() is a builder-level view.

#include "neat/session.h"
#include "neat/nodes.h"
#include "builder/Graph.h"
#include "builder/GraphPrinter.h"
#include "contracts/Validators.h"

#include <iostream>
#include <vector>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

} // namespace

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << "\n";
  print_common_flags(std::cout);
}

void print_report(const simaai::neat::ValidationReport& rep) {
  std::cout << rep.to_string() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // Build a small graph (Input -> VideoConvert -> Output).
    simaai::neat::Graph g;
    auto n0 = g.add(simaai::neat::nodes::Input());
    auto n1 = g.add(simaai::neat::nodes::VideoConvert());
    auto n2 = g.add(simaai::neat::nodes::Output());
    g.add_chain({n0, n1, n2});

    simaai::neat::GraphPrinter::Options opt;
    opt.show_backend_fragment = true;
    opt.show_element_names = true;

    std::cout << "[GraphPrinter] text\n";
    std::cout << simaai::neat::GraphPrinter::to_text(g.to_node_group_chain(), opt) << "\n\n";

    std::cout << "[GraphPrinter] DOT\n";
    std::cout << simaai::neat::GraphPrinter::to_dot(g, opt) << "\n";

    std::cout << "[GraphPrinter] Mermaid\n";
    std::cout << simaai::neat::GraphPrinter::to_mermaid(g, opt) << "\n";

    // Contract validation examples.
    simaai::neat::ContractRegistry reg = simaai::neat::validators::DefaultRegistry();

    std::vector<std::shared_ptr<simaai::neat::Node>> bad_nodes;
    bad_nodes.push_back(simaai::neat::nodes::Input());
    bad_nodes.push_back(simaai::neat::nodes::VideoConvert());
    simaai::neat::NodeGroup bad_group(std::move(bad_nodes));
    simaai::neat::ValidationContext ctx;
    ctx.mode = simaai::neat::ValidationContext::Mode::Run;

    std::cout << "[Contracts] bad group (missing Output)\n";
    print_report(reg.validate(bad_group, ctx));

    std::vector<std::shared_ptr<simaai::neat::Node>> good_nodes;
    good_nodes.push_back(simaai::neat::nodes::Input());
    good_nodes.push_back(simaai::neat::nodes::VideoConvert());
    good_nodes.push_back(simaai::neat::nodes::Output());
    simaai::neat::NodeGroup good_group(std::move(good_nodes));
    std::cout << "[Contracts] good group\n";
    print_report(reg.validate(good_group, ctx));

    // Session::describe() is a builder-level view (no GStreamer running).
    simaai::neat::Session p;
    p.add(simaai::neat::nodes::Input());
    p.add(simaai::neat::nodes::VideoConvert());
    p.add(simaai::neat::nodes::Output());

    std::cout << "[Session::describe]\n";
    std::cout << p.describe(opt) << "\n";

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
    }

    std::cout << "[OK] tutorial_0023 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
