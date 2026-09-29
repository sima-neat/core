#include "genai/HistoryPaths.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace pgi = simaai::neat::pcie::genai::internal;

namespace {
void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
} // namespace

int main() {
  try {
    const std::map<std::string, std::string> sent = {{"pcie-genai/h1-3-0.jpg", "/home/me/cat.jpg"}};

    // A card path that ends with a sent name becomes the host path.
    require(
        pgi::map_history_image_paths(
            R"([{"role":"system","content":"S"},{"role":"user","content":[{"type":"image","image":"/data/recv/pcie-genai/h1-3-0.jpg"},{"type":"text","text":"what"}]}])",
            sent) ==
            R"([{"role":"system","content":"S"},{"role":"user","content":[{"type":"image","image":"/home/me/cat.jpg"},{"type":"text","text":"what"}]}])",
        "card image path is replaced by the host path");

    // The real card path: PcieFileProvider drops the subfolder, so the pulled
    // image lands directly in the recv root (seen on mla-hl83: /tmp/pcie-recv/<file>).
    require(
        pgi::map_history_image_paths(
            R"([{"role":"user","content":[{"type":"image","image":"/tmp/pcie-recv/h1-3-0.jpg"}]}])",
            sent) == R"([{"role":"user","content":[{"type":"image","image":"/home/me/cat.jpg"}]}])",
        "a card path with only the file name is mapped too");

    // Unknown names, and a suffix match without a '/' boundary, stay as they are.
    const std::string unknown =
        R"([{"role":"user","content":[{"type":"image","image":"/data/recv/pcie-genai/h9-1-0.jpg"}]}])";
    require(pgi::map_history_image_paths(unknown, sent) == unknown, "unknown name is left");
    const std::string no_boundary =
        R"([{"role":"user","content":[{"type":"image","image":"/tmp/pcie-recv/xh1-3-0.jpg"}]}])";
    require(pgi::map_history_image_paths(no_boundary, sent) == no_boundary,
            "a file name that only ends like a sent name is not mapped");

    // Text items with the same string are not touched.
    const std::string text_only =
        R"([{"role":"user","content":[{"type":"text","text":"/data/recv/pcie-genai/h1-3-0.jpg"}]}])";
    require(pgi::map_history_image_paths(text_only, sent) == text_only, "text is not touched");

    // Not JSON, or not an array: returned unchanged, never throws.
    require(pgi::map_history_image_paths("not json", sent) == "not json", "bad JSON unchanged");
    require(pgi::map_history_image_paths(R"({"a":1})", sent) == R"({"a":1})", "object unchanged");

    std::cout << "[PASS] host history image-path mapping\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
