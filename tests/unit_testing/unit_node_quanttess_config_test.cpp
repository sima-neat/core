#include "nodes/sima/QuantTess.h"
#include "test_main.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <string>

namespace {

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_node_quanttess_config_test", ([] {
      using nlohmann::json;

      simaai::neat::QuantTessOptions opt;
      opt.config_json = json{
          {"node_name", "quant0"},
          {"input_buffers",
           json::array({json{{"name", "decoder_a"}}, json{{"name", "decoder_b"}}})},
          {"buffers",
           json{{"input", json::array({json{{"name", "legacy_a"}}, json{{"name", "legacy_b"}}})}}}};
      opt.config_dir = "/tmp";
      opt.keep_config = true;
      opt.element_name = "quant_test";
      opt.num_buffers = 4;

      simaai::neat::QuantTess quant(opt);
      require(quant.kind() == "QuantTess", "QuantTess kind mismatch");

      const std::string fragment = quant.backend_fragment(6);
      require_contains(fragment, "neatprocesscvu name=quant_test",
                       "QuantTess fragment name mismatch");
      require_contains(fragment, "stage-id=quant_test", "QuantTess stage-id mismatch");
      require_contains(fragment, "num-buffers=4", "QuantTess num-buffers mismatch");
      require_contains(fragment, "config=\"", "QuantTess should include config path in fragment");

      const auto names = quant.element_names(6);
      require(names.size() == 1 && names[0] == "quant_test", "QuantTess element name mismatch");

      const json* cfg = quant.config_json();
      require(cfg != nullptr, "QuantTess config_json missing");
      require((*cfg)["node_name"].get<std::string>() == "quant0",
              "QuantTess initial node_name mismatch");
      require(!quant.config_path().empty(), "QuantTess config_path should be materialized");

      require(throws_with(
                  []() {
                    simaai::neat::QuantTessOptions bad;
                    bad.num_buffers_locked = true;
                    bad.num_buffers_model = 4;
                    bad.num_buffers = 2;
                    (void)simaai::neat::QuantTess(bad);
                  },
                  "override is not allowed"),
              "QuantTess should reject locked num_buffers override mismatch");

      require(throws_with(
                  []() {
                    simaai::neat::QuantTessOptions bad;
                    bad.num_buffers_locked = true;
                    bad.num_buffers_model = 5;
                    bad.num_buffers = 5;
                    (void)simaai::neat::QuantTess(bad);
                  },
                  "must be 4 (async) or 1 (sync)"),
              "QuantTess should reject locked num_buffers unsupported values");
    }));
