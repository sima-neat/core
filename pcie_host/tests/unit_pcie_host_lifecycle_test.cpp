#include "simaai/neat/pcie/Model.h"

#include "model_archive_fixture_utils.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace pcie = simaai::neat::pcie;

namespace {

sima_test::ModelArchiveFixture make_lifecycle_model() {
  return sima_test::make_model_archive_fixture("pcie_host_lifecycle",
                                               {{"etc/model_mpk.json", R"json({
  "name": "pcie_host_lifecycle",
  "model_sdk_version": "2.0.0",
  "input_nodes": [{"name":"input","type":"buffer","size":1}],
  "plugins": [{
    "name": "MLA_0",
    "sequence": 1,
    "processor": "MLA",
    "config_params": {
      "desired_batch_size":1,
      "actual_batch_size":1,
      "input_shapes":[[1,1,1]],
      "output_shapes":[[1,1,1]],
      "input_dtype":"INT8",
      "output_dtype":"INT8"
    },
    "input_nodes": [{"name":"input","size":1}],
    "output_nodes": [{"name":"output","type":"buffer","size":1}],
    "type": "sgpProcess",
    "resources": {"executable":"placeholder.elf"}
  }]
})json"},
                                                {"etc/pipeline_sequence.json", R"json({
  "pipelines": [{"sequence": [{
    "sequence_id": 1,
    "name": "MLA_0",
    "pluginId": "processmla",
    "configPath": "0_process_mla.json",
    "processor": "MLA",
    "kernel": "infer",
    "input": "input"
  }]}]
})json"},
                                                {"etc/0_process_mla.json", R"json({
  "node_name":"MLA_0",
  "input_buffers":[{"name":"input"}],
  "data_type":["INT8"],
  "output_width":[1],
  "output_height":[1],
  "output_depth":[1]
})json"}});
}

} // namespace

int main() {
  try {
    for (const std::string invalid_user : {"", "-invalid", "user@host", "user name"}) {
      pcie::ConnectionOptions connection;
      connection.user = invalid_user;
      bool threw = false;
      try {
        (void)pcie::Model("unused-model.tar.gz", {}, connection);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) {
        throw std::runtime_error("invalid PCIe SSH user must be rejected");
      }
    }

    for (const std::string invalid_host : {"-invalid", "user@host", "host name"}) {
      pcie::ConnectionOptions connection;
      connection.card_host = invalid_host;
      bool threw = false;
      try {
        (void)pcie::Model("unused-model.tar.gz", {}, connection);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) {
        throw std::runtime_error("invalid PCIe card host must be rejected");
      }
    }

    {
      pcie::ConnectionOptions connection;
      connection.card_id = -1;
      bool threw = false;
      try {
        (void)pcie::Model("unused-model.tar.gz", {}, connection);
      } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()) == "card_id must be non-negative";
      }
      if (!threw) {
        throw std::runtime_error("negative PCIe card ID must be rejected");
      }
    }

    const auto model_fixture = make_lifecycle_model();
    pcie::Model model(model_fixture.tar_path);
    if (model.running()) {
      throw std::runtime_error("pcie::Model should not be running after construction");
    }
    if (model.info().inputs.empty()) {
      throw std::runtime_error("pcie::Model should load metadata during construction");
    }

    const std::string expected = "PCIe model is not built; call model.build() before run/push/pull";
    bool threw = false;
    try {
      (void)model.push(pcie::Tensor{});
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == expected;
    }
    if (!threw) {
      throw std::runtime_error("push before build must throw the expected lifecycle error");
    }
    threw = false;
    try {
      (void)model.pull(0);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == expected;
    }
    if (!threw) {
      throw std::runtime_error("pull before build must throw the expected lifecycle error");
    }
    model.close();
    model.close();
    std::cout << "[PASS] lifecycle guards\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
