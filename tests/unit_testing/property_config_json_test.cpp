#include "builder/ConfigJsonWire.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <random>
#include <string>

RUN_TEST("property_config_json_test", ([] {
           using nlohmann::json;
           using simaai::neat::set_input_buffer_name_if_exists;
           using simaai::neat::set_input_buffer_names;

           std::mt19937 rng(424242u);
           std::uniform_int_distribution<int> coin(0, 1);
           std::uniform_int_distribution<int> count_dist(0, 3);

           for (int iter = 0; iter < 250; ++iter) {
             json j = json::object();

             if (coin(rng)) {
               json in = json::array();
               const int n = count_dist(rng);
               for (int i = 0; i < n; ++i) {
                 if (coin(rng)) {
                   in.push_back(json::object({{"name", "old_" + std::to_string(i)}}));
                 } else {
                   in.push_back(json::object());
                 }
               }
               j["input_buffers"] = in;
             }

             if (coin(rng)) {
               json input = json::array();
               const int n = count_dist(rng);
               for (int i = 0; i < n; ++i) {
                 input.push_back(json::object({{"name", "legacy_" + std::to_string(i)}}));
               }
               j["buffers"] = json::object({{"input", input}});
             }

             const json original = j;
             const std::string name = "src_" + std::to_string(iter);

             const bool changed = set_input_buffer_names(j, name);
             require(changed, "set_input_buffer_names should report changed for non-empty name");

             require(j.contains("input_buffers") && j["input_buffers"].is_array() &&
                         !j["input_buffers"].empty(),
                     "set_input_buffer_names should materialize input_buffers array");
             for (const auto& entry : j["input_buffers"]) {
               require(entry.is_object(), "input_buffers entries must be objects after rewrite");
               require(entry.contains("name") && entry["name"].is_string(),
                       "input_buffers entries must include string name after rewrite");
               require(entry["name"].get<std::string>() == name,
                       "input_buffers entry name mismatch after rewrite");
             }

             require(j.contains("buffers") && j["buffers"].is_object() &&
                         j["buffers"].contains("input") && j["buffers"]["input"].is_array() &&
                         !j["buffers"]["input"].empty(),
                     "set_input_buffer_names should materialize buffers.input array");
             for (const auto& entry : j["buffers"]["input"]) {
               require(entry.is_object(), "buffers.input entries must be objects after rewrite");
               require(entry.contains("name") && entry["name"].is_string(),
                       "buffers.input entries must include string name after rewrite");
               require(entry["name"].get<std::string>() == name,
                       "buffers.input entry name mismatch after rewrite");
             }

             const json once = j;
             (void)set_input_buffer_names(j, name);
             require(j == once, "set_input_buffer_names should be idempotent for same name");

             json maybe = original;
             const bool changed_if_exists = set_input_buffer_name_if_exists(maybe, name);
             const bool had_existing_input_array = original.contains("input_buffers") &&
                                                   original["input_buffers"].is_array() &&
                                                   !original["input_buffers"].empty();
             if (!had_existing_input_array) {
               require(
                   !changed_if_exists,
                   "set_input_buffer_name_if_exists should not create missing input_buffers array");
             }
           }
         }));
