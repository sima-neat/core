#include "model_archive_fixture_utils.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/groups/internal/VideoSenderRawIngress.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <bit>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

std::string tmp_json_path(const std::string& name) {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "sima_neat_graph_io_contract";
  std::error_code ec;
  fs::create_directories(dir, ec);
  return (dir / name).string();
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open temp file for write: " + path);
  }
  out << text;
}

std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open temp file for read: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string io_case(const char* id, const std::string& msg) {
  return std::string("SESSION_IO_CASE=") + id + " " + msg;
}

} // namespace

RUN_TEST(
    "unit_graph_io_contract_test", ([] {
      using namespace simaai::neat;

      Graph graph;
      graph.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
      graph.custom("identity name=io_quote_test silent=true");
      graph.custom("appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

      const std::string save_path = tmp_json_path("graph_io_roundtrip.json");
      graph.save(save_path);
      Graph loaded = Graph::load(save_path);

      const std::string original = graph.describe_backend(false);
      const std::string restored = loaded.describe_backend(false);
      require(!original.empty(),
              io_case("roundtrip_non_empty", "saved pipeline should be non-empty"));
      require(original == restored, io_case("roundtrip_match", "save/load backend mismatch"));

      {
        Graph strict_graph;
        strict_graph.add(nodes::groups::internal::VideoSenderRawIngress(
            {.width = 64, .height = 64, .fps = 30, .require_direct = true}));
        const std::string strict_path = tmp_json_path("graph_io_strict_ingress.json");
        strict_graph.save(strict_path);
        const std::string strict_roundtrip_path = tmp_json_path("graph_io_strict_ingress_rt.json");
        Graph::load(strict_path).save(strict_roundtrip_path);
        require_contains(
            read_text(strict_roundtrip_path), "\"require_direct\":true",
            io_case("strict_ingress_roundtrip", "strict encoder ingress must survive save/load"));
      }

      Graph input_graph;
      InputOptions input_opt;
      input_opt.payload_type = PayloadType::Image;
      input_opt.format = FormatTag::RGB;
      input_opt.width = 16;
      input_opt.height = 16;
      input_opt.memory_policy = InputMemoryPolicy::Ev74;
      const auto input_node = nodes::Input(input_opt);
      const auto output_a = nodes::Output("out_a");
      const auto output_b = nodes::Output("out_b");
      input_graph.connect(input_node, output_a);
      input_graph.connect(input_node, output_b);

      const std::string memory_policy_path = tmp_json_path("graph_io_memory_policy.json");
      input_graph.save(memory_policy_path);
      const std::string saved_memory_policy_json = read_text(memory_policy_path);
      require_contains(saved_memory_policy_json, "\"memory_policy\":1",
                       io_case("memory_policy_saved", "input memory policy should be serialized"));
      require(saved_memory_policy_json.find("\"use_simaai_pool\"") == std::string::npos,
              io_case("deprecated_simaai_pool_omitted",
                      "deprecated input pool flag should not be serialized"));

      const Graph loaded_input = Graph::load(memory_policy_path);
      const std::string memory_policy_roundtrip_path =
          tmp_json_path("graph_io_memory_policy_roundtrip.json");
      loaded_input.save(memory_policy_roundtrip_path);
      require_contains(
          read_text(memory_policy_roundtrip_path), "\"memory_policy\":1",
          io_case("memory_policy_roundtrip", "input memory policy should survive save/load"));

      std::string legacy_memory_policy_json = saved_memory_policy_json;
      const std::string memory_policy_key = "\"memory_policy\":1";
      const std::size_t memory_policy_pos = legacy_memory_policy_json.find(memory_policy_key);
      require(memory_policy_pos != std::string::npos,
              io_case("legacy_simaai_pool_fixture", "memory policy fixture key missing"));
      legacy_memory_policy_json.replace(memory_policy_pos, memory_policy_key.size(),
                                        "\"use_simaai_pool\":false");
      const std::string legacy_memory_policy_path =
          tmp_json_path("graph_io_legacy_simaai_pool.json");
      write_text(legacy_memory_policy_path, legacy_memory_policy_json);
      const Graph loaded_legacy_input = Graph::load(legacy_memory_policy_path);
      const std::string legacy_roundtrip_path =
          tmp_json_path("graph_io_legacy_simaai_pool_roundtrip.json");
      loaded_legacy_input.save(legacy_roundtrip_path);
      const std::string legacy_roundtrip_json = read_text(legacy_roundtrip_path);
      require(legacy_roundtrip_json.find("\"preprocess_meta\"") == std::string::npos,
              "an absent input template must stay absent");
      require_contains(legacy_roundtrip_json, "\"memory_policy\":3",
                       io_case("legacy_simaai_pool_memory_policy",
                               "legacy pool flag should convert to SystemMemory policy"));
      require(legacy_roundtrip_json.find("\"use_simaai_pool\"") == std::string::npos,
              io_case("legacy_simaai_pool_not_reserialized",
                      "deprecated input pool flag should not be reserialized"));

      GraphOptions naming;
      naming.element_name_prefix = "camera.";
      naming.element_name_suffix = "2.view";
      Graph stream_source("stream_source", naming);
      InputOptions stream_input;
      stream_input.max_bytes = (std::uint64_t{1} << 53) + 1;
      stream_input.preprocess_meta = PreprocessMetaTemplate{
          .enabled = true,
          .target_width = 320,
          .target_height = 240,
          .resize_mode = "fit",
          .axis_perm = {2, 0, 1},
          .roi_list_enabled = true,
          .rois = {{.batch_index = 1, .x = -4, .y = 3, .width = 32, .height = 24}},
          .roi_input_batch_size = 2,
          .roi_source_width = 640,
          .roi_source_height = 480,
          .roi_source_stride_bytes = 1920,
          .roi_pad_value = 114};
      stream_source.add(nodes::Input("image", stream_input));
      stream_source.custom("identity name=step");
      stream_source.custom("identity name=camera_step");
      stream_source.add(nodes::Output("frame", OutputOptions::EveryFrame(7)));
      const auto original_backend = stream_source.describe_backend(false);
      require_contains(original_backend, "name=camera_step2_view", "raw element name");
      require_contains(original_backend, "name=camera_camera_step2_view", "distinct prefixed name");
      const auto linear_path = tmp_json_path("linear_endpoints.json");
      stream_source.save(linear_path);
      auto saved_input = [&] {
        return nlohmann::json::parse(read_text(linear_path)).at("nodes").at(0).at("input_options");
      };
      const auto original_input = saved_input();
      for (const char* key : {"roi_source_width", "enabled", "axis_perm", "rois"}) {
        auto invalid = nlohmann::json::parse(read_text(linear_path));
        invalid["nodes"][0]["input_options"]["preprocess_meta"][key] = "invalid";
        const auto path = tmp_json_path("invalid_preproc_geometry.json");
        write_text(path, invalid.dump());
        require_neat_error([&] { (void)Graph::load(path); }, error_codes::kIoParse, key);
      }
      auto invalid_roi = nlohmann::json::parse(read_text(linear_path));
      invalid_roi["nodes"][0]["input_options"]["preprocess_meta"]["rois"][0]["width"] = "wide";
      const auto invalid_path = tmp_json_path("invalid_roi_width.json");
      write_text(invalid_path, invalid_roi.dump());
      require_neat_error([&] { (void)Graph::load(invalid_path); }, error_codes::kIoParse,
                         "rois[0].width");
      Graph dormant_graph("dormant_rois");
      InputOptions dormant_input;
      dormant_input.preprocess_meta =
          PreprocessMetaTemplate{.enabled = false, .roi_list_enabled = true, .rois = {{}}};
      dormant_graph.add(nodes::Input("image", dormant_input));
      dormant_graph.add(nodes::Output("frame"));
      const auto dormant_path = tmp_json_path("disabled_preprocessing_rois.json");
      dormant_graph.save(dormant_path);
      auto saved_dormant = nlohmann::json::parse(read_text(dormant_path));
      Graph::load(dormant_path).save(dormant_path);
      require(nlohmann::json::parse(read_text(dormant_path)) == saved_dormant,
              "disabled preprocessing must preserve a dormant default ROI");
      saved_dormant["nodes"][0]["input_options"]["preprocess_meta"]["enabled"] = true;
      write_text(dormant_path, saved_dormant.dump());
      require_neat_error([&] { (void)Graph::load(dormant_path); }, error_codes::kIoParse,
                         "rois[0].width");

      const auto& meta = original_input.at("preprocess_meta");
      require(original_input.at("max_bytes") == stream_input.max_bytes &&
                  meta.at("enabled") == true && meta.at("target_width") == 320 &&
                  meta.at("axis_perm") == std::vector<int>{2, 0, 1} &&
                  meta.at("rois").at(0).at("x") == -4 && meta.at("roi_pad_value") == 114,
              "input queue size and preprocessing template should be serialized");
      auto loaded_source = Graph::load(linear_path);
      require(loaded_source.inputs() == stream_source.inputs() &&
                  loaded_source.outputs() == stream_source.outputs(),
              "linear graph round trip must preserve public endpoints");
      require(loaded_source.describe_backend(false) == original_backend,
              "linear graph round trip must preserve element names and endpoint options");
      loaded_source.save(linear_path);
      require(saved_input() == original_input,
              "input options must survive save/load with full integer precision");
      require(Graph::load(linear_path).describe_backend(false) == original_backend,
              "repeated graph round trips must preserve element names");
      Graph stream_sink("stream_sink");
      stream_sink.add(nodes::Input("frame"));
      stream_sink.add(nodes::Output("classes"));

      Graph stream_app("graph_io_default_link_stream_id");
      GraphLinkOptions stream_link;
      stream_link.stream_id = "persisted-stream-0";
      stream_app.connect(stream_source, stream_sink, stream_link);

      const std::string stream_link_path = tmp_json_path("graph_io_default_link_stream_id.json");
      stream_app.save(stream_link_path);
      const std::string stream_link_json = read_text(stream_link_path);
      require_contains(
          stream_link_json, "\"link_stream_id\":\"persisted-stream-0\"",
          io_case("default_link_stream_id_saved", "default-link stream id should be serialized"));
      require(stream_link_json.find("\"link_policy\"") == std::string::npos,
              io_case("default_link_policy_omitted",
                      "default-link stream id should not require a non-default policy"));

      const Graph loaded_stream_app = Graph::load(stream_link_path);
      const std::string stream_link_roundtrip_path =
          tmp_json_path("graph_io_default_link_stream_id_roundtrip.json");
      loaded_stream_app.save(stream_link_roundtrip_path);
      const std::string stream_link_roundtrip_json = read_text(stream_link_roundtrip_path);
      require_contains(stream_link_roundtrip_json, "\"link_stream_id\":\"persisted-stream-0\"",
                       io_case("default_link_stream_id_roundtrip",
                               "default-link stream id should survive Graph::load/save"));
      require(stream_link_roundtrip_json.find("\"link_policy\"") == std::string::npos,
              io_case("default_link_policy_roundtrip_omitted",
                      "default-link stream id roundtrip should keep default policy implicit"));

      const auto model_fixture =
          sima_test::make_model_archive_fixture("graph_io_superpoint_options",
                                                {{"etc/model_mpk.json",
                                                  R"json({
  "name": "graph_io_superpoint_options",
  "model_sdk_version": "2.0.0",
  "input_nodes": [{"name":"decoder","type":"buffer","size":1}],
  "plugins": [{
    "name": "MLA_0",
    "sequence": 1,
    "processor": "MLA",
    "config_params": {
      "desired_batch_size": 1,
      "actual_batch_size": 1,
      "number_of_quads_to_user": 1,
      "input_types": [{"scalar": "int8", "shape": [1, 1, 1]}],
      "output_types": [{"scalar": "int8", "shape": [1, 1, 1]}]
    },
    "input_nodes": [{"name":"decoder","size":1}],
    "output_nodes": [{"name":"MLA_0","type":"buffer","size":1}],
    "type": "sgpProcess",
    "resources": {"executable":"placeholder.elf"}
  }]
})json"},
                                                 {"etc/pipeline_sequence.json",
                                                  R"json({
  "pipelines": [{"sequence": [{
    "sequence_id": 1,
    "name": "MLA_0",
    "pluginId": "processmla",
    "configPath": "0_process_mla.json",
    "processor": "MLA",
    "kernel": "infer",
    "input": "decoder"
  }]}]
})json"},
                                                 {"etc/0_process_mla.json",
                                                  R"json({
  "node_name":"MLA_0",
  "input_buffers":[{"name":"decoder"}],
  "data_type":["INT8"],
  "output_width":[1],
  "output_height":[1],
  "output_depth":[1]
})json"}},
                                                false);
      // Loading the saved model fragment plans MLA; its topology is not executed here.
      const auto share = std::filesystem::path(model_fixture.root_dir) / "share";
      std::filesystem::create_directories(share);
      sima_test::write_topology_elf(share / "placeholder.elf", "data.ifm.b0", 1U, "data.ofm.b0",
                                    1U);
      const std::string archive =
          "tar -czf " + sima_test::model_archive_shell_quote(model_fixture.tar_path) + " -C " +
          sima_test::model_archive_shell_quote(model_fixture.root_dir) + " .";
      require(std::system(archive.c_str()) == 0,
              io_case("superpoint_fixture_topology", "failed to archive graph topology fixture"));
      Graph superpoint_provenance_graph;
      superpoint_provenance_graph.custom("identity name=superpoint_provenance");
      const std::string superpoint_base_path =
          tmp_json_path("graph_io_superpoint_options_base.json");
      superpoint_provenance_graph.save(superpoint_base_path);
      std::string superpoint_options_json = read_text(superpoint_base_path);
      const std::size_t root_end = superpoint_options_json.rfind('}');
      require(root_end != std::string::npos,
              io_case("superpoint_fixture_root", "saved graph root is missing"));
      const std::string model_fragment =
          ",\n  \"edges\":[],\n  \"model_fragments\":[{\"start\":0,\"end\":1,"
          "\"model_id\":\"superpoint\","
          "\"source_path\":\"" +
          model_fixture.tar_path +
          "\",\"stage_role\":\"route\",\"model_options\":{"
          "\"superpoint\":{\"profile\":2,\"nms_radius\":7,\"border_margin\":3,"
          "\"descriptor_output_dtype\":5,\"output_format\":2},"
          "\"yolox_seg_pose\":{\"pose_classes\":[0,5]}}}]\n";
      superpoint_options_json.insert(root_end, model_fragment);
      const std::string superpoint_options_path = tmp_json_path("graph_io_superpoint_options.json");
      write_text(superpoint_options_path, superpoint_options_json);

      // Serialized class IDs cannot be silently dropped or truncated into another gate.
      for (const std::string bad_gate :
           {"[0,0.5]", "[0,\"five\"]", "[0,-1]", "[0,2147483648]", "5"}) {
        std::string malformed = superpoint_options_json;
        const auto gate = malformed.find("\"pose_classes\":[0,5]");
        require(gate != std::string::npos, "pose class fixture is missing");
        malformed.replace(gate, std::string("\"pose_classes\":[0,5]").size(),
                          "\"pose_classes\":" + bad_gate);
        const auto malformed_path = tmp_json_path("graph_io_bad_pose_classes.json");
        write_text(malformed_path, malformed);
        bool refused = false;
        try {
          (void)Graph::load(malformed_path);
        } catch (const std::exception& error) {
          refused = std::string(error.what()).find("pose_classes") != std::string::npos;
        }
        require(refused, "invalid serialized pose class IDs must fail with a field diagnostic");
      }

      const Graph loaded_superpoint_model_graph = Graph::load(superpoint_options_path);
      const std::string superpoint_options_roundtrip_path =
          tmp_json_path("graph_io_superpoint_options_roundtrip.json");
      loaded_superpoint_model_graph.save(superpoint_options_roundtrip_path);
      const std::string superpoint_options_roundtrip_json =
          read_text(superpoint_options_roundtrip_path);
      require_contains(superpoint_options_roundtrip_json,
                       "\"yolox_seg_pose\":{\"pose_classes\":[0,5]}",
                       "nested YOLOX segmentation/pose settings must survive save/load");
      require_contains(
          superpoint_options_roundtrip_json, "\"superpoint\":{\"profile\":2",
          io_case("superpoint_profile_roundtrip", "SuperPoint profile should survive save/load"));
      require_contains(
          superpoint_options_roundtrip_json, "\"nms_radius\":7",
          io_case("superpoint_nms_roundtrip", "SuperPoint NMS radius should survive save/load"));
      require_contains(superpoint_options_roundtrip_json, "\"border_margin\":3",
                       io_case("superpoint_border_roundtrip",
                               "SuperPoint border margin should survive save/load"));
      require_contains(superpoint_options_roundtrip_json, "\"descriptor_output_dtype\":5",
                       io_case("superpoint_descriptor_dtype_roundtrip",
                               "SuperPoint descriptor dtype should survive save/load"));
      require_contains(superpoint_options_roundtrip_json, "\"output_format\":2",
                       io_case("superpoint_output_format_roundtrip",
                               "SuperPoint output format should survive save/load"));

      // Exercise authored Model options through the actual graph writer and
      // reader. The infer-only provenance does not execute/qualify pixels.
      auto typed_options = nlohmann::json::parse(superpoint_options_json);
      auto& typed_fragment = typed_options["model_fragments"][0];
      typed_fragment["stage_role"] = "infer";
      const float source_factor = 0x1.010102p-8f;
      const float source_step = std::bit_cast<float>(std::uint32_t{0x3c46a60c});
      typed_fragment["model_options"]["preprocess"] = {
          {"input_max_width", 1280},
          {"input_max_height", 720},
          {"resize", {{"interpolation", 4}}},
          {"rescale_factor", source_factor},
          {"quantization_step", source_step},
          {"color_conversion_placement", 1},
          {"floating_conversion", 1},
          {"normalize",
           {{"mean", {0.485f, 0.456f, 0.406f}}, {"stddev", {0.229f, 0.224f, 0.225f}}}}};
      // Zero is a valid enum value, not an omitted selection.
      for (int sampler : {0, 4}) {
        typed_options["model_fragments"][0]["model_options"]["preprocess"]["resize"]
                     ["interpolation"] = sampler;
        const auto typed_path = tmp_json_path("graph_io_typed_resize.json");
        write_text(typed_path, typed_options.dump());
        const auto typed_graph = Graph::load(typed_path);
        const auto typed_roundtrip_path = tmp_json_path("graph_io_typed_resize_roundtrip.json");
        typed_graph.save(typed_roundtrip_path);
        const auto typed_roundtrip = nlohmann::json::parse(read_text(typed_roundtrip_path));
        const auto& restored_preprocess =
            typed_roundtrip["model_fragments"][0]["model_options"]["preprocess"];
        require(restored_preprocess["resize"]["interpolation"] == sampler,
                "typed sampler intent disappeared during graph save/load");
        require(restored_preprocess["rescale_factor"].get<float>() == source_factor &&
                    restored_preprocess["quantization_step"].get<float>() == source_step &&
                    restored_preprocess["normalize"]["mean"][0].get<float>() == 0.485f &&
                    restored_preprocess["normalize"]["stddev"][0].get<float>() == 0.229f,
                "graph save/load changed original FP32 preprocessing coefficients");
        require(restored_preprocess["color_conversion_placement"] == 1 &&
                    restored_preprocess["floating_conversion"] == 1,
                "graph save/load lost preprocessing conversion policies");
      }
      for (const std::string key : {"rescale_factor", "quantization_step"}) {
        for (const auto& invalid : {nlohmann::json(1e300), nlohmann::json("not-a-number")}) {
          auto malformed = typed_options;
          malformed["model_fragments"][0]["model_options"]["preprocess"][key] = invalid;
          const auto path = tmp_json_path("graph_io_bad_source_coefficient.json");
          write_text(path, malformed.dump());
          bool rejected = false;
          try {
            (void)Graph::load(path);
          } catch (const std::exception& error) {
            rejected = std::string(error.what()).find(key) != std::string::npos;
          }
          require(rejected, "unrepresentable/malformed source coefficient must fail before cast");
        }
      }
      for (const std::string key : {"floating_conversion", "color_conversion_placement"}) {
        for (const auto& invalid : {nlohmann::json(1.5), nlohmann::json(999)}) {
          auto malformed = typed_options;
          malformed["model_fragments"][0]["model_options"]["preprocess"][key] = invalid;
          const auto path = tmp_json_path("graph_io_bad_source_policy.json");
          write_text(path, malformed.dump());
          bool rejected = false;
          try {
            (void)Graph::load(path);
          } catch (const std::exception& error) {
            rejected = std::string(error.what()).find(key) != std::string::npos;
          }
          require(rejected, "invalid saved preprocessing policy must fail instead of truncating");
        }
      }
      for (const auto& bad_sampler :
           {nlohmann::json(1.5), nlohmann::json(999), nlohmann::json("BilinearAntialiased")}) {
        auto malformed = typed_options;
        malformed["model_fragments"][0]["model_options"]["preprocess"]["resize"]["interpolation"] =
            bad_sampler;
        const auto path = tmp_json_path("graph_io_bad_typed_resize.json");
        write_text(path, malformed.dump());
        bool rejected = false;
        try {
          (void)Graph::load(path);
        } catch (const std::exception& error) {
          rejected = std::string(error.what()).find("interpolation") != std::string::npos;
        }
        require(rejected, "invalid serialized typed sampler silently changed its meaning");
      }
      // Letterbox placement survives save/load; the default Center stays implicit.
      for (int placement : {0, 1}) {
        auto placed = typed_options;
        auto& resize = placed["model_fragments"][0]["model_options"]["preprocess"]["resize"];
        resize = {{"enable", 1}, {"mode", 1}, {"placement", placement}};
        const auto path = tmp_json_path("graph_io_letterbox_placement.json");
        write_text(path, placed.dump());
        const auto roundtrip_path = tmp_json_path("graph_io_letterbox_placement_roundtrip.json");
        Graph::load(path).save(roundtrip_path);
        const auto restored = nlohmann::json::parse(read_text(roundtrip_path));
        const auto& restored_resize =
            restored["model_fragments"][0]["model_options"]["preprocess"]["resize"];
        require(placement == 0 ? !restored_resize.contains("placement")
                               : restored_resize["placement"] == placement,
                "letterbox placement changed during graph save/load");
      }
      for (const auto& bad_placement :
           {nlohmann::json(1.5), nlohmann::json(2), nlohmann::json("TopLeft")}) {
        auto malformed = typed_options;
        malformed["model_fragments"][0]["model_options"]["preprocess"]["resize"]["placement"] =
            bad_placement;
        const auto path = tmp_json_path("graph_io_bad_letterbox_placement.json");
        write_text(path, malformed.dump());
        bool rejected = false;
        try {
          (void)Graph::load(path);
        } catch (const std::exception& error) {
          rejected = std::string(error.what()).find("placement") != std::string::npos;
        }
        require(rejected, "invalid serialized letterbox placement silently changed its meaning");
      }

      for (const auto shorter : {nlohmann::json(nullptr), nlohmann::json(256)}) {
        auto crop = typed_options;
        auto& resize = crop["model_fragments"][0]["model_options"]["preprocess"]["resize"];
        resize["mode"] = 2;
        resize["shorter_side"] = shorter;
        const auto path = tmp_json_path("graph_io_crop.json");
        write_text(path, crop.dump());
        const auto loaded = Graph::load(path);
        const auto roundtrip = tmp_json_path("graph_io_crop_roundtrip.json");
        loaded.save(roundtrip);
        const auto restored = nlohmann::json::parse(read_text(roundtrip));
        const auto& actual =
            restored["model_fragments"][0]["model_options"]["preprocess"]["resize"];
        require(actual["mode"] == 2 && (shorter.is_null() ? !actual.contains("shorter_side")
                                                          : actual["shorter_side"] == shorter),
                "Crop/shorter_side graph round trip changed optional semantics");
      }
      for (const auto& invalid :
           {nlohmann::json(0), nlohmann::json(-1), nlohmann::json(4097), nlohmann::json(1.5),
            nlohmann::json(true), nlohmann::json("256")}) {
        auto malformed = typed_options;
        auto& resize = malformed["model_fragments"][0]["model_options"]["preprocess"]["resize"];
        resize["mode"] = 2;
        resize["shorter_side"] = invalid;
        const auto path = tmp_json_path("graph_io_bad_crop.json");
        write_text(path, malformed.dump());
        bool rejected = false;
        try {
          (void)Graph::load(path);
        } catch (const std::exception& error) {
          rejected = std::string(error.what()).find("shorter_side") != std::string::npos;
        }
        require(rejected, "invalid shorter_side changed meaning during graph load");
      }
      // A SAM decoder's multimask_output survives save/load; true stays implicit.
      for (const bool multimask : {true, false}) {
        auto sam = typed_options;
        auto& model_options = sam["model_fragments"][0]["model_options"];
        model_options["decode_type"] = static_cast<int>(BoxDecodeType::EdgeTamSeg);
        model_options.erase("multimask_output");
        if (!multimask)
          model_options["multimask_output"] = false;
        const auto path = tmp_json_path("graph_io_multimask_output.json");
        write_text(path, sam.dump());
        const auto roundtrip_path = tmp_json_path("graph_io_multimask_output_roundtrip.json");
        Graph::load(path).save(roundtrip_path);
        const auto restored = nlohmann::json::parse(read_text(roundtrip_path));
        const auto& restored_options = restored["model_fragments"][0]["model_options"];
        require(restored_options["decode_type"] == static_cast<int>(BoxDecodeType::EdgeTamSeg) &&
                    (multimask ? !restored_options.contains("multimask_output")
                               : restored_options["multimask_output"] == false),
                "multimask_output changed during graph save/load");
      }
      {
        auto malformed = typed_options;
        malformed["model_fragments"][0]["model_options"]["multimask_output"] = "single";
        const auto path = tmp_json_path("graph_io_bad_multimask_output.json");
        write_text(path, malformed.dump());
        bool rejected = false;
        try {
          (void)Graph::load(path);
        } catch (const std::exception& error) {
          rejected = std::string(error.what()).find("multimask_output") != std::string::npos;
        }
        require(rejected, "a non-boolean multimask_output must be rejected");
      }

      Graph realtime_app("graph_io_realtime_link_options");
      GraphLinkOptions realtime_link;
      realtime_link.policy = GraphLinkPolicy::RealtimeLatestByStream;
      realtime_link.queue_depth = 4;
      realtime_link.max_inflight_per_stream = 4;
      realtime_link.max_inflight_total = 16;
      realtime_link.stream_id = "persisted-realtime-stream";
      realtime_app.connect(stream_source, stream_sink, realtime_link);

      const std::string realtime_link_path = tmp_json_path("graph_io_realtime_link_options.json");
      realtime_app.save(realtime_link_path);
      const std::string realtime_link_json = read_text(realtime_link_path);
      require_contains(
          realtime_link_json, "\"link_policy\":\"realtime_latest_by_stream\"",
          io_case("realtime_link_policy_saved", "realtime link policy should be serialized"));
      require_contains(realtime_link_json, "\"link_queue_depth\":4",
                       io_case("realtime_link_queue_depth_saved",
                               "realtime link queue depth should be serialized"));
      require_contains(
          realtime_link_json, "\"link_max_inflight_per_stream\":4",
          io_case("realtime_link_max_inflight_saved",
                  "realtime per-stream raw-frame inflight limit should be serialized"));
      require_contains(realtime_link_json, "\"link_max_inflight_total\":16",
                       io_case("realtime_link_max_inflight_total_saved",
                               "realtime total raw-frame inflight limit should be serialized"));

      const Graph loaded_realtime_app = Graph::load(realtime_link_path);
      const std::string realtime_roundtrip_path =
          tmp_json_path("graph_io_realtime_link_options_roundtrip.json");
      loaded_realtime_app.save(realtime_roundtrip_path);
      const std::string realtime_roundtrip_json = read_text(realtime_roundtrip_path);
      require_contains(
          realtime_roundtrip_json, "\"link_max_inflight_per_stream\":4",
          io_case("realtime_link_max_inflight_roundtrip",
                  "realtime per-stream raw-frame inflight limit should survive load/save"));
      require_contains(realtime_roundtrip_json, "\"link_max_inflight_total\":16",
                       io_case("realtime_link_max_inflight_total_roundtrip",
                               "realtime total raw-frame inflight limit should survive load/save"));

      std::string removed_every_frame_json = realtime_link_json;
      const std::string latest_policy = "realtime_latest_by_stream";
      const std::size_t policy_pos = removed_every_frame_json.find(latest_policy);
      require(policy_pos != std::string::npos,
              io_case("removed_every_frame_fixture", "realtime policy fixture key missing"));
      removed_every_frame_json.replace(policy_pos, latest_policy.size(),
                                       "realtime_every_frame_by_stream");
      const std::string removed_every_frame_path =
          tmp_json_path("graph_io_removed_every_frame_policy.json");
      write_text(removed_every_frame_path, removed_every_frame_json);
      require_neat_error([&]() { (void)Graph::load(removed_every_frame_path); },
                         error_codes::kIoParse, "unsupported edge link_policy",
                         "realtime_every_frame_by_stream");

      const std::string missing_file = tmp_json_path("graph_io_missing_input.json");
      {
        std::error_code ec;
        std::filesystem::remove(missing_file, ec);
      }
      require_neat_error([&]() { (void)Graph::load(missing_file); }, error_codes::kIoOpen,
                         "io.open", "failed to open file for reading");

      const std::string tmp_dir =
          (std::filesystem::temp_directory_path() / "sima_neat_graph_io_contract").string();
      {
        std::error_code ec;
        std::filesystem::create_directories(tmp_dir, ec);
      }
      require_neat_error([&]() { graph.save(tmp_dir); }, error_codes::kIoOpen, "io.open",
                         "failed to open file for writing");

      const std::string invalid_root = tmp_json_path("graph_io_invalid_root.json");
      write_text(invalid_root, "[]\n");
      require_neat_error([&]() { (void)Graph::load(invalid_root); }, error_codes::kIoParse,
                         "io.parse", "invalid JSON root");

      const std::string missing_nodes = tmp_json_path("graph_io_missing_nodes.json");
      write_text(missing_nodes, "{\"version\":1}\n");
      require_neat_error([&]() { (void)Graph::load(missing_nodes); }, error_codes::kIoParse,
                         "io.parse", "missing required 'nodes' array");

      const std::string invalid_node_entry = tmp_json_path("graph_io_invalid_node_entry.json");
      write_text(invalid_node_entry, "{\"nodes\":[5]}\n");
      require_neat_error([&]() { (void)Graph::load(invalid_node_entry); }, error_codes::kIoParse,
                         "io.parse", "node entry must be object");

      const std::string missing_fragment = tmp_json_path("graph_io_missing_fragment.json");
      write_text(missing_fragment,
                 "{\"nodes\":[{\"kind\":\"Gst\",\"label\":\"x\",\"elements\":[]}]}\n");
      require_neat_error([&]() { (void)Graph::load(missing_fragment); }, error_codes::kIoParse,
                         "io.parse", "node fragment is empty");

      const std::string malformed_json = tmp_json_path("graph_io_malformed.json");
      write_text(malformed_json, "{\"nodes\":[{\"kind\":\"Gst\" \"fragment\":\"identity\"}]}\n");
      require_neat_error([&]() { (void)Graph::load(malformed_json); }, error_codes::kIoParse,
                         "offset=", "near='");
    }));
