#define SIMA_NEAT_INTERNAL 1
#include "model/internal/ModelPack.h"
#include "pipeline/internal/MemoryBackendPolicy.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "test_main.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_monolithic_topology_elf(const std::filesystem::path& path) {
  struct Elf64Header {
    std::uint8_t ident[16]{};
    std::uint16_t type = 0;
    std::uint16_t machine = 0;
    std::uint32_t version = 0;
    std::uint64_t entry = 0;
    std::uint64_t program_header_offset = 0;
    std::uint64_t section_header_offset = 0;
    std::uint32_t flags = 0;
    std::uint16_t header_size = 0;
    std::uint16_t program_header_size = 0;
    std::uint16_t program_header_count = 0;
    std::uint16_t section_header_size = 0;
    std::uint16_t section_header_count = 0;
    std::uint16_t section_name_table_index = 0;
  };
  struct Elf64SectionHeader {
    std::uint32_t name = 0;
    std::uint32_t type = 0;
    std::uint64_t flags = 0;
    std::uint64_t address = 0;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint32_t link = 0;
    std::uint32_t info = 0;
    std::uint64_t alignment = 0;
    std::uint64_t entry_size = 0;
  };
  static_assert(sizeof(Elf64Header) == 64U);
  static_assert(sizeof(Elf64SectionHeader) == 64U);

  constexpr char names[] = "\0.shstrtab\0data.ifm.b0\0data.ofm.b0\0";
  const std::uint64_t names_offset = sizeof(Elf64Header);
  const std::uint64_t sections_offset = (names_offset + sizeof(names) + 7U) & ~std::uint64_t{7U};

  Elf64Header header;
  header.ident[0] = 0x7fU;
  header.ident[1] = 'E';
  header.ident[2] = 'L';
  header.ident[3] = 'F';
  header.ident[4] = 2U;
  header.ident[5] = 1U;
  header.ident[6] = 1U;
  header.type = 1U;
  header.machine = 183U;
  header.version = 1U;
  header.section_header_offset = sections_offset;
  header.header_size = sizeof(Elf64Header);
  header.section_header_size = sizeof(Elf64SectionHeader);
  header.section_header_count = 4U;
  header.section_name_table_index = 1U;

  std::vector<Elf64SectionHeader> sections(4U);
  sections[1].name = 1U;
  sections[1].type = 3U;
  sections[1].offset = names_offset;
  sections[1].size = sizeof(names);
  sections[1].alignment = 1U;
  sections[2].name = 11U;
  sections[2].type = 1U;
  sections[3].name = 23U;
  sections[3].type = 1U;

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.is_open(), "failed to create synthetic MLA ELF");
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(names, sizeof(names));
  const auto written = names_offset + sizeof(names);
  std::vector<char> padding(static_cast<std::size_t>(sections_offset - written), 0);
  output.write(padding.data(), static_cast<std::streamsize>(padding.size()));
  output.write(reinterpret_cast<const char*>(sections.data()),
               static_cast<std::streamsize>(sections.size() * sizeof(sections.front())));
  require(output.good(), "failed to write synthetic MLA ELF");
}

void write_manifest(const std::filesystem::path& path) {
  std::ofstream output(path);
  require(output.is_open(), "failed to create synthetic multi-MLA MPK");
  output << R"json({
    "name":"modelpack-two-mla","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":64}],
    "plugins":[
      {"name":"MLA_encoder","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":64}],
       "output_nodes":[{"name":"encoded","size":64}],
       "resources":{"executable":"encoder.so"}},
      {"name":"MLA_decoder","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"encoded","size":64}],
       "output_nodes":[{"name":"decoded","size":64}],
       "resources":{"executable":"decoder.elf"}},
      {"name":"publish","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"decoded","size":64}],
       "output_nodes":[{"name":"output","size":64}]}
    ]
  })json";
  require(output.good(), "failed to write synthetic multi-MLA MPK");
}

} // namespace

RUN_TEST("unit_modelpack_multi_mla_plan_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::internal::ExecutionStageKind;
           using simaai::neat::internal::ModelPack;
           using simaai::neat::internal::ModelStage;
           using simaai::neat::internal::PipelineType;
           using simaai::neat::pipeline_internal::MemoryBackendPolicy;
           using simaai::neat::pipeline_internal::sima::FrameArenaRole;

           require(::setenv("SIMA_NEAT_MEMORY_BACKEND", "dmabuf-plan", 1) == 0,
                   "failed to select strict DMA-BUF backend");
           const auto root = fs::temp_directory_path() / "neat-modelpack-two-mla-unit";
           std::error_code ec;
           fs::remove_all(root, ec);
           fs::create_directories(root / "etc", ec);
           fs::create_directories(root / "lib", ec);
           fs::create_directories(root / "share", ec);
           require(!ec, "failed to create synthetic package layout");
           write_manifest(root / "etc" / "modelpack_two_mla_mpk.json");
           write_monolithic_topology_elf(root / "share" / "encoder.so");
           write_monolithic_topology_elf(root / "share" / "decoder.elf");

           ModelPack model(root.string());
           require(model.memory_backend_decision().backend == MemoryBackendPolicy::DmaBufPlan &&
                       model.memory_backend_decision().admission.eligible(),
                   "synthetic package did not pass strict multi-stage admission");
           require(model.mpk_contract().has_value(), "synthetic package lost its MPK contract");
           const auto& mpk = *model.mpk_contract();
           const auto* first_mla =
               simaai::neat::pipeline_internal::sima::get_first_mla_stage_io_contract(mpk);
           const auto* last_mla =
               simaai::neat::pipeline_internal::sima::get_last_mla_stage_io_contract(mpk);
           require(first_mla && first_mla->name == "MLA_encoder" && last_mla &&
                       last_mla->name == "MLA_decoder" &&
                       simaai::neat::pipeline_internal::sima::get_mla_stage_io_contract(mpk) ==
                           nullptr,
                   "MLA boundary queries must be explicit and the singular query fail ambiguous");
           const auto& route_graph = model.route_graph();
           require(route_graph.mla_plugin_index >= 0 && route_graph.last_mla_plugin_index >= 0 &&
                       route_graph.mla_plugin_index != route_graph.last_mla_plugin_index,
                   "route graph must retain distinct first and terminal MLA boundaries");
           const auto plan = model.execution_plan();
           require(plan.pre.empty() && plan.post.empty() && plan.infer.size() == 2U,
                   "multi-MLA execution plan must contain exactly two inference stages");
           require(plan.infer[0].kind == ExecutionStageKind::Mla &&
                       plan.infer[0].stage_name == "MLA_encoder" &&
                       plan.infer[1].kind == ExecutionStageKind::Mla &&
                       plan.infer[1].stage_name == "MLA_decoder",
                   "ModelPack lost compiler-authored MLA stage order or identity");

           const auto facts = model.stage_facts_for_model_stage(ModelStage::MlaOnly);
           require(facts.size() == 2U && facts[0].mla_compiled.has_value() &&
                       facts[1].mla_compiled.has_value(),
                   "ModelPack did not render one typed ProcessMLA contract per MLA stage");
           const auto& first = facts[0].mla_compiled->runtime_contract;
           const auto& second = facts[1].mla_compiled->runtime_contract;
           require(first.frame_arena_role == FrameArenaRole::Allocate &&
                       second.frame_arena_role == FrameArenaRole::ReuseInput,
                   "the first MLA must allocate and the later MLA must reuse the graph arena");
           require(first.frame_arena_size_bytes != 0U &&
                       first.frame_arena_size_bytes == second.frame_arena_size_bytes,
                   "both MLA stages must project the same graph-wide arena extent");
           require(first.physical_outputs.size() == 1U && second.physical_inputs.size() == 1U &&
                       first.physical_outputs[0].source_byte_offset ==
                           second.physical_inputs[0].source_byte_offset,
                   "the later MLA IFM must reuse the exact preceding MLA OFM arena region");
           require(facts[0].mla_compiled->payload.model_path.ends_with("/share/encoder.so") &&
                       facts[1].mla_compiled->payload.model_path.ends_with("/share/decoder.elf"),
                   "an MLA .so/ELF must stay bound to its exact MPK stage identity");
           fs::remove_all(root, ec);

           // Optional exact AFE 2.1 package gate for qualification jobs. The
           // fixture remains external because it is a compiler artifact, not
           // test source. This proves ModelPack resolves both MLA ELFs and the
           // A65 shared object before it accepts the strict backend.
           if (const char* real_package = std::getenv("SIMANEAT_AFE21_MULTI_MLA_PACKAGE");
               real_package != nullptr && *real_package != '\0') {
             ModelPack real_model(
                 real_package, "application/vnd.simaai.tensor", "FP32", /*depth=*/1,
                 /*max_width=*/640, /*max_height=*/480, /*max_depth=*/1,
                 /*normalize=*/false, {}, {}, /*preproc_next_cpu=*/{}, PipelineType::QuantTess);
             require(real_model.memory_backend_decision().admission.eligible(),
                     "exact AFE 2.1 multi-MLA/A65 package failed ModelPack admission");
             const auto real_plan = real_model.execution_plan();
             require(real_plan.infer.size() == 2U &&
                         real_plan.infer[0].kind == ExecutionStageKind::Mla &&
                         real_plan.infer[1].kind == ExecutionStageKind::Mla,
                     "exact AFE 2.1 package lost its two compiler-authored MLA stages");

             // Until Phase 5 renders the compiler-authored interstitial EV74
             // DAG, the second packed MLA input must not be projected as if
             // the two MLA stages were adjacent. Keep this as an explicit
             // fail-closed gate rather than admitting a semantically wrong
             // runtime pipeline.
             real_model.set_model_managed_stage_facts(
                 /*processcvu_preproc_single_output_handoff=*/true, std::nullopt, {});
             bool rejected_unscheduled_middle = false;
             try {
               (void)real_model.stage_facts_for_model_stage(ModelStage::MlaOnly);
             } catch (const std::runtime_error& error) {
               rejected_unscheduled_middle =
                   std::string(error.what()).find("packed IFM child") != std::string::npos &&
                   std::string(error.what()).find("no exact upstream TensorBuffer view") !=
                       std::string::npos;
             }
             require(rejected_unscheduled_middle,
                     "unscheduled interstitial EV74 materializations did not fail closed");
           }
         }));
