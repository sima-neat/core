#include "mpk/MpKPipelineAdapter.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>
#include <vector>

RUN_TEST("unit_mpk_pipeline_adapter_test", ([] {
           using namespace simaai::neat::mpk;

           SequenceSplit split;
           split.pre = {
               SequenceEntry{.sequence_id = 1,
                             .name = "preproc_0",
                             .plugin_id = "processcvu",
                             .config_path = "0_preproc.json",
                             .processor = "CVU",
                             .kernel = "preproc"},
           };
           split.infer = {
               SequenceEntry{.sequence_id = 2,
                             .name = "infer_cpu",
                             .plugin_id = "processcpu",
                             .config_path = "1_infer_cpu.json",
                             .processor = "CPU",
                             .kernel = "infer"},
               SequenceEntry{.sequence_id = 3,
                             .name = "infer_mla",
                             .plugin_id = "processmla",
                             .config_path = "2_infer_mla.json",
                             .processor = "MLA",
                             .kernel = "infer"},
           };
           split.post = {
               SequenceEntry{.sequence_id = 4,
                             .name = "detess_0",
                             .plugin_id = "detessdequant",
                             .config_path = "3_detess.json",
                             .processor = "CVU",
                             .kernel = "detessdequant"},
           };

           {
             const auto all = MpKPipelineAdapter::adapt(split, MpKPipelineAdapterOptions{});
             require(all.size() == 4, "default adapter options should keep all stages");
             const auto names = MpKPipelineAdapter::stage_names(all);
             require(names.size() == 4, "stage_names should mirror adapted sequence length");
             require(names[0] == "preproc_0", "adapted order should preserve pre stage first");
             require(names[1] == "infer_cpu", "adapted order should preserve infer stage order");
             require(names[2] == "infer_mla", "adapted order should preserve infer stage order");
             require(names[3] == "detess_0", "adapted order should preserve post stage last");
           }

           {
             MpKPipelineAdapterOptions opt;
             opt.include_pre = false;
             opt.include_post = false;
             const auto infer_only = MpKPipelineAdapter::adapt(split, opt);
             require(infer_only.size() == 2, "infer-only adaptation should keep only infer stages");
             require(infer_only[0].name == "infer_cpu", "infer-only ordering mismatch");
             require(infer_only[1].name == "infer_mla", "infer-only ordering mismatch");
           }

           {
             MpKPipelineAdapterOptions opt;
             opt.include_pre = false;
             opt.include_post = false;
             opt.mla_only = true;
             const auto mla_only = MpKPipelineAdapter::adapt(split, opt);
             require(mla_only.size() == 1,
                     "mla_only adaptation should filter non-MLA infer stages");
             require(mla_only[0].name == "infer_mla", "mla_only stage name mismatch");
           }

           {
             const std::vector<SequenceEntry> seq = {
                 split.pre.front(),
                 split.infer[0],
                 split.infer[1],
                 split.post.front(),
             };
             MpKPipelineAdapterOptions opt;
             opt.include_post = false;
             const auto no_post = MpKPipelineAdapter::adapt(seq, opt);
             require(no_post.size() == 3,
                     "adapt(sequence) should honor options through split path");
             require(no_post.back().name == "infer_mla",
                     "adapt(sequence) should drop trailing post stage");
           }
         }));
