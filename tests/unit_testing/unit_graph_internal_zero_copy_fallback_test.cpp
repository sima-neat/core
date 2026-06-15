#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/internal/InputStream.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

std::shared_ptr<simaai::neat::runtime::RunCore>
make_balanced_zero_copy_core(bool public_output_contract) {
  simaai::neat::RunOptions opt;
  opt.preset = simaai::neat::RunPreset::Balanced;
  opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;

  simaai::neat::InputStreamOptions stream_opt;
  stream_opt.copy_output = false;
  stream_opt.public_output_contract = public_output_contract;

  return simaai::neat::runtime::RunCore::start_single_pipeline(
      simaai::neat::InputStream{}, opt, stream_opt, simaai::neat::RunMode::Async);
}

} // namespace

RUN_TEST("unit_graph_internal_zero_copy_fallback_test", ([] {
           auto public_core = make_balanced_zero_copy_core(true);
           require(public_core->pipeline.zero_copy_fallback_enabled,
                   "public balanced zero-copy output should keep the copy fallback enabled");
           require(!public_core->pipeline.copy_output_latched.load(std::memory_order_relaxed),
                   "public zero-copy output should start unlatched");

           auto graph_internal_core = make_balanced_zero_copy_core(false);
           require(!graph_internal_core->pipeline.zero_copy_fallback_enabled,
                   "graph-internal zero-copy transport must not clone tensors back to CPU under "
                   "queue pressure");
           require(
               !graph_internal_core->pipeline.copy_output_latched.load(std::memory_order_relaxed),
               "graph-internal zero-copy transport should start unlatched");
         }))
