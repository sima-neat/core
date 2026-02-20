/**
 * @file
 * @ingroup mpk
 * @brief Sequence adapter helpers for pre/infer/post MPK stage partitions.
 */
#pragma once

#include "mpk/PipelineSequence.h"

#include <string>
#include <vector>

namespace simaai::neat::mpk {

struct MpKPipelineAdapterOptions {
  bool include_pre = true;
  bool include_infer = true;
  bool include_post = true;
  bool mla_only = false;
};

class MpKPipelineAdapter {
public:
  static std::vector<SequenceEntry> adapt(const std::vector<SequenceEntry>& seq,
                                          const MpKPipelineAdapterOptions& opt = {});

  static std::vector<SequenceEntry> adapt(const SequenceSplit& split,
                                          const MpKPipelineAdapterOptions& opt = {});

  static std::vector<std::string> stage_names(const std::vector<SequenceEntry>& seq);
};

} // namespace simaai::neat::mpk
