#include "mpk/MpKPipelineAdapter.h"

namespace simaai::neat::mpk {

std::vector<SequenceEntry> MpKPipelineAdapter::adapt(const std::vector<SequenceEntry>& seq,
                                                     const MpKPipelineAdapterOptions& opt) {
  return adapt(split_sequence_for_infer(seq), opt);
}

std::vector<SequenceEntry> MpKPipelineAdapter::adapt(const SequenceSplit& split,
                                                     const MpKPipelineAdapterOptions& opt) {
  std::vector<SequenceEntry> out;

  if (opt.include_pre) {
    out.insert(out.end(), split.pre.begin(), split.pre.end());
  }

  if (opt.include_infer) {
    for (const auto& entry : split.infer) {
      if (opt.mla_only && entry.processor != "MLA") {
        continue;
      }
      out.push_back(entry);
    }
  }

  if (opt.include_post) {
    out.insert(out.end(), split.post.begin(), split.post.end());
  }

  return out;
}

std::vector<std::string> MpKPipelineAdapter::stage_names(const std::vector<SequenceEntry>& seq) {
  std::vector<std::string> names;
  names.reserve(seq.size());
  for (const auto& entry : seq) {
    names.push_back(entry.name);
  }
  return names;
}

} // namespace simaai::neat::mpk
