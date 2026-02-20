/**
 * @file
 * @ingroup mpk
 * @brief Helpers for parsing model pipeline_sequence.json.
 */
#pragma once

#include <string>
#include <vector>

namespace simaai::neat::mpk {

struct SequenceEntry {
  int sequence_id = 0;
  std::string name;
  std::string plugin_id;
  std::string config_path;
  std::string processor;
  std::string kernel;
};

struct SequenceSplit {
  std::vector<SequenceEntry> pre;   // user-controlled pre adapters (leading)
  std::vector<SequenceEntry> infer; // model-owned inference block
  std::vector<SequenceEntry> post;  // user-controlled post adapters (trailing)
};

// Parse pipeline_sequence.json (first pipeline) into ordered entries.
std::vector<SequenceEntry> load_pipeline_sequence(const std::string& etc_dir);

// Split sequence into pre/infer/post. Pre removes leading preproc/quanttess.
// Post removes trailing detessdequant/boxdecode. Internal occurrences are preserved in infer.
SequenceSplit split_sequence_for_infer(const std::vector<SequenceEntry>& seq);

bool is_pre_adapter_kernel(const std::string& kernel);
bool is_post_adapter_kernel(const std::string& kernel);

} // namespace simaai::neat::mpk
