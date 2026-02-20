#include "mpk/PipelineSequence.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (!data || size == 0 || size > (1U << 16)) {
    return 0;
  }

  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path base = fs::temp_directory_path(ec) / "sima_neat_fuzz_pipeline_sequence";
  fs::create_directories(base, ec);

  const fs::path seq_path = base / "pipeline_sequence.json";
  {
    std::ofstream out(seq_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      return 0;
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  }

  try {
    (void)simaai::neat::mpk::load_pipeline_sequence(base.string());
  } catch (...) {
    // Rejections are expected for malformed payloads.
  }

  return 0;
}
