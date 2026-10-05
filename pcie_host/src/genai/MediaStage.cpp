#include "genai/MediaStage.h"
#include "Protocol.h"
#include <cstring>
#include <fstream>
#include <sys/stat.h>

namespace simaai::neat::pcie::genai::internal {
namespace {
constexpr std::size_t max_bytes = 256 * 1024 * 1024;
void le(std::ostream& out, uint32_t value, unsigned bytes) {
  for (unsigned i = 0; i < bytes; ++i)
    out.put(static_cast<char>(value >> (8 * i)));
}
} // namespace
MediaStage::MediaStage(const ConnectionOptions& options, const std::string& session,
                       uint64_t request)
    : root_(std::filesystem::absolute(options.media_directory).lexically_normal()),
      serve_(options.media_serve_root) {
  wire::validate_session(session);
  directory_ = root_ / "neat-genai" / session / std::to_string(request);
}
void MediaStage::prepare() {
  if (prepared_)
    return;
  if (std::filesystem::canonical(root_) != root_)
    throw std::runtime_error("Media serve root must be canonical");
  std::filesystem::create_directories(directory_.parent_path());
  if (std::filesystem::canonical(directory_.parent_path()) != directory_.parent_path())
    throw std::runtime_error("Media staging path contains a symlink");
  chmod(directory_.parent_path().c_str(), 0700);
  if (!std::filesystem::create_directory(directory_))
    throw std::runtime_error("Media request directory already exists");
  chmod(directory_.c_str(), 0700);
  prepared_ = true;
}
MediaStage::~MediaStage() {
  if (!prepared_)
    return;
  std::error_code ec;
  std::filesystem::remove_all(directory_, ec);
  std::filesystem::remove(directory_.parent_path(), ec); // only if this session is empty
}
Json MediaStage::descriptor(const std::filesystem::path& path) {
  auto size = std::filesystem::file_size(path);
  if (size > max_bytes || bytes_ > max_bytes - size)
    throw std::length_error("Request media exceeds 256 MiB");
  bytes_ += size;
  return {
      {"root", serve_}, {"name", path.lexically_relative(root_).generic_string()}, {"bytes", size}};
}
Json MediaStage::file(const std::filesystem::path& source) {
  if (!std::filesystem::is_regular_file(source) || std::filesystem::file_size(source) > max_bytes)
    throw std::invalid_argument("Media file missing or too large: " + source.string());
  prepare();
  auto dest = directory_ / (std::to_string(count_++) + ".media");
  std::filesystem::copy_file(source, dest);
  return descriptor(dest);
}
Json MediaStage::tensor(const Tensor& t, bool audio, uint32_t rate) {
  const auto expected = audio ? TensorDType::Float32 : TensorDType::UInt8;
  if (t.dtype != expected || !t.data || t.byte_offset < 0 ||
      (audio ? (t.shape.size() != 1 || !rate) : (t.shape.size() != 3 || t.shape[2] != 3)))
    throw std::invalid_argument("Expected Float32 mono [N] audio or UInt8 RGB [H,W,3] image");
  const std::size_t item = audio ? 4 : 1;
  auto strides =
      t.strides_bytes.empty() ? detail::contiguous_tensor_strides(t.shape, item) : t.strides_bytes;
  auto needed = detail::checked_required_span_bytes(t.shape, strides, item);
  if (static_cast<std::size_t>(t.byte_offset) > t.size_bytes ||
      needed > t.size_bytes - t.byte_offset || needed > max_bytes)
    throw std::invalid_argument("Media tensor storage does not cover its shape/strides");
  if (!audio && t.image_format != PixelFormat::Unknown && t.image_format != PixelFormat::RGB)
    throw std::invalid_argument("GenAI image tensors must be RGB");
  prepare();
  auto path = directory_ / (std::to_string(count_++) + (audio ? ".wav" : ".ppm"));
  std::ofstream out(path, std::ios::binary);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  auto data = static_cast<const char*>(t.data) + t.byte_offset;
  if (audio) {
    const uint32_t bytes = static_cast<uint32_t>(t.shape[0] * 4);
    if (rate > 384000)
      throw std::invalid_argument("Audio sample rate too large");
    out.write("RIFF", 4);
    le(out, 36 + bytes, 4);
    out.write("WAVEfmt ", 8);
    le(out, 16, 4);
    le(out, 3, 2);
    le(out, 1, 2);
    le(out, rate, 4);
    le(out, rate * 4, 4);
    le(out, 4, 2);
    le(out, 32, 2);
    out.write("data", 4);
    le(out, bytes, 4);
    for (int64_t i = 0; i < t.shape[0]; ++i)
      out.write(data + i * strides[0], 4);
  } else {
    out << "P6\n" << t.shape[1] << ' ' << t.shape[0] << "\n255\n";
    for (int64_t y = 0; y < t.shape[0]; ++y)
      for (int64_t x = 0; x < t.shape[1]; ++x)
        for (int64_t c = 0; c < 3; ++c)
          out.put(data[y * strides[0] + x * strides[1] + c * strides[2]]);
  }
  out.close();
  return descriptor(path);
}
Json MediaStage::images(const std::vector<Tensor>& tensors) {
  Json result = Json::array();
  for (const auto& t : tensors)
    result.push_back(tensor(t, false, 0));
  return result;
}
Json MediaStage::encode(const GenerationRequest& r) {
  if (r.audio && r.audio_file)
    throw std::invalid_argument("Use either audio or audio_file");
  if (r.prompt && !r.messages.empty())
    throw std::invalid_argument("Use either prompt or messages");
  Json j = {{"images", images(r.images)},
            {"messages", Json::array()},
            {"max_new_tokens", r.max_new_tokens},
            {"enable_thinking", r.enable_thinking},
            {"language", r.language},
            {"asr_task", r.asr_task == ASRTask::Translate ? "translate" : "transcribe"},
            {"tools", r.tools},
            {"tool_choice", r.tool_choice}};
  if (r.prompt)
    j["prompt"] = *r.prompt;
  if (r.system_prompt)
    j["system_prompt"] = *r.system_prompt;
  if (r.audio_file)
    j["audio_file"] = file(*r.audio_file);
  if (r.audio)
    j["audio_file"] = tensor(*r.audio, true, r.sample_rate);
  for (const auto& m : r.messages) {
    Json message = {{"role", m.role},
                    {"content", m.content},
                    {"images", images(m.images)},
                    {"tool_calls", m.tool_calls}};
    if (m.name)
      message["name"] = *m.name;
    if (m.tool_call_id)
      message["tool_call_id"] = *m.tool_call_id;
    j["messages"].push_back(std::move(message));
  }
  return j;
}
} // namespace simaai::neat::pcie::genai::internal
