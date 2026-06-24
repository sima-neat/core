#include <neat.h>

#include "gst/GstInit.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/runtime/RunCore.h"

#include <gst/gst.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

using simaai::neat::AutoFlag;
using simaai::neat::BoxDecodeType;
using simaai::neat::BoxDecodeTypeOption;
using simaai::neat::Graph;
using simaai::neat::InputKind;
using simaai::neat::Model;
using simaai::neat::NormalizePreset;
using simaai::neat::PCIeSinkOptions;
using simaai::neat::PCIeSrcOptions;
using simaai::neat::PreprocessColorFormat;
using simaai::neat::PreprocessGraphFamily;
using simaai::neat::ResolvedPreprocessPlan;
using simaai::neat::ResizeMode;

constexpr int kInitialReadinessGraceSeconds = 15;
constexpr const char* kProgramName = "pcie-pipeline-builder";
constexpr const char* kSrcReadyMessage = "neat-pcie-src-pads-active";
constexpr const char* kSinkReadyMessage = "neat-pcie-sink-started";

volatile std::sig_atomic_t g_stop_requested = 0;

enum class Mode { Tensor, Accelerator, Image, BoxDecode };

struct CliOptions {
  std::filesystem::path model;
  std::filesystem::path model_options;
  int queue = -1;
  bool accelerator = false;
  bool print_gst = false;
};

struct ResolvedOptions {
  Mode mode = Mode::Tensor;
  Model::Options model_options;
};

struct Status {
  int schema = 1;
  std::string state;
  int queue = 0;
  pid_t pid = 0;
  std::string mode;
  std::string model;
  std::string started_at;
  std::string updated_at;
  std::string message;
  std::optional<std::string> error_code;
};

struct ReadinessState {
  bool src_ready = false;
  bool sink_ready = false;
  bool pipeline_ready = false;

  bool ready() const {
    return pipeline_ready;
  }

  std::string message() const {
    if (ready())
      return "pipeline armed and ready for PCIe host";
    return "pipeline starting; waiting for GStreamer PLAYING";
  }
};

class PciePipelineError : public std::runtime_error {
public:
  PciePipelineError(std::string code, std::string message)
      : std::runtime_error(message), code_(std::move(code)) {}

  const std::string& code() const noexcept {
    return code_;
  }

private:
  std::string code_;
};

std::string usage() {
  return "usage: pcie-pipeline-builder --model <model.tar.gz> "
         "--queue <0..5> [--accelerator] "
         "[--model-options <options.json>] [--print-gst]";
}

std::string mode_name(const Mode mode) {
  switch (mode) {
  case Mode::Tensor:
    return "tensor";
  case Mode::Accelerator:
    return "accelerator";
  case Mode::Image:
    return "image";
  case Mode::BoxDecode:
    return "boxdecode";
  }
  return "unknown";
}

bool parse_int(const std::string& text, int* out) {
  if (!out || text.empty())
    return false;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || !end || *end != '\0')
    return false;
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    return false;
  *out = static_cast<int>(value);
  return true;
}

std::string lower_copy(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return text;
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opt;
  bool have_model = false;
  bool have_model_options = false;
  bool have_queue = false;
  bool have_accelerator = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc)
        throw PciePipelineError("usage", std::string("missing value for ") + name);
      return argv[++i];
    };

    if (arg == "--model") {
      if (have_model)
        throw PciePipelineError("usage", "duplicate --model");
      opt.model = require_value("--model");
      have_model = true;
    } else if (arg == "--model-options") {
      if (have_model_options)
        throw PciePipelineError("usage", "duplicate --model-options");
      opt.model_options = require_value("--model-options");
      have_model_options = true;
    } else if (arg == "--queue") {
      if (have_queue)
        throw PciePipelineError("usage", "duplicate --queue");
      const std::string value = require_value("--queue");
      if (!parse_int(value, &opt.queue))
        throw PciePipelineError("usage", "invalid --queue '" + value + "'");
      have_queue = true;
    } else if (arg == "--accelerator") {
      if (have_accelerator)
        throw PciePipelineError("usage", "duplicate --accelerator");
      opt.accelerator = true;
      have_accelerator = true;
    } else if (arg == "--print-gst") {
      opt.print_gst = true;
    } else {
      throw PciePipelineError("usage", "unknown option '" + arg + "'");
    }
  }

  if (!have_model || !have_queue)
    throw PciePipelineError("usage", "missing required option");
  if (opt.queue < 0 || opt.queue > 5)
    throw PciePipelineError("usage", "--queue must be in range 0..5");
  if (opt.model.empty())
    throw PciePipelineError("usage", "--model must not be empty");
  if (have_model_options && opt.model_options.empty())
    throw PciePipelineError("usage", "--model-options must not be empty");
  if (opt.accelerator && have_model_options)
    throw PciePipelineError("usage", "--accelerator cannot be combined with --model-options");
  return opt;
}

void require_object(const nlohmann::json& value, const std::string& path) {
  if (!value.is_object())
    throw PciePipelineError("model_options", path + " must be an object");
}

void reject_unknown_fields(const nlohmann::json& object, const std::set<std::string>& allowed,
                           const std::string& path) {
  require_object(object, path);
  for (const auto& item : object.items()) {
    if (allowed.find(item.key()) == allowed.end()) {
      throw PciePipelineError("model_options", "unknown field " + path + "." + item.key());
    }
  }
}

int json_int_field(const nlohmann::json& object, const char* key, const std::string& path,
                   int min_value, int max_value) {
  const auto it = object.find(key);
  if (it == object.end())
    return 0;
  if (!it->is_number_integer()) {
    throw PciePipelineError("model_options", path + "." + key + " must be an integer");
  }
  const long long value = it->get<long long>();
  if (value < min_value || value > max_value) {
    throw PciePipelineError("model_options", path + "." + key + " is out of range");
  }
  return static_cast<int>(value);
}

float json_float_field(const nlohmann::json& object, const char* key, const std::string& path,
                       float min_value, float max_value) {
  const auto it = object.find(key);
  if (it == object.end())
    return 0.0f;
  if (!it->is_number()) {
    throw PciePipelineError("model_options", path + "." + key + " must be a number");
  }
  const float value = it->get<float>();
  if (value < min_value || value > max_value) {
    throw PciePipelineError("model_options", path + "." + key + " is out of range");
  }
  return value;
}

std::string json_string_field(const nlohmann::json& object, const char* key,
                              const std::string& path) {
  const auto it = object.find(key);
  if (it == object.end())
    return {};
  if (!it->is_string()) {
    throw PciePipelineError("model_options", path + "." + key + " must be a string");
  }
  return it->get<std::string>();
}

std::array<float, 3> json_float3_field(const nlohmann::json& object, const char* key,
                                       const std::string& path) {
  const auto it = object.find(key);
  if (it == object.end())
    return {0.0f, 0.0f, 0.0f};
  if (!it->is_array() || it->size() != 3) {
    throw PciePipelineError("model_options",
                            path + "." + key + " must be an array with three numbers");
  }
  std::array<float, 3> out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (!(*it)[i].is_number()) {
      throw PciePipelineError("model_options", path + "." + key + " must contain only numbers");
    }
    out[i] = (*it)[i].get<float>();
  }
  return out;
}

ResizeMode parse_resize_mode(const std::string& raw) {
  const std::string value = lower_copy(raw);
  if (value == "stretch")
    return ResizeMode::Stretch;
  if (value == "letterbox")
    return ResizeMode::Letterbox;
  if (value == "crop")
    return ResizeMode::Crop;
  throw PciePipelineError("model_options", "invalid preprocess.resize.mode '" + raw + "'");
}

PreprocessColorFormat parse_input_color_format(const std::string& raw) {
  const std::string value = lower_copy(raw);
  if (value == "auto" || value.empty())
    return PreprocessColorFormat::Auto;
  if (value == "rgb")
    return PreprocessColorFormat::RGB;
  if (value == "bgr")
    return PreprocessColorFormat::BGR;
  if (value == "gray8" || value == "gray" || value == "grayscale")
    return PreprocessColorFormat::GRAY8;
  if (value == "nv12")
    return PreprocessColorFormat::NV12;
  if (value == "i420" || value == "iyuv")
    return PreprocessColorFormat::I420;
  throw PciePipelineError("model_options",
                          "invalid preprocess.color_convert.input_format '" + raw + "'");
}

PreprocessColorFormat parse_output_color_format(const std::string& raw) {
  const std::string value = lower_copy(raw);
  if (value == "auto" || value.empty())
    return PreprocessColorFormat::Auto;
  if (value == "rgb")
    return PreprocessColorFormat::RGB;
  if (value == "bgr")
    return PreprocessColorFormat::BGR;
  if (value == "nv12" || value == "i420" || value == "iyuv" || value == "gray" ||
      value == "gray8" || value == "grayscale") {
    throw PciePipelineError(
        "model_options", "preprocess.color_convert.output_format supports only auto, rgb, or bgr");
  }
  throw PciePipelineError("model_options",
                          "invalid preprocess.color_convert.output_format '" + raw + "'");
}

NormalizePreset parse_normalize_preset(const std::string& raw) {
  const std::string value = lower_copy(raw);
  if (value.empty() || value == "none")
    return NormalizePreset::None;
  if (value == "imagenet" || value == "image_net")
    return NormalizePreset::ImageNet;
  if (value == "coco_yolo" || value == "coco-yolo" || value == "yolo")
    return NormalizePreset::COCO_YOLO;
  throw PciePipelineError("model_options", "invalid preprocess.normalize.preset '" + raw + "'");
}

BoxDecodeType parse_model_options_decode_type(const std::string& raw) {
  const std::string value = lower_copy(raw);
  if (value.empty() || value == "auto" || value == "unspecified")
    return BoxDecodeType::Unspecified;
  const auto parsed = simaai::neat::pipeline_internal::sima::parse_box_decode_type_token(value);
  if (!parsed.has_value()) {
    throw PciePipelineError("model_options", "invalid boxdecode.decode_type '" + raw + "'");
  }
  if (*parsed == BoxDecodeType::Yolo) {
    throw PciePipelineError("model_options",
                            "boxdecode.decode_type must be concrete, not generic yolo");
  }
  return *parsed;
}

BoxDecodeTypeOption parse_model_options_decode_type_option(const std::string& raw) {
  const auto parsed =
      simaai::neat::pipeline_internal::sima::parse_box_decode_type_option_token(raw);
  if (!parsed.has_value()) {
    throw PciePipelineError("model_options", "invalid boxdecode.decode_type_option '" + raw + "'");
  }
  return *parsed;
}

void apply_preprocess_model_options(const nlohmann::json& preprocess, Model::Options* opt) {
  reject_unknown_fields(preprocess, {"input_max", "resize", "color_convert", "normalize"},
                        "preprocess");

  if (const auto it = preprocess.find("input_max"); it != preprocess.end()) {
    reject_unknown_fields(*it, {"width", "height", "depth"}, "preprocess.input_max");
    if (it->contains("width")) {
      opt->preprocess.input_max_width =
          json_int_field(*it, "width", "preprocess.input_max", 0, 4096);
    }
    if (it->contains("height")) {
      opt->preprocess.input_max_height =
          json_int_field(*it, "height", "preprocess.input_max", 0, 4096);
    }
    if (it->contains("depth")) {
      opt->preprocess.input_max_depth = json_int_field(*it, "depth", "preprocess.input_max", 0, 4);
    }
  }

  if (const auto it = preprocess.find("resize"); it != preprocess.end()) {
    reject_unknown_fields(*it, {"mode", "pad_value", "scaling_type"}, "preprocess.resize");
    opt->preprocess.resize.enable = AutoFlag::On;
    if (it->contains("mode"))
      opt->preprocess.resize.mode =
          parse_resize_mode(json_string_field(*it, "mode", "preprocess.resize"));
    if (it->contains("pad_value")) {
      opt->preprocess.resize.pad_value =
          json_int_field(*it, "pad_value", "preprocess.resize", 0, 255);
    }
    if (it->contains("scaling_type"))
      opt->preprocess.resize.scaling_type =
          json_string_field(*it, "scaling_type", "preprocess.resize");
  }

  if (const auto it = preprocess.find("color_convert"); it != preprocess.end()) {
    reject_unknown_fields(*it, {"input_format", "output_format"}, "preprocess.color_convert");
    opt->preprocess.color_convert.enable = AutoFlag::On;
    if (it->contains("input_format")) {
      opt->preprocess.color_convert.input_format = parse_input_color_format(
          json_string_field(*it, "input_format", "preprocess.color_convert"));
    }
    if (it->contains("output_format")) {
      opt->preprocess.color_convert.output_format = parse_output_color_format(
          json_string_field(*it, "output_format", "preprocess.color_convert"));
    }
  }

  if (const auto it = preprocess.find("normalize"); it != preprocess.end()) {
    reject_unknown_fields(*it, {"preset", "mean", "stddev"}, "preprocess.normalize");
    bool normalize_requested = false;
    if (it->contains("preset")) {
      opt->preprocess.preset =
          parse_normalize_preset(json_string_field(*it, "preset", "preprocess.normalize"));
      normalize_requested = opt->preprocess.preset != NormalizePreset::None;
    }
    if (it->contains("mean")) {
      opt->preprocess.normalize.mean = json_float3_field(*it, "mean", "preprocess.normalize");
      opt->preprocess.normalize.has_explicit_stats = true;
      normalize_requested = true;
    }
    if (it->contains("stddev")) {
      opt->preprocess.normalize.stddev = json_float3_field(*it, "stddev", "preprocess.normalize");
      for (const float value : opt->preprocess.normalize.stddev) {
        if (value == 0.0f) {
          throw PciePipelineError("model_options",
                                  "preprocess.normalize.stddev values must be non-zero");
        }
      }
      opt->preprocess.normalize.has_explicit_stats = true;
      normalize_requested = true;
    }
    if (normalize_requested)
      opt->preprocess.normalize.enable = AutoFlag::On;
  }
}

void apply_boxdecode_model_options(const nlohmann::json& boxdecode, Model::Options* opt) {
  reject_unknown_fields(boxdecode,
                        {"decode_type", "decode_type_option", "score_threshold",
                         "nms_iou_threshold", "top_k", "num_classes"},
                        "boxdecode");
  if (boxdecode.contains("decode_type")) {
    opt->decode_type =
        parse_model_options_decode_type(json_string_field(boxdecode, "decode_type", "boxdecode"));
  }
  if (boxdecode.contains("decode_type_option")) {
    opt->decode_type_option = parse_model_options_decode_type_option(
        json_string_field(boxdecode, "decode_type_option", "boxdecode"));
  }
  if (boxdecode.contains("score_threshold")) {
    opt->score_threshold = json_float_field(boxdecode, "score_threshold", "boxdecode", 0.0f, 1.0f);
  }
  if (boxdecode.contains("nms_iou_threshold")) {
    opt->nms_iou_threshold =
        json_float_field(boxdecode, "nms_iou_threshold", "boxdecode", 0.0f, 1.0f);
  }
  if (boxdecode.contains("top_k")) {
    opt->top_k = json_int_field(boxdecode, "top_k", "boxdecode", 0, 1000000);
  }
  if (boxdecode.contains("num_classes")) {
    opt->num_classes = json_int_field(boxdecode, "num_classes", "boxdecode", 0, 1000000);
  }
}

nlohmann::json read_model_options_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw PciePipelineError("model_options", "failed to open --model-options " + path.string());
  }

  nlohmann::json root;
  try {
    in >> root;
  } catch (const nlohmann::json::exception& e) {
    throw PciePipelineError("model_options",
                            "failed to parse --model-options " + path.string() + ": " + e.what());
  }
  return root;
}

void validate_model_options_root(const nlohmann::json& root) {
  reject_unknown_fields(root, {"schema", "preprocess", "boxdecode"}, "root");
  if (!root.contains("schema") || !root["schema"].is_number_integer() ||
      root["schema"].get<int>() != 1) {
    throw PciePipelineError("model_options", "--model-options schema must be integer 1");
  }
}

Mode apply_model_options_json(const nlohmann::json& root, Model::Options* opt) {
  validate_model_options_root(root);

  const bool has_preprocess = root.contains("preprocess");
  const bool has_boxdecode = root.contains("boxdecode");
  if (!has_preprocess) {
    if (has_boxdecode) {
      throw PciePipelineError("model_options", "boxdecode options require a preprocess object");
    }
    throw PciePipelineError("model_options", "--model-options must contain a preprocess object");
  }
  apply_preprocess_model_options(root.at("preprocess"), opt);
  if (const auto it = root.find("boxdecode"); it != root.end()) {
    apply_boxdecode_model_options(*it, opt);
    return Mode::BoxDecode;
  }
  return Mode::Image;
}

void apply_route_owned_options(const Mode mode, Model::Options* opt) {
  opt->upstream_name = "n0_pciesrc";
  opt->preprocess.kind =
      (mode == Mode::Tensor || mode == Mode::Accelerator) ? InputKind::Tensor : InputKind::Image;
}

ResolvedOptions resolve_options(const CliOptions& opt) {
  ResolvedOptions resolved;
  if (opt.accelerator) {
    resolved.mode = Mode::Accelerator;
  } else if (opt.model_options.empty()) {
    resolved.mode = Mode::Tensor;
  } else {
    const nlohmann::json root = read_model_options_file(opt.model_options);
    resolved.mode = apply_model_options_json(root, &resolved.model_options);
  }
  apply_route_owned_options(resolved.mode, &resolved.model_options);
  return resolved;
}

std::filesystem::path canonical_model_path(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path out = std::filesystem::weakly_canonical(path, ec);
  if (ec)
    out = std::filesystem::absolute(path, ec);
  if (ec)
    out = path;
  return out.lexically_normal();
}

std::string getenv_or(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  if (value && *value)
    return value;
  return fallback;
}

std::string now_utc_iso8601() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&now, &tm);
  char buf[32]{};
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string status_to_json(const Status& s) {
  nlohmann::ordered_json out{
      {"schema", s.schema},
      {"state", s.state},
      {"queue", s.queue},
      {"pid", static_cast<long long>(s.pid)},
      {"mode", s.mode},
      {"model", s.model},
      {"started_at", s.started_at},
      {"updated_at", s.updated_at},
      {"message", s.message},
      {"error_code", s.error_code.has_value() ? nlohmann::ordered_json(*s.error_code)
                                              : nlohmann::ordered_json(nullptr)},
  };
  return out.dump(2) + "\n";
}

class StatusWriter {
public:
  explicit StatusWriter(std::filesystem::path path) : path_(std::move(path)) {}

  void write(Status status) const {
    status.updated_at = now_utc_iso8601();
    const std::filesystem::path tmp =
        path_.string() + ".tmp." + std::to_string(static_cast<long long>(getpid()));
    {
      std::ofstream out(tmp, std::ios::out | std::ios::trunc);
      if (!out)
        throw PciePipelineError("lifecycle", "failed to open status temp file " + tmp.string());
      out << status_to_json(status);
      out.flush();
      if (!out)
        throw PciePipelineError("lifecycle", "failed to write status temp file " + tmp.string());
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path_, ec);
    if (ec) {
      std::filesystem::remove(tmp);
      throw PciePipelineError("lifecycle", "failed to atomically replace status file " +
                                               path_.string() + ": " + ec.message());
    }
  }

private:
  std::filesystem::path path_;
};

bool pid_is_live(pid_t pid) {
  if (pid <= 0)
    return false;
  if (::kill(pid, 0) == 0)
    return true;
  return errno == EPERM;
}

std::optional<pid_t> read_pid_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in)
    return std::nullopt;
  long long value = -1;
  in >> value;
  if (!in || value <= 0 || value > std::numeric_limits<pid_t>::max())
    return std::nullopt;
  return static_cast<pid_t>(value);
}

bool proc_cmdline_contains(pid_t pid, const std::string& needle) {
  std::ifstream in("/proc/" + std::to_string(static_cast<long long>(pid)) + "/cmdline",
                   std::ios::in | std::ios::binary);
  if (!in)
    return false;
  std::string cmd((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  for (char& ch : cmd) {
    if (ch == '\0')
      ch = ' ';
  }
  return cmd.find(needle) != std::string::npos;
}

void require_directory(const std::filesystem::path& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec))
    return;
  throw PciePipelineError("lifecycle", "required lifecycle directory is missing: " + path.string());
}

class QueueOwnership {
public:
  QueueOwnership(std::filesystem::path pid_path, std::filesystem::path status_path)
      : pid_path_(std::move(pid_path)), status_path_(std::move(status_path)) {}

  void acquire() {
    if (std::filesystem::exists(pid_path_)) {
      const std::optional<pid_t> old_pid = read_pid_file(pid_path_);
      if (old_pid.has_value() && pid_is_live(*old_pid) &&
          proc_cmdline_contains(*old_pid, kProgramName)) {
        throw PciePipelineError("queue_busy", "queue already owned by live " +
                                                  std::string(kProgramName) + " pid " +
                                                  std::to_string(static_cast<long long>(*old_pid)));
      }
      std::error_code ec;
      std::filesystem::remove(pid_path_, ec);
      std::filesystem::remove(status_path_, ec);
    }

    const int fd = ::open(pid_path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
      if (errno == EEXIST)
        throw PciePipelineError("queue_busy",
                                "queue pid file already exists: " + pid_path_.string());
      throw PciePipelineError("lifecycle", "failed to create pid file " + pid_path_.string() +
                                               ": " + std::strerror(errno));
    }
    const std::string text = std::to_string(static_cast<long long>(getpid())) + "\n";
    const ssize_t wrote = ::write(fd, text.data(), text.size());
    const int close_rc = ::close(fd);
    if (wrote != static_cast<ssize_t>(text.size()) || close_rc != 0) {
      std::error_code ec;
      std::filesystem::remove(pid_path_, ec);
      throw PciePipelineError("lifecycle", "failed to write pid file " + pid_path_.string());
    }
    owned_ = true;
  }

  void release() noexcept {
    if (!owned_)
      return;
    const std::optional<pid_t> current = read_pid_file(pid_path_);
    if (current.has_value() && *current == getpid()) {
      std::error_code ec;
      std::filesystem::remove(pid_path_, ec);
    }
    owned_ = false;
  }

  ~QueueOwnership() {
    release();
  }

private:
  std::filesystem::path pid_path_;
  std::filesystem::path status_path_;
  bool owned_ = false;
};

void redirect_to_log(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd < 0) {
    throw PciePipelineError("lifecycle", "failed to open log file " + path.string() + ": " +
                                             std::strerror(errno));
  }
  if (::dup2(fd, STDOUT_FILENO) < 0 || ::dup2(fd, STDERR_FILENO) < 0) {
    const int saved_errno = errno;
    ::close(fd);
    throw PciePipelineError("lifecycle", "failed to redirect stdout/stderr to log: " +
                                             std::string(std::strerror(saved_errno)));
  }
  ::close(fd);
}

void on_signal(int) {
  g_stop_requested = 1;
}

void install_signal_handlers() {
  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
}

void validate_boxdecode_contract(const Model& model, const std::string& backend) {
  const auto plan = model.resolved_preprocess_plan();
  if (!plan.enabled || plan.graph_family != PreprocessGraphFamily::Preproc) {
    throw PciePipelineError(
        "model_contract",
        "boxdecode mode requires an image/generic-preproc route before inference");
  }

  const auto info = model.info();
  const bool selected_boxdecode = info.selection.selected_post_kind == "boxdecode";
  const bool backend_has_boxdecode = backend.find("boxdecode") != std::string::npos ||
                                     backend.find("BoxDecode") != std::string::npos;
  if (!info.capabilities.has_post_boxdecode || (!selected_boxdecode && !backend_has_boxdecode)) {
    throw PciePipelineError(
        "model_contract",
        "boxdecode mode requires a model route that resolves to a boxdecode postprocess");
  }
}

BoxDecodeType infer_boxdecode_type_from_mpk(const std::filesystem::path& model_path) {
  simaai::neat::internal::ModelPack pack(model_path.string());
  const auto& mpk = pack.mpk_contract();
  if (!mpk.has_value()) {
    throw PciePipelineError("model_contract",
                            "boxdecode mode requires MPK metadata with a boxdecode contract");
  }

  std::string error;
  const auto route_flags =
      simaai::neat::pipeline_internal::sima::resolve_model_managed_boxdecode_route_flags_from_mpk(
          *mpk, nullptr, &error);
  if (!route_flags.has_value()) {
    throw PciePipelineError(
        "model_contract", "boxdecode mode could not resolve MPK boxdecode route flags: " +
                              (error.empty() ? std::string("missing MPK/upstream facts") : error));
  }

  error.clear();
  const auto contract =
      simaai::neat::pipeline_internal::sima::build_boxdecode_static_contract_from_mpk(
          *mpk, *route_flags, &error);
  if (!contract.has_value()) {
    throw PciePipelineError(
        "model_contract", "boxdecode mode could not derive MPK boxdecode contract: " +
                              (error.empty() ? std::string("missing MPK/upstream facts") : error));
  }

  if (!simaai::neat::pipeline_internal::sima::is_box_decode_type_specified(contract->decode_type) ||
      contract->decode_type == BoxDecodeType::Yolo) {
    throw PciePipelineError("model_contract",
                            "boxdecode mode requires MPK metadata with a concrete decode type");
  }

  return contract->decode_type;
}

void validate_image_preprocess_plan(const Mode mode, const Model& model,
                                    const Model::Options& requested_options) {
  if (mode != Mode::Image && mode != Mode::BoxDecode) {
    return;
  }

  const ResolvedPreprocessPlan plan = model.resolved_preprocess_plan();
  std::vector<std::string> issues;

  if (!plan.enabled) {
    issues.push_back("preprocess plan is disabled");
  }
  if (plan.resolved_kind != InputKind::Image) {
    issues.push_back("resolved preprocess input kind is not Image");
  }
  if (plan.effective.resize.enable != AutoFlag::On) {
    issues.push_back("preprocess.resize did not resolve to On");
  }
  if (plan.effective.resize.width <= 0 || plan.effective.resize.height <= 0) {
    issues.push_back("preprocess.resize target width/height are unresolved");
  }
  if (plan.mla_contract.width > 0 && plan.effective.resize.width > 0 &&
      plan.effective.resize.width != plan.mla_contract.width) {
    std::ostringstream ss;
    ss << "preprocess.resize.width=" << plan.effective.resize.width
       << " does not match MLA contract width=" << plan.mla_contract.width;
    issues.push_back(ss.str());
  }
  if (plan.mla_contract.height > 0 && plan.effective.resize.height > 0 &&
      plan.effective.resize.height != plan.mla_contract.height) {
    std::ostringstream ss;
    ss << "preprocess.resize.height=" << plan.effective.resize.height
       << " does not match MLA contract height=" << plan.mla_contract.height;
    issues.push_back(ss.str());
  }

  const auto& requested = requested_options.preprocess;
  if (requested.input_max_width > 0 &&
      plan.effective.input_max_width != requested.input_max_width) {
    std::ostringstream ss;
    ss << "effective input_max_width=" << plan.effective.input_max_width
       << " does not match requested input_max_width=" << requested.input_max_width;
    issues.push_back(ss.str());
  }
  if (requested.input_max_height > 0 &&
      plan.effective.input_max_height != requested.input_max_height) {
    std::ostringstream ss;
    ss << "effective input_max_height=" << plan.effective.input_max_height
       << " does not match requested input_max_height=" << requested.input_max_height;
    issues.push_back(ss.str());
  }
  if (requested.input_max_depth > 0 &&
      plan.effective.input_max_depth != requested.input_max_depth) {
    std::ostringstream ss;
    ss << "effective input_max_depth=" << plan.effective.input_max_depth
       << " does not match requested input_max_depth=" << requested.input_max_depth;
    issues.push_back(ss.str());
  }

  if (!issues.empty()) {
    std::ostringstream ss;
    ss << "image preprocess plan validation failed for mode=" << mode_name(mode) << ": ";
    for (std::size_t i = 0; i < issues.size(); ++i) {
      if (i != 0) {
        ss << "; ";
      }
      ss << issues[i];
    }
    ss << "\n" << plan.to_debug_string();
    throw PciePipelineError("model_contract", ss.str());
  }
}

Graph compose_graph(const CliOptions& opt, const ResolvedOptions& resolved,
                    const std::filesystem::path& model_path, std::unique_ptr<Model>* model_owner) {
  Model::Options model_options = resolved.model_options;
  if (resolved.mode == Mode::BoxDecode &&
      !simaai::neat::pipeline_internal::sima::is_box_decode_type_specified(
          model_options.decode_type)) {
    model_options.decode_type = infer_boxdecode_type_from_mpk(model_path);
  }

  auto model = std::make_unique<Model>(model_path.string(), model_options);
  validate_image_preprocess_plan(resolved.mode, *model, model_options);

  Graph graph("pcie-pipeline");
  PCIeSrcOptions src_options;
  src_options.queue = opt.queue;
  graph.add(simaai::neat::nodes::PCIeSrc(src_options));

  if (resolved.mode == Mode::Accelerator) {
    graph.add(model->inference());
  } else {
    graph.add(model->graph());
  }

  PCIeSinkOptions sink_options;
  sink_options.queue = opt.queue;
  graph.add(simaai::neat::nodes::PCIeSink(sink_options));
  *model_owner = std::move(model);
  return graph;
}

std::string compose_backend_pipeline(const CliOptions& opt, const ResolvedOptions& resolved,
                                     const std::filesystem::path& model_path) {
  std::unique_ptr<Model> model_owner;
  Graph graph = compose_graph(opt, resolved, model_path, &model_owner);
  const std::string backend = graph.describe_backend(false);
  if (resolved.mode == Mode::BoxDecode && model_owner)
    validate_boxdecode_contract(*model_owner, backend);
  return backend;
}

std::string gst_error_message(GstMessage* msg) {
  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(msg, &error, &debug);
  std::ostringstream out;
  out << "GStreamer ERROR";
  if (GST_MESSAGE_SRC(msg)) {
    out << " from " << GST_OBJECT_NAME(GST_MESSAGE_SRC(msg));
  }
  if (error && error->message)
    out << ": " << error->message;
  if (debug && *debug)
    out << " (" << debug << ")";
  if (error)
    g_error_free(error);
  if (debug)
    g_free(debug);
  return out.str();
}

bool pipeline_playing_or_pending(GstElement* pipeline) {
  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_VOID_PENDING;
  const GstStateChangeReturn rc = gst_element_get_state(pipeline, &state, &pending, 0);
  return rc == GST_STATE_CHANGE_SUCCESS || rc == GST_STATE_CHANGE_NO_PREROLL ||
         state == GST_STATE_PLAYING || pending == GST_STATE_PLAYING;
}

std::string pipeline_state_summary(GstElement* pipeline) {
  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_VOID_PENDING;
  const GstStateChangeReturn rc = gst_element_get_state(pipeline, &state, &pending, 0);
  std::ostringstream out;
  out << "state=" << gst_element_state_get_name(state)
      << " pending=" << gst_element_state_get_name(pending)
      << " rc=" << static_cast<int>(rc);
  return out.str();
}

std::string immediate_bus_error_or_empty(GstBus* bus) {
  if (!bus)
    return {};
  GstMessage* msg = gst_bus_pop_filtered(
      bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  if (!msg)
    return {};

  std::string message;
  const GstMessageType type = GST_MESSAGE_TYPE(msg);
  if (type == GST_MESSAGE_ERROR) {
    message = gst_error_message(msg);
  } else if (type == GST_MESSAGE_EOS) {
    message = "GStreamer pipeline reached EOS during state change";
  }
  gst_message_unref(msg);
  return message;
}

class ManagedPipeline {
public:
  ManagedPipeline(Graph graph, std::unique_ptr<Model> model_owner)
      : graph_(std::move(graph)), model_owner_(std::move(model_owner)) {}

  ~ManagedPipeline() {
    stop();
  }

  ReadinessState start_and_probe_ready() {
    run_ = graph_.build();
    const auto core = simaai::neat::run_internal::core(run_);
    pipeline_ = core ? core->pipeline.stream.pipeline_handle() : nullptr;
    if (!pipeline_) {
      throw PciePipelineError("pipeline_build",
                              "Graph::build did not expose a GStreamer pipeline handle");
    }

    bus_ = gst_element_get_bus(pipeline_);
    if (!bus_)
      throw PciePipelineError("pipeline_build", "failed to get GStreamer bus");

    const GstStateChangeReturn set_state_rc = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (set_state_rc == GST_STATE_CHANGE_FAILURE) {
      std::string message = "failed to set pipeline to PLAYING";
      const std::string bus_error = immediate_bus_error_or_empty(bus_);
      if (!bus_error.empty())
        message += ": " + bus_error;
      message += " (" + pipeline_state_summary(pipeline_) + ")";
      throw PciePipelineError("gst_state", message);
    }

    ReadinessState readiness;
    readiness.pipeline_ready = pipeline_playing_or_pending(pipeline_);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kInitialReadinessGraceSeconds);

    while (!g_stop_requested) {
      if (readiness.ready())
        return readiness;

      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline)
        break;
      const auto remaining_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
      const guint64 wait_ns =
          static_cast<guint64>(std::max<long long>(1, std::min<long long>(250, remaining_ms))) *
          GST_MSECOND;

      GstMessage* msg = gst_bus_timed_pop_filtered(
          bus_, wait_ns,
          static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_ELEMENT |
                                      GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_EOS));
      if (!msg) {
        readiness.pipeline_ready = readiness.pipeline_ready || pipeline_playing_or_pending(pipeline_);
        continue;
      }
      handle_startup_message(msg, &readiness);
      gst_message_unref(msg);
    }

    if (g_stop_requested)
      throw PciePipelineError("signal", "startup interrupted");
    return readiness;
  }

  void run_until_error_or_stop(ReadinessState readiness,
                               const std::function<void(const ReadinessState&)>& on_ready) {
    bool ready_reported = readiness.ready();
    while (!g_stop_requested) {
      GstMessage* msg = gst_bus_timed_pop_filtered(
          bus_, 250 * GST_MSECOND,
          static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_ELEMENT |
                                      GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_EOS));
      if (!msg) {
        readiness.pipeline_ready =
            readiness.pipeline_ready || pipeline_playing_or_pending(pipeline_);
        if (!ready_reported && readiness.ready() && on_ready) {
          ready_reported = true;
          on_ready(readiness);
        }
        continue;
      }
      const GstMessageType type = GST_MESSAGE_TYPE(msg);
      if (type == GST_MESSAGE_ERROR) {
        const std::string message = gst_error_message(msg);
        gst_message_unref(msg);
        throw PciePipelineError("gst_state", message);
      }
      if (type == GST_MESSAGE_EOS) {
        gst_message_unref(msg);
        throw PciePipelineError("gst_state", "GStreamer pipeline reached EOS");
      }
      const bool was_ready = readiness.ready();
      handle_startup_message(msg, &readiness);
      if (!was_ready && !ready_reported && readiness.ready() && on_ready) {
        ready_reported = true;
        on_ready(readiness);
      }
      gst_message_unref(msg);
    }
  }

  void stop() noexcept {
    if (bus_) {
      gst_object_unref(bus_);
      bus_ = nullptr;
    }
    run_.stop();
    pipeline_ = nullptr;
  }

private:
  static void handle_startup_message(GstMessage* msg, ReadinessState* readiness) {
    const GstMessageType type = GST_MESSAGE_TYPE(msg);
    if (type == GST_MESSAGE_ERROR)
      throw PciePipelineError("gst_state", gst_error_message(msg));
    if (type == GST_MESSAGE_EOS)
      throw PciePipelineError("gst_state", "GStreamer pipeline reached EOS during startup");
    if (type == GST_MESSAGE_ELEMENT) {
      const GstStructure* structure = gst_message_get_structure(msg);
      if (!structure)
        return;
      const char* name = gst_structure_get_name(structure);
      if (!name)
        return;
      if (std::strcmp(name, kSrcReadyMessage) == 0) {
        if (!readiness->src_ready)
          std::cout << "PCIe source peer active" << std::endl;
        readiness->src_ready = true;
      }
      if (std::strcmp(name, kSinkReadyMessage) == 0) {
        if (!readiness->sink_ready)
          std::cout << "PCIe sink peer started" << std::endl;
        readiness->sink_ready = true;
      }
      return;
    }
    if (type == GST_MESSAGE_STATE_CHANGED && GST_MESSAGE_SRC(msg) &&
        GST_IS_PIPELINE(GST_MESSAGE_SRC(msg))) {
      GstState old_state = GST_STATE_NULL;
      GstState new_state = GST_STATE_NULL;
      GstState pending = GST_STATE_VOID_PENDING;
      gst_message_parse_state_changed(msg, &old_state, &new_state, &pending);
      (void)old_state;
      if (new_state == GST_STATE_PLAYING || pending == GST_STATE_PLAYING)
        readiness->pipeline_ready = true;
    }
  }

  Graph graph_;
  std::unique_ptr<Model> model_owner_;
  simaai::neat::Run run_;
  GstElement* pipeline_ = nullptr;
  GstBus* bus_ = nullptr;
};

Status make_status(const CliOptions& opt, const ResolvedOptions& resolved,
                   const std::filesystem::path& model_path) {
  const std::string now = now_utc_iso8601();
  Status status;
  status.state = "starting";
  status.queue = opt.queue;
  status.pid = getpid();
  status.mode = mode_name(resolved.mode);
  status.model = model_path.string();
  status.started_at = now;
  status.updated_at = now;
  status.message = "building pipeline";
  return status;
}

void set_status(StatusWriter& writer, Status& status, std::string state, std::string message,
                std::optional<std::string> error_code = std::nullopt) {
  status.state = std::move(state);
  status.message = std::move(message);
  status.error_code = std::move(error_code);
  writer.write(status);
}

int run_builder(const CliOptions& opt) {
  const std::filesystem::path model_path = canonical_model_path(opt.model);
  const ResolvedOptions resolved = resolve_options(opt);

  if (opt.print_gst) {
    std::cout << compose_backend_pipeline(opt, resolved, model_path) << '\n';
    return 0;
  }

  const std::filesystem::path run_dir = getenv_or("SIMA_NEAT_PCIE_RUN_DIR", "/run/sima-neat/pcie");
  const std::filesystem::path log_dir =
      getenv_or("SIMA_NEAT_PCIE_LOG_DIR", "/var/log/sima-neat/pcie");
  require_directory(run_dir);
  require_directory(log_dir);

  const std::string queue_name = "q" + std::to_string(opt.queue);
  const std::filesystem::path pid_path = run_dir / (queue_name + ".pid");
  const std::filesystem::path status_path = run_dir / (queue_name + ".status");
  const std::filesystem::path log_path = log_dir / (queue_name + ".log");

  QueueOwnership ownership(pid_path, status_path);
  ownership.acquire();
  redirect_to_log(log_path);
  install_signal_handlers();

  StatusWriter status_writer(status_path);
  Status status = make_status(opt, resolved, model_path);
  set_status(status_writer, status, "starting", "building pipeline");

  std::unique_ptr<ManagedPipeline> pipeline;
  try {
    Graph graph;
    std::unique_ptr<Model> model_owner;
    try {
      graph = compose_graph(opt, resolved, model_path, &model_owner);
      if (resolved.mode == Mode::BoxDecode && model_owner)
        validate_boxdecode_contract(*model_owner, graph.describe_backend(false));
    } catch (const PciePipelineError&) {
      throw;
    } catch (const std::exception& e) {
      throw PciePipelineError("model_contract", e.what());
    }
    pipeline = std::make_unique<ManagedPipeline>(std::move(graph), std::move(model_owner));
    set_status(status_writer, status, "starting", "starting GStreamer pipeline");
    ReadinessState readiness = pipeline->start_and_probe_ready();
    if (readiness.ready()) {
      set_status(status_writer, status, "ready", readiness.message());
    } else {
      set_status(status_writer, status, "waiting", readiness.message());
    }

    pipeline->run_until_error_or_stop(readiness, [&](const ReadinessState& updated) {
      set_status(status_writer, status, "ready", updated.message());
    });
    set_status(status_writer, status, "stopping", "signal received");
    pipeline->stop();
    ownership.release();
    set_status(status_writer, status, "exited", "pipeline stopped");
    return 0;
  } catch (const PciePipelineError& e) {
    if (e.code() == "signal") {
      set_status(status_writer, status, "stopping", e.what());
      if (pipeline)
        pipeline->stop();
      ownership.release();
      set_status(status_writer, status, "exited", "pipeline stopped");
      return 0;
    }
    set_status(status_writer, status, "failed", e.what(), e.code());
    if (pipeline)
      pipeline->stop();
    ownership.release();
    return 1;
  } catch (const std::exception& e) {
    set_status(status_writer, status, "failed", e.what(), "pipeline_build");
    if (pipeline)
      pipeline->stop();
    ownership.release();
    return 1;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    CliOptions opt = parse_cli(argc, argv);
    simaai::neat::gst_init_once();
    return run_builder(opt);
  } catch (const PciePipelineError& e) {
    if (e.code() == "usage") {
      std::cerr << e.what() << '\n' << usage() << '\n';
      return 2;
    }
    std::cerr << e.code() << ": " << e.what() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
