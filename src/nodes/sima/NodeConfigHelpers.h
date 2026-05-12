#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <algorithm>
#include <cstddef>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace simaai::neat::node_helpers {

using json = nlohmann::json;

inline json load_json_file(const std::string& path, const char* label) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  json j;
  in >> j;
  return j;
}

template <typename T> inline T json_or(const json& cfg, const char* key, T fallback) {
  const auto it = cfg.find(key);
  if (it == cfg.end() || it->is_null()) {
    return fallback;
  }
  return it->get<T>();
}

template <typename T> inline std::vector<T> json_vector_or(const json& cfg, const char* key) {
  const auto it = cfg.find(key);
  if (it == cfg.end() || it->is_null()) {
    return {};
  }
  return it->get<std::vector<T>>();
}

inline std::vector<std::vector<int>> json_shape_list_or(const json& cfg,
                                                        std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    const auto it = cfg.find(key);
    if (it == cfg.end() || it->is_null() || !it->is_array()) {
      continue;
    }
    if (it->empty()) {
      return {};
    }
    if (it->front().is_array()) {
      return it->get<std::vector<std::vector<int>>>();
    }
    return {it->get<std::vector<int>>()};
  }
  return {};
}

inline void write_json_file(const json& j, const std::string& path, const char* label) {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
  }
  out << j.dump(2);
}

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

inline std::string write_json_memfd(const json& j, int* out_fd, const char* label,
                                    const char* memfd_name) {
#ifdef SYS_memfd_create
  const int fd = static_cast<int>(::syscall(SYS_memfd_create, memfd_name, MFD_CLOEXEC));
  if (fd < 0) {
    throw std::runtime_error(std::string(label) + ": memfd_create failed: " + std::strerror(errno));
  }
  const std::string payload = j.dump(2);
  const char* ptr = payload.data();
  std::size_t remaining = payload.size();
  while (remaining > 0U) {
    const ssize_t wrote = ::write(fd, ptr, remaining);
    if (wrote < 0) {
      const std::string reason = std::strerror(errno);
      ::close(fd);
      throw std::runtime_error(std::string(label) + ": memfd write failed: " + reason);
    }
    ptr += static_cast<std::size_t>(wrote);
    remaining -= static_cast<std::size_t>(wrote);
  }
  if (::lseek(fd, 0, SEEK_SET) < 0) {
    const std::string reason = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error(std::string(label) + ": memfd seek failed: " + reason);
  }
  if (out_fd) {
    *out_fd = fd;
  }
  return "/proc/self/fd/" + std::to_string(fd);
#else
  (void)j;
  (void)out_fd;
  (void)label;
  (void)memfd_name;
  throw std::runtime_error(std::string(label) + ": memfd_create unavailable for ephemeral config");
#endif
}

inline void rewrite_json_memfd(int fd, const json& j, const char* label) {
  if (fd < 0) {
    throw std::runtime_error(std::string(label) + ": invalid memfd");
  }
  if (::ftruncate(fd, 0) != 0) {
    throw std::runtime_error(std::string(label) +
                             ": memfd truncate failed: " + std::strerror(errno));
  }
  if (::lseek(fd, 0, SEEK_SET) < 0) {
    throw std::runtime_error(std::string(label) + ": memfd seek failed: " + std::strerror(errno));
  }
  const std::string payload = j.dump(2);
  const char* ptr = payload.data();
  std::size_t remaining = payload.size();
  while (remaining > 0U) {
    const ssize_t wrote = ::write(fd, ptr, remaining);
    if (wrote < 0) {
      throw std::runtime_error(std::string(label) +
                               ": memfd write failed: " + std::strerror(errno));
    }
    ptr += static_cast<std::size_t>(wrote);
    remaining -= static_cast<std::size_t>(wrote);
  }
}

struct ProcessCvuRuntimeConfigOptions {
  std::string graph_family;
  int graph_id = -1;
  std::string input_dtype_default = "FP32";
  std::string output_dtype_default = "FP32";
  bool output_dtype_follows_input = false;
  bool set_tessellate = false;
  bool include_q_fields = false;
  bool include_slice_shapes = false;
  bool allow_shape_only_tensor_desc = false;
};

inline pipeline_internal::sima::stagesemantics::CompiledProcessCvuRuntimeConfig
build_processcvu_runtime_config(const json& cfg, ProcessCvuRuntimeConfigOptions options) {
  using RuntimeConfig = pipeline_internal::sima::stagesemantics::CompiledProcessCvuRuntimeConfig;
  namespace tensorsemantics = pipeline_internal::sima::tensorsemantics;

  RuntimeConfig runtime;
  runtime.graph_family = std::move(options.graph_family);
  runtime.graph_name = json_or<std::string>(cfg, "graph_name", runtime.graph_family);
  runtime.graph_id = options.graph_id;
  runtime.default_input_name = json_or<std::string>(cfg, "input_name", "input_tensor");
  const std::string output_name = json_or<std::string>(cfg, "output_name", "output_tensor");
  runtime.runtime_output_names = {output_name};
  runtime.published_output_names = {output_name};
  runtime.primary_output_name = output_name;

  runtime.input_shapes = json_shape_list_or(cfg, {"input_shapes", "input_shape"});
  runtime.output_shapes = json_shape_list_or(cfg, {"output_shapes", "output_shape"});
  if (runtime.input_shapes.empty()) {
    throw std::runtime_error("build_processcvu_runtime_config: missing input_shapes");
  }
  if (runtime.output_shapes.empty()) {
    throw std::runtime_error("build_processcvu_runtime_config: missing output_shapes");
  }
  if (options.include_slice_shapes) {
    runtime.slice_shapes = json_shape_list_or(cfg, {"slice_shapes", "slice_shape"});
    if (runtime.slice_shapes.empty()) {
      throw std::runtime_error("build_processcvu_runtime_config: missing slice_shapes");
    }
  }

  runtime.input_dtype = json_or<std::string>(cfg, "input_dtype", options.input_dtype_default);
  runtime.output_dtype = json_or<std::string>(
      cfg, "output_dtype",
      options.output_dtype_follows_input ? runtime.input_dtype : options.output_dtype_default);
  runtime.out_dtype = runtime.output_dtype;
  const std::string input_layout_raw = json_or<std::string>(cfg, "input_layout", std::string{});
  const std::string output_layout_raw = json_or<std::string>(cfg, "output_layout", std::string{});
  const std::string input_layout = tensorsemantics::normalize_layout_token(input_layout_raw);
  const std::string output_layout = tensorsemantics::normalize_layout_token(output_layout_raw);
  if ((!input_layout_raw.empty() && input_layout.empty()) ||
      (!output_layout_raw.empty() && output_layout.empty())) {
    throw std::runtime_error("build_processcvu_runtime_config: invalid input_layout/output_layout");
  }
  if (!options.allow_shape_only_tensor_desc && (input_layout.empty() || output_layout.empty())) {
    throw std::runtime_error("build_processcvu_runtime_config: explicit input_layout/output_layout "
                             "are required to synthesize typed tensors");
  }
  runtime.byte_align = json_or<int>(cfg, "byte_align", 1);
  runtime.input_stride = json_or<int>(cfg, "input_stride", 0);
  runtime.output_stride = json_or<int>(cfg, "output_stride", 0);
  runtime.input_offset = json_or<int>(cfg, "input_offset", 0);
  if (!output_layout.empty()) {
    runtime.runtime_output_logical_layout_list.assign(runtime.output_shapes.size(), output_layout);
  }

  runtime.input_tensors.reserve(runtime.input_shapes.size());
  for (const auto& shape : runtime.input_shapes) {
    sima_ev_tensor_desc desc{};
    std::string error_detail;
    const bool ok =
        input_layout.empty()
            ? tensorsemantics::build_generic_dense_tensor_desc(
                  shape, runtime.input_dtype, &desc, &error_detail,
                  "node_config_input_tensor_output_missing", "node_config_input_shape_rank_invalid",
                  "node_config_input_shape_dim_invalid", "node_config_input_dtype_invalid",
                  "node_config_input_dense_stride_output_missing")
            : tensorsemantics::build_dense_tensor_desc(
                  shape, runtime.input_dtype, input_layout, &desc, &error_detail,
                  "node_config_input_tensor_output_missing", "node_config_input_shape_rank_invalid",
                  "node_config_input_shape_dim_invalid", "node_config_input_dtype_invalid",
                  "node_config_input_dense_stride_output_missing");
    if (!ok) {
      throw std::runtime_error(
          "build_processcvu_runtime_config: could not synthesize typed input tensor" +
          (error_detail.empty() ? std::string() : std::string(": ") + error_detail));
    }
    runtime.input_tensors.push_back(desc);
  }
  runtime.output_tensors.reserve(runtime.output_shapes.size());
  for (const auto& shape : runtime.output_shapes) {
    sima_ev_tensor_desc desc{};
    std::string error_detail;
    const bool ok =
        output_layout.empty()
            ? tensorsemantics::build_generic_dense_tensor_desc(
                  shape, runtime.output_dtype, &desc, &error_detail,
                  "node_config_output_tensor_output_missing",
                  "node_config_output_shape_rank_invalid", "node_config_output_shape_dim_invalid",
                  "node_config_output_dtype_invalid",
                  "node_config_output_dense_stride_output_missing")
            : tensorsemantics::build_dense_tensor_desc(
                  shape, runtime.output_dtype, output_layout, &desc, &error_detail,
                  "node_config_output_tensor_output_missing",
                  "node_config_output_shape_rank_invalid", "node_config_output_shape_dim_invalid",
                  "node_config_output_dtype_invalid",
                  "node_config_output_dense_stride_output_missing");
    if (!ok) {
      throw std::runtime_error(
          "build_processcvu_runtime_config: could not synthesize typed output tensor" +
          (error_detail.empty() ? std::string() : std::string(": ") + error_detail));
    }
    runtime.output_tensors.push_back(desc);
  }

  if (options.include_q_fields) {
    runtime.has_q_scale = cfg.contains("q_scale");
    runtime.q_scale = json_or<double>(cfg, "q_scale", 0.0);
    runtime.has_q_zp = cfg.contains("q_zp");
    runtime.q_zp = json_or<std::int64_t>(cfg, "q_zp", 0);
    runtime.q_scale_list = json_vector_or<double>(cfg, "q_scale_array");
    runtime.q_zp_list = json_vector_or<std::int64_t>(cfg, "q_zp_array");
  }

  if (options.set_tessellate) {
    runtime.tessellate = 1;
  }
  return runtime;
}

} // namespace simaai::neat::node_helpers
