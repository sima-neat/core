#pragma once

#include "pipeline/TensorCore.h"
#include "pipeline/Run.h"
#include "pipeline/NeatError.h"
#ifdef SIMA_NEAT_INTERNAL
#include "pipeline/internal/DispatcherRecovery.h"
#include "pipeline/runtime/RunInternal.h"
#endif

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

struct SkipTest : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline bool is_long_test_context() {
  const char* v = std::getenv("SIMA_TEST_IS_LONG");
  if (!v || !*v)
    return false;
  return std::string(v) != "0";
}

inline int skip_long_test(const std::string& msg) {
  std::cout << "[SKIP-LONG] " << msg << "\n";
  return 77;
}

inline int fail_test(const std::string& msg) {
  std::cerr << "[FAIL] " << msg << "\n";
  return 1;
}

inline int skip_test(const std::string& msg) {
  if (is_long_test_context()) {
    return skip_long_test(msg);
  }
  return fail_test("[SKIP_USED] " + msg);
}

inline bool is_dispatcher_unavailable(const std::string& msg) {
#ifdef SIMA_NEAT_INTERNAL
  return simaai::neat::pipeline_internal::match_dispatcher_unavailable(msg);
#else
  (void)msg;
  return false;
#endif
}

inline void skip_test_exception(const std::string& msg) {
  if (is_long_test_context()) {
    throw SkipTest(msg);
  }
  throw std::runtime_error("[SKIP_USED] " + msg);
}

inline void skip_long_test_exception(const std::string& msg) {
  throw SkipTest(msg);
}

inline bool env_flag(const char* key, bool def = false) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def;
  return std::string(v) != "0";
}

inline bool file_exists(const std::string& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

inline void require(bool cond, const std::string& msg) {
  if (!cond)
    throw std::runtime_error(msg);
}

inline void require_contains(const std::string& haystack, const std::string& needle,
                             const std::string& msg) {
  if (haystack.find(needle) == std::string::npos) {
    throw std::runtime_error(msg + " (missing: " + needle + ")");
  }
}

inline void require_neat_error(const std::function<void()>& fn, const std::string& expected_code,
                               const std::string& what_fragment = {},
                               const std::string& note_fragment = {}) {
  try {
    fn();
    throw std::runtime_error("expected NeatError but no exception was thrown");
  } catch (const simaai::neat::NeatError& e) {
    require(e.report().error_code == expected_code,
            "unexpected NeatError.report().error_code: expected=" + expected_code +
                " actual=" + e.report().error_code);
    if (!what_fragment.empty()) {
      require_contains(std::string(e.what()), what_fragment,
                       "missing expected fragment in NeatError::what()");
    }
    if (!note_fragment.empty()) {
      require_contains(e.report().repro_note, note_fragment,
                       "missing expected fragment in GraphReport.repro_note");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("expected NeatError, got std::exception: ") + e.what());
  }
}

inline bool wait_for_reneg(simaai::neat::Run& run, std::uint64_t target, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
#ifdef SIMA_NEAT_INTERNAL
    if (simaai::neat::run_internal::input_stats(run).renegotiations >= target)
      return true;
#else
    (void)run;
    (void)target;
    return false;
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#ifdef SIMA_NEAT_INTERNAL
  return simaai::neat::run_internal::input_stats(run).renegotiations >= target;
#else
  return false;
#endif
}

inline std::uint64_t caps_changes_for(const simaai::neat::Run& run,
                                      const std::string& element_name) {
#ifdef SIMA_NEAT_INTERNAL
  const simaai::neat::RunDiagSnapshot snap = simaai::neat::run_internal::diag_snapshot(run);
  for (const auto& flow : snap.element_flows) {
    if (flow.element_name == element_name)
      return flow.caps_changes;
  }
#else
  (void)run;
  (void)element_name;
#endif
  return 0;
}

inline simaai::neat::Tensor make_nv12_tensor(int w, int h, uint8_t value = 0) {
  const std::size_t y_size = static_cast<std::size_t>(w * h);
  const std::size_t uv_size = static_cast<std::size_t>(w * h / 2);
  auto storage = simaai::neat::make_cpu_owned_storage(y_size + uv_size);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, value, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::NV12, ""};

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  simaai::neat::Plane uv;
  uv.role = simaai::neat::PlaneRole::UV;
  uv.shape = {h / 2, w};
  uv.strides_bytes = {w, 1};
  uv.byte_offset = static_cast<int64_t>(y_size);

  t.planes = {y, uv};
  return t;
}

inline simaai::neat::Tensor
make_color_tensor(int w, int h, simaai::neat::ImageSpec::PixelFormat fmt, uint8_t value = 0x11) {
  const std::size_t bytes = static_cast<std::size_t>(w * h * 3);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, value, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, 3};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{fmt, ""};
  return t;
}
