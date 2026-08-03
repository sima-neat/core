/**
 * @file SessionNaming.cpp
 * @brief Element naming/rewriting helpers for Graph.
 *
 * This is a mechanical split from the original monolithic Graph.cpp.
 * No behavior is intended to change.
 */
#include "pipeline/Graph.h"
#include "GraphDetail.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/DispatcherRecovery.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "gst/internal/GstLaunchBindings.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "builder/GraphPrinter.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

NameTransform make_name_transform(const GraphOptions& opt) {
  NameTransform out;
  out.prefix = opt.element_name_prefix;
  out.suffix = opt.element_name_suffix;
  return out;
}

namespace {

bool is_rtsp_payloader_name(std::string_view name) {
  if (name.size() < 4)
    return false;
  if (name.rfind("pay", 0) != 0)
    return false;
  for (size_t i = 3; i < name.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(name[i]);
    if (c < '0' || c > '9')
      return false;
  }
  return true;
}

} // namespace

bool name_transform_enabled(const NameTransform& t) {
  return !t.prefix.empty() || !t.suffix.empty();
}

std::string apply_name_transform(const NameTransform& t, const std::string& name) {
  if (name.empty() || !name_transform_enabled(t))
    return name;
  if (is_rtsp_payloader_name(name))
    return name;
  std::string out = name;
  if (!t.prefix.empty() && out.rfind(t.prefix, 0) != 0) {
    out = t.prefix + out;
  }
  if (!t.suffix.empty()) {
    if (out.size() < t.suffix.size() ||
        out.compare(out.size() - t.suffix.size(), t.suffix.size(), t.suffix) != 0) {
      out += t.suffix;
    }
  }
  return sanitize_name(out);
}

std::vector<std::string> apply_name_transform(const NameTransform& t,
                                              const std::vector<std::string>& names) {
  if (!name_transform_enabled(t))
    return names;
  std::vector<std::string> out;
  out.reserve(names.size());
  for (const auto& name : names) {
    out.push_back(apply_name_transform(t, name));
  }
  return out;
}

std::string rewrite_fragment_names(const std::string& fragment,
                                   const std::unordered_map<std::string, std::string>& mapping) {
  using namespace gst::launch;
  static constexpr std::string_view kAliasProperties[] = {"stage-id", "op-buff-name"};
  const Analysis analysis = analyze(fragment);
  return rewrite(fragment, analysis, mapping, kAliasProperties).text;
}

NodeFragment make_node_fragment(const std::shared_ptr<Node>& node, int index,
                                const NameTransform& transform) {
  NodeFragment out;
  if (!node)
    return out;
  const std::string original_fragment = node->backend_fragment(index);
  const auto analysis = gst::launch::analyze(original_fragment);
  std::vector<std::string> base_names = node->element_names(index);
  for (const auto* binding : gst::launch::explicit_name_bindings(analysis)) {
    if (std::find(base_names.begin(), base_names.end(), binding->canonical_value) ==
        base_names.end()) {
      base_names.push_back(binding->canonical_value);
    }
  }

  if (!name_transform_enabled(transform)) {
    out.fragment = original_fragment;
    out.element_names = std::move(base_names);
    return out;
  }

  std::unordered_map<std::string, std::string> mapping;
  mapping.reserve(base_names.size());
  out.element_names.reserve(base_names.size());
  for (const auto& base : base_names) {
    std::string renamed = apply_name_transform(transform, base);
    mapping.emplace(base, renamed);
    out.element_names.push_back(std::move(renamed));
  }
  out.fragment = rewrite_fragment_names(original_fragment, mapping);
  const auto rewritten_analysis = gst::launch::analyze(out.fragment);
  for (const auto* binding : gst::launch::explicit_name_bindings(rewritten_analysis)) {
    if (std::find(out.element_names.begin(), out.element_names.end(), binding->canonical_value) ==
        out.element_names.end()) {
      out.element_names.push_back(binding->canonical_value);
    }
  }
  return out;
}

} // namespace simaai::neat
