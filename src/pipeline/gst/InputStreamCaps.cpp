#include "InputStreamInternal.h"

namespace simaai::neat {
namespace {

[[noreturn]] void throw_shape_change_requires_rebuild(InputStream::State& st, const char* where,
                                                      const char* reason) {
  st.renegotiation_blocked.fetch_add(1, std::memory_order_relaxed);
  const char* tag = where ? where : "InputStream";
  std::ostringstream oss;
  oss << tag << ": shape_change_requires_rebuild";
  if (reason && *reason) {
    oss << " (" << reason << ")";
  }
  throw std::runtime_error(oss.str());
}

const char* sample_spec_kind_name(SampleMediaKind kind) {
  switch (kind) {
  case SampleMediaKind::RawVideo:
    return "video/x-raw";
  case SampleMediaKind::Tensor:
    return "application/vnd.simaai.tensor";
  case SampleMediaKind::Encoded:
    return "encoded";
  }
  return "unknown";
}

std::string sample_spec_payload_summary(const SampleSpec& spec) {
  std::ostringstream oss;
  oss << "kind=" << sample_spec_kind_name(spec.kind);
  if (!spec.caps_string.empty()) {
    oss << " caps=\"" << spec.caps_string << "\"";
  } else if (!spec.media_type.empty() || !spec.format.empty()) {
    oss << " media=" << spec.media_type << " format=" << spec.format;
  }
  if (!spec.shape.empty()) {
    oss << " shape=[";
    for (std::size_t i = 0; i < spec.shape.size(); ++i) {
      if (i) {
        oss << ",";
      }
      oss << spec.shape[i];
    }
    oss << "]";
  } else if (spec.width > 0 || spec.height > 0 || spec.depth > 0) {
    oss << " shape=" << spec.width << "x" << spec.height << "x" << spec.depth;
  }
  if (spec.required_bytes_actual > 0U) {
    oss << " bytes=" << spec.required_bytes_actual;
  }
  return oss.str();
}

} // namespace

bool apply_caps_or_throw(InputStream::State& st, const SampleSpec& caps_spec, const char* where) {
  const char* tag = where ? where : "InputStream::apply_caps";
  if (!st.appsrc) {
    throw std::runtime_error(std::string(tag) + ": appsrc not available");
  }
  if (!st.src_opt.caps_override.empty()) {
    st.renegotiation_blocked.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error(std::string(tag) + ": caps_override is set; renegotiation disabled");
  }
  ensure_alloc_for_bytes(st, caps_spec.required_bytes_actual, tag);

  const CapKey& key = caps_spec.caps_key;
  const bool had_key = st.current_key.has_value();
  const bool changed = !had_key || (st.current_key && *st.current_key != key);
  if (had_key && changed) {
    st.renegotiations.fetch_add(1, std::memory_order_relaxed);
  }
  if (changed && st.on_caps_change && st.last_spec.has_value()) {
    st.on_caps_change(*st.last_spec, caps_spec);
  }

  GstCaps* caps = caps_from_spec(caps_spec);
  gst_app_src_set_caps(GST_APP_SRC(st.appsrc), caps);
  if (st.current_caps) {
    gst_caps_unref(st.current_caps);
  }
  st.current_caps = gst_caps_ref(caps);
  if (caps_spec.kind == SampleMediaKind::RawVideo) {
    gst_video_info_from_caps(&st.current_vinfo, caps);
  } else {
    std::memset(&st.current_vinfo, 0, sizeof(st.current_vinfo));
  }
  gst_caps_unref(caps);

  st.current_key = key;
  st.last_spec = caps_spec;
  st.pending_key.reset();
  st.pending_count = 0;
  return true;
}

CapsDecision maybe_update_caps_for_spec(InputStream::State& st, const SampleSpec& spec,
                                        const char* where) {
  SampleSpec caps_spec = spec;
  if (caps_spec.kind == SampleMediaKind::RawVideo) {
    caps_spec.fps_n = st.src_opt.fps_n;
    caps_spec.fps_d = st.src_opt.fps_d;
  }
  caps_spec.caps_key = capkey_from_spec(caps_spec);
  caps_spec.caps_string = caps_string_from_spec(caps_spec);
  CapKey key = caps_spec.caps_key;

  if (spec.kind == SampleMediaKind::Encoded) {
    if (!st.current_key.has_value()) {
      apply_caps_or_throw(st, caps_spec, where);
      return CapsDecision::Push;
    }
    if (*st.current_key != key) {
      st.renegotiation_blocked.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error(std::string(where ? where : "InputStream") +
                               ": encoded caps change not supported");
    }
    return CapsDecision::Push;
  }

  if (spec.kind == SampleMediaKind::Tensor) {
    if (st.current_tensor_spec.has_value() && !tensor_spec_matches(*st.current_tensor_spec, spec)) {
      st.renegotiation_blocked.fetch_add(1, std::memory_order_relaxed);
      std::ostringstream oss;
      oss << (where ? where : "InputStream") << ": tensor changes are not supported"
          << " current_key="
          << (st.current_key.has_value() ? st.current_key->to_string() : std::string("<none>"))
          << " new_key=" << key.to_string() << " current_tensor{w=" << st.current_tensor_spec->width
          << ",h=" << st.current_tensor_spec->height << ",d=" << st.current_tensor_spec->depth
          << ",shape=[";
      for (std::size_t i = 0; i < st.current_tensor_spec->shape.size(); ++i) {
        if (i) {
          oss << ",";
        }
        oss << st.current_tensor_spec->shape[i];
      }
      oss << "]} new_tensor{w=" << spec.width << ",h=" << spec.height << ",d=" << spec.depth
          << ",shape=[";
      for (std::size_t i = 0; i < spec.shape.size(); ++i) {
        if (i) {
          oss << ",";
        }
        oss << spec.shape[i];
      }
      oss << "]}";
      throw std::runtime_error(oss.str());
    }
    if (!st.current_key.has_value()) {
      const bool applied = apply_caps_or_throw(st, caps_spec, where);
      if (applied) {
        st.current_tensor_spec = spec;
      }
      return CapsDecision::Push;
    }
    if (*st.current_key != key) {
      st.renegotiation_blocked.fetch_add(1, std::memory_order_relaxed);
      std::ostringstream oss;
      oss << (where ? where : "InputStream") << ": tensor caps change not supported"
          << " current_key=" << st.current_key->to_string() << " new_key=" << key.to_string();
      if (st.current_tensor_spec.has_value()) {
        oss << " current_tensor{w=" << st.current_tensor_spec->width
            << ",h=" << st.current_tensor_spec->height << ",d=" << st.current_tensor_spec->depth
            << ",shape=[";
        for (std::size_t i = 0; i < st.current_tensor_spec->shape.size(); ++i) {
          if (i) {
            oss << ",";
          }
          oss << st.current_tensor_spec->shape[i];
        }
        oss << "]}";
      }
      oss << " new_tensor{w=" << spec.width << ",h=" << spec.height << ",d=" << spec.depth
          << ",shape=[";
      for (std::size_t i = 0; i < spec.shape.size(); ++i) {
        if (i) {
          oss << ",";
        }
        oss << spec.shape[i];
      }
      oss << "]}";
      throw std::runtime_error(oss.str());
    }
    if (!st.current_tensor_spec.has_value()) {
      st.current_tensor_spec = spec;
    }
    return CapsDecision::Push;
  }

  if (st.current_tensor_spec.has_value() && spec.kind != SampleMediaKind::Tensor) {
    st.renegotiation_blocked.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream oss;
    oss << (where ? where : "InputStream")
        << ": input payload kind/caps mismatch. This stream was built for tensor input; "
        << "expected " << sample_spec_payload_summary(*st.current_tensor_spec) << ", received "
        << sample_spec_payload_summary(caps_spec)
        << ". If you are feeding a cv::Mat or RGB/BGR/GRAY image Tensor, enable Generic "
           "Preproc with Model::Options::preprocess.kind=InputKind::Image and "
           "preprocess.enable=AutoFlag::On; otherwise feed a Tensor that already matches "
           "the expected tensor caps.";
    throw std::runtime_error(oss.str());
  }

  if (st.current_tensor_spec.has_value()) {
    st.current_tensor_spec.reset();
  }

  if (!st.current_key.has_value()) {
    apply_caps_or_throw(st, caps_spec, where);
    return CapsDecision::Push;
  }
  if (*st.current_key == key) {
    st.pending_key.reset();
    st.pending_count = 0;
    if (st.pending_buffer) {
      discard_pending_buffer(st, "pending_regime_cancelled");
    }
    return CapsDecision::Push;
  }

  const CapKey& cur = *st.current_key;
  const bool non_geometry_change = cur.media_type != key.media_type || cur.format != key.format ||
                                   cur.fps_n != key.fps_n || cur.fps_d != key.fps_d;
  const bool format_only_change = (cur.media_type == key.media_type) && (cur.fps_n == key.fps_n) &&
                                  (cur.fps_d == key.fps_d) && (cur.format != key.format);
  // CVU ingress-dynamic path currently consumes width/height/stride/format from caps.
  // Keep media/fps transitions blocked in this mode until plugin+pipeline support is
  // validated end-to-end; fully-dynamic graphs may allow broader renegotiation.
  const bool allow_ingress_cvu_non_geometry =
      (st.dynamic_capability == InputStreamOptions::DynamicCapability::IngressDynamicCvuOnly) &&
      st.allow_ingress_cvu_format_renegotiation && format_only_change;

  if (st.shape_policy == InputStreamOptions::ShapePolicy::LockedByCapsOverride ||
      !st.src_opt.caps_override.empty()) {
    st.renegotiation_blocked.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error(std::string(where ? where : "InputStream") +
                             ": caps_override_blocks_renegotiation");
  }

  if (st.dynamic_capability == InputStreamOptions::DynamicCapability::StaticOnly) {
    throw_shape_change_requires_rebuild(st, where, "dynamic capability is static-only");
  }

  if (st.dynamic_capability == InputStreamOptions::DynamicCapability::IngressDynamicCvuOnly &&
      non_geometry_change && !allow_ingress_cvu_non_geometry) {
    throw_shape_change_requires_rebuild(
        st, where, "only raw-video geometry or format changes are allowed in this mode");
  }

  if (non_geometry_change &&
      st.dynamic_capability != InputStreamOptions::DynamicCapability::FullyDynamic &&
      !allow_ingress_cvu_non_geometry) {
    throw_shape_change_requires_rebuild(st, where, "format/media/fps change is not supported");
  }

  const int stable = st.opt.stability_frames;
  const bool stability_gate_applies = !non_geometry_change;
  if (stable > 1 && stability_gate_applies) {
    if (!st.pending_key.has_value() || *st.pending_key != key) {
      st.pending_key = key;
      st.pending_count = 1;
      return CapsDecision::Queue;
    }
    st.pending_count += 1;
    if (st.pending_count < stable)
      return CapsDecision::Queue;
    apply_caps_or_throw(st, caps_spec, where);
    return CapsDecision::Flush;
  }

  apply_caps_or_throw(st, caps_spec, where);
  return CapsDecision::Push;
}
} // namespace simaai::neat
