#pragma once

#include "dmabuf_test_utils.h"
#include "gst/GstInit.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorMath.h"
#include <simaai/scheduler/client/dma_memory.h>
#include "test_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sima_test {

// Independent test arithmetic: the public validation API currently checks
// composite/semantic contracts, but not dense bounds. Do not let a forged
// Storage size or an ownership flag serve as its own physical-span oracle.
inline std::size_t checked_owned_test_span(const std::vector<int64_t>& shape,
                                           const std::vector<int64_t>& strides, std::size_t element,
                                           const std::string& context) {
  require(strides.empty() || strides.size() == shape.size(), context + ": stride rank mismatch");
  require(element != 0, context + ": unknown element size");
  if (shape.empty()) {
    return 0; // Opaque payload metadata need not declare a dense view.
  }
  bool empty = false;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    require(shape[i] >= 0 && (strides.empty() || strides[i] >= 0),
            context + ": negative shape or stride");
    empty = empty || shape[i] == 0;
  }
  if (empty) {
    return 0;
  }
  std::size_t span = element;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const auto extent = static_cast<std::size_t>(shape[i]);
    if (strides.empty()) {
      require(extent <= std::numeric_limits<std::size_t>::max() / span,
              context + ": contiguous span overflow");
      span *= extent;
    } else {
      const auto stride = static_cast<std::size_t>(strides[i]);
      require(extent == 1 ||
                  stride <= (std::numeric_limits<std::size_t>::max() - span) / (extent - 1),
              context + ": strided span overflow");
      span += (extent - 1) * stride; // Zero-stride broadcast is an admitted view.
    }
  }
  return span;
}

inline void require_owned_test_binding_bounds(const simaai::neat::Tensor& tensor,
                                              std::size_t live_extent, const std::string& context) {
  require(tensor.storage && tensor.storage->size_bytes <= live_extent,
          context + ": Storage extent exceeds selected live memory");
  const auto extent = tensor.storage->size_bytes;
  const auto element = simaai::neat::pipeline_internal::dtype_bytes(tensor.dtype);
  const auto check = [&](int64_t offset, std::size_t span) {
    require(offset >= 0 && static_cast<std::size_t>(offset) <= extent &&
                span <= extent - static_cast<std::size_t>(offset),
            context + ": tensor view exceeds selected live carrier bounds");
  };
  if (tensor.is_dense()) {
    check(tensor.byte_offset,
          checked_owned_test_span(tensor.shape, tensor.strides_bytes, element, context));
    return;
  }
  require(tensor.byte_offset == 0, context + ": composite tensor has nonzero offset");
  for (const auto& plane : tensor.planes) {
    // Plane access is row-based. The column stride is optional, unlike a dense
    // tensor's rank-matched expression; Tensor::validate checks its semantics.
    if (plane.shape.size() >= 2) {
      const std::vector<int64_t> shape{plane.shape[0], plane.shape[1]};
      const std::vector<int64_t> strides =
          plane.strides_bytes.empty()
              ? std::vector<int64_t>{}
              : std::vector<int64_t>{plane.strides_bytes[0], static_cast<int64_t>(element)};
      check(plane.byte_offset, checked_owned_test_span(shape, strides, element, context));
    } else {
      check(plane.byte_offset, checked_owned_test_span(plane.shape, {}, element, context));
    }
  }
}

// StorageKind describes the carrier, not whether it borrows a producer pool.
// Check the public Owned policy separately from actual backing independence.
inline GstBuffer* require_dma_owned_output(const simaai::neat::Sample& sample,
                                           const std::vector<DmaBufBackingIdentity>& producer_ids,
                                           const std::string& context) {
  namespace neat = simaai::neat;
  namespace internal = neat::pipeline_internal;
  require(sample.owned, context + ": sample is not Owned");
  require(!internal::sample_has_device_gstsample_producer_lifetime(sample),
          context + ": Owned sample retains a producer lifetime");
  require(!internal::sample_has_zero_copy_loans(sample),
          context + ": Owned sample retains a producer loan");
  const auto tensors = neat::tensors_from_sample(sample, true);
  require(!tensors.empty(), context + ": no Owned tensors");
  GstBuffer* carrier = nullptr;
  for (const auto& tensor : tensors) {
    require(tensor.storage && tensor.storage->kind == neat::StorageKind::GstSample &&
                tensor.storage->holder,
            context + ": Owned output is not an independent DMA carrier");
    auto* holder = static_cast<GstSample*>(tensor.storage->holder.get());
    require(GST_IS_SAMPLE(holder), context + ": Owned holder is not a GstSample");
    auto* buffer = gst_sample_get_buffer(holder);
    require(buffer && gst_buffer_n_memory(buffer) > 0, context + ": Owned carrier has no payload");
    require(!gst_buffer_get_parent_buffer_meta(buffer),
            context + ": Owned carrier retains a parent buffer");
    // One independent Sample may legitimately use several carrier envelopes.
    // Validate only the binding published by this tensor, not unused siblings.
    if (!carrier) {
      carrier = buffer;
    }
    std::string diagnostic;
    require(tensor.validate(&diagnostic), context + ": invalid Owned tensor span: " + diagnostic);
    const bool whole = tensor.route.memory_index == -1 && tensor.route.segment_name.empty() &&
                       tensor.storage->size_bytes == gst_buffer_get_size(buffer);
    require(whole || (tensor.route.memory_index >= 0 &&
                      static_cast<guint>(tensor.route.memory_index) < gst_buffer_n_memory(buffer)),
            context + ": Owned selected memory index is invalid");
    const guint begin = whole ? 0 : static_cast<guint>(tensor.route.memory_index);
    const guint end = whole ? gst_buffer_n_memory(buffer) : begin + 1;
    std::size_t live_extent = 0;
    for (guint i = begin; i < end; ++i) {
      auto* memory = gst_buffer_peek_memory(buffer, i);
      const auto span = dmabuf_span(memory);
      require(span.length <= std::numeric_limits<std::size_t>::max() - live_extent,
              context + ": selected live carrier extent overflow");
      live_extent += span.length;
      const auto identity = span.backing;
      require(std::find(producer_ids.begin(), producer_ids.end(), identity) == producer_ids.end(),
              context + ": Owned output reuses a producer allocation");
    }
    require_owned_test_binding_bounds(tensor, live_extent, context);
  }
  return carrier;
}

inline std::vector<std::vector<std::uint8_t>> read_dma_carrier(GstBuffer* buffer,
                                                               const std::string& context) {
  namespace dma = simaai::scheduler::client::dma;
  std::vector<std::vector<std::uint8_t>> bytes;
  require(buffer && gst_buffer_n_memory(buffer) > 0, context + ": empty DMA carrier");
  for (guint i = 0; i < gst_buffer_n_memory(buffer); ++i) {
    dma::Error error;
    auto view = dma::DmaBufView::fromGstMemory(gst_buffer_peek_memory(buffer, i), &error);
    require(view.has_value(), context + ": invalid DMA view: " + error.message());
    auto mapping = view->map(dma::CpuAccess::Read, &error);
    require(mapping && mapping->data() && mapping->size() == view->length(),
            context + ": DMA read failed: " + error.message());
    const auto* begin = static_cast<const std::uint8_t*>(mapping->data());
    bytes.emplace_back(begin, begin + mapping->size());
    require(mapping->finish(&error), context + ": DMA read retirement failed: " + error.message());
  }
  return bytes;
}

inline void require_explicit_host_copy(const simaai::neat::Sample& sample,
                                       const std::string& context) {
  namespace neat = simaai::neat;
  for (const auto& tensor : neat::tensors_from_sample(sample, true)) {
    const auto expected =
        tensor.is_nv12() ? tensor.copy_nv12_contiguous() : tensor.copy_payload_bytes();
    for (const auto& host : {tensor.cpu(), tensor.clone()}) {
      require(host.storage && host.storage->kind == neat::StorageKind::CpuOwned &&
                  host.device.type == neat::DeviceType::CPU,
              context + ": explicit cpu()/clone() did not produce CPU-owned storage");
      const auto actual = host.is_nv12() ? host.copy_nv12_contiguous() : host.copy_payload_bytes();
      require(actual == expected, context + ": explicit host copy changed the payload");
    }
  }
}

// Observe the actual terminal producer before Core projects/copies its output.
// Snapshots retain bytes and identities only, never a GstBuffer/GstMemory loan.
// Weak notify state owns itself until producer destruction, including exceptions.
class OwnedOutputEvidence {
  struct Observation {
    GstClockTime pts;
    std::vector<DmaBufBackingIdentity> identities;
    std::vector<std::vector<std::uint8_t>> bytes;
    bool matched = false;
  };
  struct Probe {
    GstPad* pad;
    gulong id;
  };
  struct State {
    std::mutex mutex;
    std::vector<Observation> observations;
    std::vector<std::shared_ptr<std::atomic<bool>>> released;
    std::vector<Probe> probes;
    std::string failure;
    bool closing = false;
  };

  static void watch_lifetime(State& state, GstMiniObject* object) {
    auto released = std::make_shared<std::atomic<bool>>(false);
    auto context = std::make_unique<std::shared_ptr<std::atomic<bool>>>(released);
    state.released.push_back(released);
    gst_mini_object_weak_ref(
        object,
        [](gpointer data, GstMiniObject*) {
          std::unique_ptr<std::shared_ptr<std::atomic<bool>>> owner(
              static_cast<std::shared_ptr<std::atomic<bool>>*>(data));
          (*owner)->store(true);
        },
        context.release());
  }

  static void record(const std::shared_ptr<State>& state, GstBuffer* buffer) {
    require(buffer && GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer)),
            "Owned producer has no buffer/PTS");
    Observation observation{GST_BUFFER_PTS(buffer), {}, read_dma_carrier(buffer, "Owned producer")};
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closing) {
      return;
    }
    watch_lifetime(*state, GST_MINI_OBJECT(buffer));
    for (guint i = 0; i < gst_buffer_n_memory(buffer); ++i) {
      auto* memory = gst_buffer_peek_memory(buffer, i);
      observation.identities.push_back(dmabuf_identity(memory));
      // Pooled publication views can die before their actual allocation owner.
      while (memory->parent) {
        memory = memory->parent;
      }
      watch_lifetime(*state, GST_MINI_OBJECT(memory));
    }
    state->observations.push_back(std::move(observation));
  }

  static void fail(const std::shared_ptr<State>& state, const std::string& message) {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->failure = message;
  }

  static GstPadProbeReturn observe(GstPad*, GstPadProbeInfo* info, gpointer data) noexcept {
    const auto state = *static_cast<std::shared_ptr<State>*>(data);
    try {
      if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER_LIST) {
        auto* list = gst_pad_probe_info_get_buffer_list(info);
        require(list, "Owned producer has no buffer list");
        for (guint i = 0; i < gst_buffer_list_length(list); ++i) {
          record(state, gst_buffer_list_get(list, i));
        }
      } else {
        record(state, gst_pad_probe_info_get_buffer(info));
      }
    } catch (const std::exception& error) {
      fail(state, error.what());
    } catch (...) {
      fail(state, "unknown Owned producer observation failure");
    }
    return GST_PAD_PROBE_OK;
  }

  static gboolean element_added(GSignalInvocationHint*, guint count, const GValue* values,
                                gpointer data) noexcept {
    const auto state = *static_cast<std::shared_ptr<State>*>(data);
    try {
      require(count == 3, "unexpected Owned element observation signature");
      auto* element = GST_ELEMENT(g_value_get_object(&values[2]));
      auto* factory = gst_element_get_factory(element);
      if (!factory ||
          std::string(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory))) != "appsink" ||
          (std::string(GST_ELEMENT_NAME(element)) != "mysink" &&
           !std::string(GST_ELEMENT_NAME(element)).starts_with("mysink_"))) {
        return TRUE;
      }
      // Graph namespaces its terminal Output as mysink_<scope>. These tests
      // build one live producer graph per observer; verify() rejects ambiguity.
      std::unique_ptr<GstPad, decltype(&gst_object_unref)> pad(
          gst_element_get_static_pad(element, "sink"), gst_object_unref);
      require(bool(pad), "Owned terminal appsink has no sink pad");
      std::lock_guard<std::mutex> lock(state->mutex);
      if (state->closing ||
          std::any_of(state->probes.begin(), state->probes.end(),
                      [&](const Probe& probe) { return probe.pad == pad.get(); })) {
        return TRUE;
      }
      state->probes.reserve(state->probes.size() + 1);
      auto context = std::make_unique<std::shared_ptr<State>>(state);
      const auto types =
          static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_BUFFER_LIST);
      const auto id = gst_pad_add_probe(pad.get(), types, observe, context.get(), [](gpointer p) {
        delete static_cast<std::shared_ptr<State>*>(p);
      });
      require(id != 0, "could not attach Owned terminal probe");
      context.release();
      state->probes.push_back({pad.release(), id});
    } catch (const std::exception& error) {
      fail(state, error.what());
    } catch (...) {
      fail(state, "unknown Owned element observation failure");
    }
    return TRUE;
  }

public:
  OwnedOutputEvidence() : state_(std::make_shared<State>()) {
    simaai::neat::gst_init_once();
    auto* klass = g_type_class_ref(GST_TYPE_BIN);
    signal_ = g_signal_lookup("deep-element-added", GST_TYPE_BIN);
    g_type_class_unref(klass);
    require(signal_ != 0, "Owned observation signal unavailable");
    auto context = std::make_unique<std::shared_ptr<State>>(state_);
    hook_ = g_signal_add_emission_hook(signal_, 0, element_added, context.get(), [](gpointer p) {
      delete static_cast<std::shared_ptr<State>*>(p);
    });
    require(hook_ != 0, "could not install Owned terminal observation hook");
    context.release();
  }
  OwnedOutputEvidence(const OwnedOutputEvidence&) = delete;
  OwnedOutputEvidence& operator=(const OwnedOutputEvidence&) = delete;
  ~OwnedOutputEvidence() {
    detach();
  }

  void detach() noexcept {
    if (hook_) {
      g_signal_remove_emission_hook(signal_, hook_);
      hook_ = 0;
    }
    std::vector<Probe> probes;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->closing = true;
      probes.swap(state_->probes);
    }
    for (const auto& probe : probes) {
      gst_pad_remove_probe(probe.pad, probe.id);
      gst_object_unref(probe.pad);
    }
  }

  // Its snapshot completed (including checked READ END) before reaching appsink.
  void verify(const simaai::neat::Sample& sample, const std::string& context) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    require(state_->failure.empty(), context + ": " + state_->failure);
    require(state_->probes.size() == 1, context + ": ambiguous/missing terminal producer probe");
    const auto it =
        std::find_if(state_->observations.begin(), state_->observations.end(),
                     [&](const Observation& observation) {
                       return !observation.matched &&
                              observation.pts == static_cast<GstClockTime>(sample.pts_ns);
                     });
    require(it != state_->observations.end(), context + ": no matching terminal producer frame");
    std::vector<DmaBufBackingIdentity> all_ids;
    for (const auto& observation : state_->observations) {
      all_ids.insert(all_ids.end(), observation.identities.begin(), observation.identities.end());
    }
    (void)require_dma_owned_output(sample, all_ids, context);
    // Compare each selected physical payload; enclosing-envelope equality would
    // require copying unselected siblings and forbid valid separate carriers.
    std::vector<std::pair<GstBuffer*, guint>> inspected;
    for (const auto& tensor : simaai::neat::tensors_from_sample(sample, true)) {
      auto* holder = static_cast<GstSample*>(tensor.storage->holder.get());
      auto* buffer = gst_sample_get_buffer(holder);
      const bool whole = tensor.route.memory_index < 0;
      const guint begin = whole ? 0 : static_cast<guint>(tensor.route.memory_index);
      const guint end = whole ? gst_buffer_n_memory(buffer) : begin + 1;
      for (guint index = begin; index < end; ++index) {
        const auto binding = std::make_pair(buffer, index);
        if (std::find(inspected.begin(), inspected.end(), binding) != inspected.end()) {
          continue;
        }
        inspected.push_back(binding);
        namespace dma = simaai::scheduler::client::dma;
        dma::Error error;
        auto view = dma::DmaBufView::fromGstMemory(gst_buffer_peek_memory(buffer, index), &error);
        require(view.has_value(),
                context + ": invalid Owned selected DMA view: " + error.message());
        auto mapping = view->map(dma::CpuAccess::Read, &error);
        require(mapping && mapping->data() && mapping->size() == view->length(),
                context + ": Owned selected DMA read failed: " + error.message());
        const auto* begin_bytes = static_cast<const std::uint8_t*>(mapping->data());
        const std::vector<std::uint8_t> actual(begin_bytes, begin_bytes + mapping->size());
        require(mapping->finish(&error),
                context + ": Owned DMA read retirement: " + error.message());
        require(std::find(it->bytes.begin(), it->bytes.end(), actual) != it->bytes.end(),
                context + ": selected independent carrier differs from current producer bytes");
      }
    }
    it->matched = true;
    it->bytes.clear();
  }

  void require_reused_producer(const std::string& context) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    require(!state_->observations.empty(), context + ": no producer observations");
    const auto& first = state_->observations.front().identities;
    require(
        std::any_of(state_->observations.begin() + 1, state_->observations.end(),
                    [&](const Observation& later) {
                      return std::any_of(later.identities.begin(), later.identities.end(),
                                         [&](const DmaBufBackingIdentity& identity) {
                                           return std::find(first.begin(), first.end(), identity) !=
                                                  first.end();
                                         });
                    }),
        context + ": no actual producer allocation reuse while first Owned output was retained");
  }

  // Call after detaching probes AND destroying every producer Graph/Run. Buffer
  // destruction alone is not pool return, so also observe root memory.
  void require_released(std::size_t count, const std::string& context) const {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (true) {
      {
        std::lock_guard<std::mutex> lock(state_->mutex);
        require(state_->failure.empty(), context + ": " + state_->failure);
        require(state_->closing && state_->probes.empty(),
                context + ": observer is still attached");
        require(state_->observations.size() == count &&
                    std::all_of(state_->observations.begin(), state_->observations.end(),
                                [](const Observation& observation) { return observation.matched; }),
                context + ": unmatched/unexpected producer outputs");
        if (std::all_of(state_->released.begin(), state_->released.end(),
                        [](const auto& released) { return released->load(); })) {
          return;
        }
      }
      require(std::chrono::steady_clock::now() < deadline,
              context +
                  ": Owned output retained producer buffer/allocation after graph destruction");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

private:
  std::shared_ptr<State> state_;
  guint signal_ = 0;
  gulong hook_ = 0;
};

} // namespace sima_test
