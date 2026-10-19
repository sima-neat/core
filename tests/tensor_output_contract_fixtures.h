#pragma once

#include "dmabuf_test_utils.h"
#include "gst/GstInit.h"
#include "pipeline/gst/InputStreamInternal.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include "test_utils.h"
#include <simaai/scheduler/client/tensor_buffer.h>
#include <algorithm>

#include <atomic>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace sima_test::output_contract {

namespace neat = simaai::neat;
namespace internal = neat::pipeline_internal;
namespace transport = simaai::gst;
namespace client = simaai::scheduler::client;

template <typename Actual, typename Expected>
inline void require_equal(const Actual& actual, const Expected& expected,
                          const std::string& context) {
  std::ostringstream diagnostic;
  diagnostic << context << ": expected=" << expected << " actual=" << actual;
  require(actual == expected, diagnostic.str());
}

// The installed Runtime, not a hand-authored Core descriptor, is the authority
// for the physical carrier and logical binding used by each regression oracle.
inline client::TensorBufferView admitted_view(GstSample* sample, const std::string& context) {
  client::TensorBufferView view;
  std::string error;
  require(client::tensor_buffer_create_view(sample, &view, &error),
          context + ": Runtime admission: " + error);
  return view;
}

inline const transport::TensorBufferTensorView* find_logical(const client::TensorBufferView& view,
                                                             int logical_index) {
  const auto tensors = view.tensors();
  const auto found = std::find_if(tensors.begin(), tensors.end(), [=](const auto& tensor) {
    return tensor.logical_index == logical_index;
  });
  return found == tensors.end() ? nullptr : &*found;
}

inline void require_segment_table(const neat::Tensor& tensor,
                                  const client::TensorBufferView& admitted,
                                  const std::string& context) {
  require(bool(tensor.storage), context + ": missing storage");
  require_equal(tensor.storage->sima_segments.size(), admitted.segment_count(),
                context + ": physical segment count");
  for (std::size_t i = 0; i != admitted.segment_count(); ++i) {
    require_equal(tensor.storage->sima_segments[i].name, admitted.segment_name(i),
                  context + ": physical segment name " + std::to_string(i));
    require_equal(tensor.storage->sima_segments[i].size_bytes, admitted.segment_size(i),
                  context + ": live physical segment size " + std::to_string(i));
  }
}

struct Frame {
  std::shared_ptr<GstBuffer> buffer;
  std::shared_ptr<GstSample> sample;
  std::shared_ptr<std::atomic<bool>> released;
  std::vector<DmaBufSpan> dma_spans;
  std::uint8_t seed;
};

inline transport::TensorBufferPublishContract contract(bool mixed = false) {
  transport::TensorBufferPublishContract result;
  result.stage_key = "output-contract.same-stage";
  result.physical_outputs = {{0, 64, "arena0"}};
  if (mixed) {
    result.physical_outputs.push_back({1, 64, "arena1"});
  }
  for (int i = 0; i != 2; ++i) {
    transport::TensorBufferPublishLogicalOutput head;
    head.logical_index = 2 + 5 * i;
    head.backend_output_index = 11 + 2 * i;
    head.route_slot = 5 + 4 * i;
    head.memory_index = head.physical_index = mixed ? i : 0;
    head.logical_name = i == 0 ? "boxes" : "scores";
    head.backend_name = "ofm" + std::to_string(head.backend_output_index);
    head.segment_name = "arena" + std::to_string(head.memory_index);
    head.byte_offset = i == 0 ? 0 : mixed ? 8 : 16;
    head.size_bytes = 16;
    head.dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
    head.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
    head.shape = {4, 4};
    head.stride_bytes = {4, 1};
    head.quant = transport::TensorBufferQuantView{0, -1, {0.25}, {-7}};
    result.logical_outputs.push_back(std::move(head));
  }
  return result;
}

inline Frame
frame(const transport::TensorBufferPublishContract& publication, const std::vector<bool>& dma,
      std::uint8_t seed, bool compact = false, std::size_t backing_size = 64,
      const char* caps_variant = "original",
      const transport::TensorBufferPublicationOwner* frozen_owner = nullptr,
      const std::vector<transport::TensorBufferPhysicalBinding>* fixed_bindings = nullptr) {
  neat::gst_init_once();
  Frame result;
  result.seed = seed;
  result.buffer = {gst_buffer_new(), gst_buffer_unref};
  result.released = std::make_shared<std::atomic<bool>>(false);
  require(bool(result.buffer), "output fixture buffer allocation");
  for (std::size_t i = 0; i != dma.size(); ++i) {
    std::shared_ptr<GstBuffer> allocation(
        dma[i] ? allocate_cma_dmabuf(backing_size)
               : gst_buffer_new_allocate(nullptr, backing_size, nullptr),
        gst_buffer_unref);
    require(bool(allocation), "output fixture memory allocation");
    std::shared_ptr<GstSample> sample(gst_sample_new(allocation.get(), nullptr, nullptr, nullptr),
                                      gst_sample_unref);
    neat::Tensor reference;
    reference.storage = internal::make_gst_sample_storage(sample.get());
    auto storage = internal::tensor_view_from_sample_memory(reference, 0).storage;
    {
      const auto mapping = storage->map(neat::MapMode::Write);
      require(mapping.data && mapping.size_bytes >= backing_size, "output fixture mapping");
      for (std::size_t j = 0; j != backing_size; ++j) {
        static_cast<std::uint8_t*>(mapping.data)[j] = static_cast<std::uint8_t>(seed + 64 * i + j);
      }
    }
    auto* memory = gst_buffer_peek_memory(allocation.get(), 0);
    if (dma[i]) {
      result.dma_spans.push_back(dmabuf_span(memory));
    }
    gst_buffer_append_memory(result.buffer.get(), gst_memory_ref(memory));
  }
  std::vector<transport::TensorBufferPhysicalBinding> bindings;
  for (std::size_t i = 0; i != dma.size(); ++i) {
    bindings.push_back({static_cast<int>(i), 0, backing_size});
  }
  transport::TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require(transport::tensor_buffer_prepare_meta_template_from_contract(
              publication, fixed_bindings ? *fixed_bindings : bindings, &prepared, &error),
          "output fixture metadata: " + error);
  if (compact) {
    auto frozen = frozen_owner ? *frozen_owner
                               : transport::tensor_buffer_freeze_publication(prepared, &error);
    require(frozen &&
                transport::tensor_buffer_attach_publication(result.buffer.get(), frozen, &error),
            "output fixture compact metadata: " + error);
  } else {
    require(transport::tensor_buffer_attach_prepared_meta(result.buffer.get(), prepared, &error),
            "output fixture legacy metadata: " + error);
  }
  auto* released = new std::shared_ptr<std::atomic<bool>>(result.released);
  gst_mini_object_weak_ref(
      GST_MINI_OBJECT(result.buffer.get()),
      [](gpointer data, GstMiniObject*) {
        std::unique_ptr<std::shared_ptr<std::atomic<bool>>> flag(
            static_cast<std::shared_ptr<std::atomic<bool>>*>(data));
        (*flag)->store(true);
      },
      released);
  std::shared_ptr<GstCaps> caps(gst_caps_new_simple("application/vnd.simaai.tensor",
                                                    "fixture-variant", G_TYPE_STRING, caps_variant,
                                                    nullptr),
                                gst_caps_unref);
  result.sample = {gst_sample_new(result.buffer.get(), caps.get(), nullptr, nullptr),
                   gst_sample_unref};
  require(bool(result.sample), "output fixture sample allocation");
  return result;
}

inline GstBuffer* selected_buffer(const neat::Tensor& tensor, const std::string& context) {
  require(tensor.storage && tensor.storage->kind == neat::StorageKind::GstSample &&
              tensor.storage->holder,
          context + ": missing GstSample carrier");
  auto* holder = static_cast<GstSample*>(tensor.storage->holder.get());
  require(GST_IS_SAMPLE(holder), context + ": invalid GstSample carrier");
  auto* buffer = gst_sample_get_buffer(holder);
  require(buffer && tensor.route.memory_index >= 0 &&
              static_cast<guint>(tensor.route.memory_index) < gst_buffer_n_memory(buffer),
          context + ": selected memory binding out of range");
  return buffer;
}

inline GstMemory* selected_memory(const neat::Tensor& tensor, const std::string& context) {
  return gst_buffer_peek_memory(selected_buffer(tensor, context), tensor.route.memory_index);
}

inline void require_payload(const neat::Tensor& tensor, std::uint8_t seed, int physical_index,
                            std::size_t offset, std::size_t count, const std::string& context) {
  const auto mapping = tensor.view_read();
  require(mapping.data && mapping.size_bytes >= count, context + ": unavailable payload");
  for (std::size_t j = 0; j != count; ++j) {
    require(static_cast<const std::uint8_t*>(mapping.data)[j] ==
                static_cast<std::uint8_t>(seed + 64 * physical_index + offset + j),
            context + ": incorrect current-frame byte " + std::to_string(j));
  }
}

} // namespace sima_test::output_contract
