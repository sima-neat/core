#include "model/internal/ModelInternal.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/StageInputCacheIdentity.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include "test_main.h"

#include <gst/gst.h>

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

using namespace simaai::neat;

namespace {

Tensor head(int index, const std::shared_ptr<TensorBuffer>& storage) {
  Tensor tensor;
  tensor.storage = storage;
  tensor.dtype = TensorDType::Int8;
  tensor.layout = TensorLayout::HW;
  tensor.shape = {2, 4};
  tensor.strides_bytes = {4, 1};
  tensor.route.stage_key = "producer";
  tensor.route.logical_index = index;
  tensor.route.physical_index = index;
  tensor.route.memory_index = 0;
  tensor.route.route_slot = index;
  tensor.route.name = "head" + std::to_string(index);
  tensor.route.backend_name = tensor.route.name;
  tensor.route.segment_name = "carrier";
  return tensor;
}

Sample inputs(bool shared = false, std::size_t capacity = 64) {
  auto first = make_cpu_owned_storage(capacity);
  auto second = shared ? first : make_cpu_owned_storage(capacity);
  return sample_from_tensors({head(0, first), head(1, second)});
}

pipeline_internal::StageInputCacheIdentity
identity(const Sample& sample, pipeline_internal::StageInputCacheGeometryPolicy policy =
                                   pipeline_internal::StageInputCacheGeometryPolicy::Exact) {
  std::string reason;
  auto key = pipeline_internal::stage_input_cache_identity(sample, &reason, policy);
  require(key.has_value(), "expected cache-eligible fixture: " + reason);
  return *key;
}

void ensure_gst_ready() {
  gst_init(nullptr, nullptr);
  const gchar* tags[] = {nullptr};
  for (const auto* name : {"GstSimaSampleMeta", "GstSimaMeta", SIMA_TENSOR_SET_META_NAME})
    if (!gst_meta_get_info(name))
      (void)gst_meta_register_custom(name, tags, nullptr, nullptr, nullptr);
}

// Check actual existing transport projection, separately from the source-layout key. Key
// generation itself must never invoke this helper or materialize/map source payloads.
using ProjectedHead = std::tuple<int, std::int64_t, std::uint64_t, std::vector<std::int64_t>,
                                 std::vector<std::int64_t>>;
std::vector<ProjectedHead> projection(const Sample& input) {
  std::string error;
  auto holder = pipeline_internal::sample_to_gst_envelope_holder(input, &error);
  require(holder != nullptr, "could not project fixture: " + error);
  pipeline_internal::TensorBufferView view;
  require(pipeline_internal::tensor_buffer_descriptor_from_sample(
              static_cast<GstSample*>(holder.get()), &view, &error),
          "could not read projected fixture: " + error);
  std::vector<ProjectedHead> heads;
  for (const auto& tensor : view.tensors)
    heads.emplace_back(tensor.memory_index, tensor.byte_offset, tensor.size_bytes, tensor.shape,
                       tensor.stride_bytes);
  return heads;
}

Sample gst_input(std::size_t capacity = 64, std::uint64_t binding_offset = 0) {
  auto source = inputs(true, capacity);
  source.tensors.resize(1);
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, capacity, nullptr);
  require(buffer != nullptr, "could not allocate Gst fixture");
  pipeline_internal::attach_tensor_set_meta_from_tensors(buffer, source.tensors);
  std::string error;
  const bool attached =
      simaai::gst::tensor_buffer_attach_physical_bindings(buffer, {{0, binding_offset, 8}}, &error);
  require(attached, "could not attach physical binding fixture: " + error);
  GstCaps* caps = gst_caps_from_string("application/vnd.simaai.tensor");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_buffer_unref(buffer);
  gst_caps_unref(caps);
  require(sample != nullptr, "could not allocate GstSample fixture");
  source.tensors.front().storage = pipeline_internal::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  return source;
}

enum class GstCarrierTopology { SharedBuffer, SeparateBuffersSharedMemory, IndependentCarriers };

Sample gst_six_heads(bool shared_wrapper,
                     GstCarrierTopology topology = GstCarrierTopology::SharedBuffer) {
  constexpr std::size_t capacity = 64;
  require(!shared_wrapper || topology == GstCarrierTopology::SharedBuffer,
          "shared six-head wrapper requires one actual GstBuffer");
  const auto cpu_storage = make_cpu_owned_storage(capacity);
  TensorList tensors;
  for (int i = 0; i < 6; ++i) {
    Tensor tensor = head(i, cpu_storage);
    tensor.byte_offset = tensor.route.physical_byte_offset = i * 8;
    tensor.route.physical_index = 0;
    tensors.push_back(std::move(tensor));
  }
  Sample source = sample_from_tensors(tensors);
  GstBuffer* parent = gst_buffer_new_allocate(nullptr, capacity, nullptr);
  require(parent != nullptr, "could not allocate six-head parent fixture");
  GstCaps* caps = gst_caps_from_string("application/vnd.simaai.tensor");
  require(caps != nullptr, "could not allocate six-head caps fixture");
  const auto attach_contract = [&](GstBuffer* buffer) {
    // Author identical complete descriptor/binding facts on every actual carrier. Only
    // buffer/memory sharing varies; no metadata projection or payload mapping is needed.
    pipeline_internal::attach_tensor_set_meta_from_tensors(buffer, tensors);
    std::string error;
    require(simaai::gst::tensor_buffer_attach_physical_bindings(buffer, {{0, 0, capacity}}, &error),
            "could not attach six-head physical bindings: " + error);
  };
  // Attach once while writable, before separate Sample holders reference this buffer.
  attach_contract(parent);
  std::shared_ptr<TensorBuffer> common_storage;
  for (auto& tensor : source.tensors) {
    if (shared_wrapper && common_storage) {
      tensor.storage = common_storage;
      continue;
    }
    GstBuffer* buffer = nullptr;
    if (topology == GstCarrierTopology::SharedBuffer) {
      buffer = gst_buffer_ref(parent);
    } else if (topology == GstCarrierTopology::SeparateBuffersSharedMemory) {
      buffer = gst_buffer_new();
      require(buffer != nullptr, "could not allocate shared-memory envelope fixture");
      gst_buffer_append_memory(buffer, gst_memory_ref(gst_buffer_peek_memory(parent, 0)));
    } else {
      buffer = gst_buffer_new_allocate(nullptr, capacity, nullptr);
      require(buffer != nullptr, "could not allocate independent six-head carrier fixture");
    }
    if (topology != GstCarrierTopology::SharedBuffer)
      attach_contract(buffer);
    GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
    gst_buffer_unref(buffer);
    require(sample != nullptr, "could not allocate six-head GstSample fixture");
    tensor.storage = pipeline_internal::make_gst_sample_storage(sample);
    gst_sample_unref(sample);
    require(tensor.storage != nullptr, "could not create six-head storage fixture");
    if (shared_wrapper)
      common_storage = tensor.storage;
  }
  gst_buffer_unref(parent);
  gst_caps_unref(caps);
  return source;
}

void expect_different(const Sample& original, const std::function<void(Sample&)>& change,
                      const std::string& description) {
  Sample changed = original;
  change(changed);
  require(identity(original) != identity(changed), description + " must change compatibility");
}

} // namespace

RUN_TEST(
    "unit_stage_cache_identity_test", ([] {
      ensure_gst_ready();
      const Sample original = inputs();
      const Sample fresh = inputs();
      auto six_heads = [](bool shared) {
        const auto parent = make_cpu_owned_storage(64);
        TensorList heads;
        for (int i = 0; i < 6; ++i) {
          Tensor tensor = head(i, shared ? parent : make_cpu_owned_storage(64));
          tensor.byte_offset = i * 8;
          tensor.route.physical_index = 0;
          heads.push_back(std::move(tensor));
        }
        return sample_from_tensors(heads);
      };
      require(identity(six_heads(true)) == identity(six_heads(true)),
              "six shared heads must reuse across new producer allocations");
      require(identity(six_heads(true)) != identity(six_heads(false)),
              "six shared heads must not alias six independent carriers");
      auto aliased_holders = [](bool shared_owner) {
        const auto first = make_cpu_owned_storage(64);
        const auto second = shared_owner ? first : make_cpu_owned_storage(64);
        auto* first_data = static_cast<std::uint8_t*>(first->data);
        auto* second_data = static_cast<std::uint8_t*>(second->data) + 8;
        auto first_holder = std::shared_ptr<void>(first->holder, first_data);
        auto second_holder = std::shared_ptr<void>(second->holder, second_data);
        return sample_from_tensors(
            {head(0, make_cpu_external_storage(first_data, 64, first_holder)),
             head(1, make_cpu_external_storage(second_data, 56, second_holder))});
      };
      require(identity(aliased_holders(true)) == identity(aliased_holders(true)),
              "alias-view owner partitions must be independent of addresses");
      require(identity(aliased_holders(true)) != identity(aliased_holders(false)),
              "different alias pointers sharing one owner must not become split owners");
      require(identity(original) == identity(fresh),
              "new addresses with the same full source layout must reuse");
      require(identity(inputs(true)) != identity(original),
              "shared and independent carriers must not alias");
      require(identity(inputs(false, 80)) != identity(original),
              "exact source capacity changes must miss");

      expect_different(
          original, [](Sample& s) { s.stream_label += "_alternate"; },
          "Sample field routing label");
      expect_different(
          original, [](Sample& s) { s.tensors[1].byte_offset = 8; }, "later head offset");
      expect_different(
          original, [](Sample& s) { s.tensors[1].strides_bytes = {8, 1}; }, "later head strides");
      expect_different(
          original, [](Sample& s) { s.tensors[1].shape = {1, 4}; }, "later head shape");
      expect_different(
          original, [](Sample& s) { s.tensors[1].route.memory_index = 1; },
          "later head memory identity");
      expect_different(
          original, [](Sample& s) { s.tensors[1].semantic.quant = QuantSpec{}; },
          "later head quantization presence");
      Sample quantized = inputs();
      quantized.tensors[1].semantic.quant = QuantSpec{};
      expect_different(
          quantized, [](Sample& s) { s.tensors[1].semantic.quant->scale = 0.5f; },
          "later head quantization scale");

      Sample dynamic = fresh;
      dynamic.frame_id = 19;
      dynamic.stream_id = "different_dynamic_credit_stream";
      dynamic.pts_ns = 123456;
      dynamic.attributes = {{"frame", "payload"}};
      require(identity(original) == identity(dynamic),
              "credit stream, attributes and timestamps are not prepared layout");

      Sample geometry = fresh;
      geometry.tensors[0].semantic.preprocess = PreprocessRuntimeMeta{};
      geometry.tensors[0].semantic.preprocess->original_width = 640;
      geometry.tensors[0].semantic.preprocess->original_height = 480;
      geometry.tensors[0].semantic.preprocess->resize_mode = "letterbox";
      Sample changed_original = geometry;
      changed_original.tensors[0].semantic.preprocess->original_width = 1920;
      require(identity(geometry) != identity(changed_original),
              "exact/SuperPoint preparation must include original image size");
      const auto policy = pipeline_internal::StageInputCacheGeometryPolicy::DynamicOriginalSize;
      require(identity(geometry, policy) == identity(changed_original, policy),
              "qualified dynamic BoxDecode original-size resolver may reuse its stage");
      Sample changed_mode = geometry;
      changed_mode.tensors[0].semantic.preprocess->resize_mode = "stretch";
      require(identity(geometry, policy) != identity(changed_mode, policy),
              "resize mode remains a checked preparation discriminator");
      Sample changed_model = geometry;
      changed_model.tensors[0].semantic.preprocess->scaled_width = 320;
      require(identity(geometry, policy) != identity(changed_model, policy),
              "geometry used to derive model dimensions must remain in compatibility");
      Sample changed_padding = geometry;
      changed_padding.tensors[0].semantic.preprocess->pad_left = 8;
      require(identity(geometry, policy) != identity(changed_padding, policy),
              "padding used to derive model dimensions must remain in compatibility");

      int map_calls = 0;
      Sample guarded = inputs();
      for (auto& tensor : guarded.tensors)
        tensor.storage->map_fn = [&map_calls](MapMode) -> Mapping {
          ++map_calls;
          throw std::runtime_error("cache identity mapped payload");
        };
      pipeline_internal::reset_tensor_io_stats();
      (void)identity(guarded);
      const auto stats = pipeline_internal::snapshot_tensor_io_stats();
      require(map_calls == 0 && stats.tensor_copy_count == 0 && stats.gst_memory_map_calls == 0,
              "cache identity must not map or copy payloads");

      std::weak_ptr<void> weak_owner;
      pipeline_internal::StageInputCacheIdentity retained_key;
      {
        Sample temporary = inputs();
        weak_owner = temporary.tensors[0].storage->holder;
        retained_key = identity(temporary);
      }
      require(weak_owner.expired() && !retained_key.value.empty(),
              "stored key must not retain input ownership");

      Sample unsupported = inputs();
      unsupported.tensors[1].storage = std::make_shared<TensorBuffer>();
      unsupported.tensors[1].storage->kind = StorageKind::DeviceHandle;
      std::string reason;
      require(!pipeline_internal::stage_input_cache_identity(unsupported, &reason) &&
                  !reason.empty(),
              "unknown representations must bypass cache rather than reuse partial keys");
      Sample invalid = original;
      invalid.tensors[1].strides_bytes = {4};
      bool rejected = false;
      try {
        (void)identity(invalid);
      } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("stride rank") != std::string::npos;
      }
      require(rejected, "invalid metadata must preserve an actionable diagnostic");

      // Historical producer offsets remain conservative source facts. Independent Owned
      // heads are rebased by execution ingress; strided CPU heads are packed densely.
      Sample owned_a = inputs();
      for (auto& tensor : owned_a.tensors) {
        tensor.byte_offset = 8;
        tensor.route.physical_byte_offset = 256;
        tensor.strides_bytes = {12, 1};
      }
      Sample owned_b = inputs();
      for (auto& tensor : owned_b.tensors) {
        tensor.byte_offset = 8;
        tensor.route.physical_byte_offset = 256;
        tensor.strides_bytes = {12, 1};
      }
      require(identity(owned_a) == identity(owned_b), "Owned layout equality must be stable");
      const auto projected_a = projection(owned_a);
      require(projected_a == projection(owned_b),
              "equal supported Owned source keys must have equal canonical projections");
      for (const auto& entry : projected_a) {
        require(std::get<1>(entry) == 0, "Owned source offsets must not be applied twice");
        require(std::get<4>(entry) == std::vector<std::int64_t>({4, 1}),
                "non-contiguous Owned inputs must project dense transport strides");
      }

      const Sample gst_a = gst_input();
      const Sample gst_b = gst_input();
      Sample gst_geometry = gst_input();
      auto* geometry_sample =
          static_cast<GstSample*>(gst_geometry.tensors[0].storage->holder.get());
      PreprocessRuntimeMeta producer_geometry;
      producer_geometry.original_width = 640;
      producer_geometry.original_height = 480;
      producer_geometry.resize_mode = "letterbox";
      require(simaai::neat::write_simaai_preprocess_meta(gst_sample_get_buffer(geometry_sample),
                                                         producer_geometry),
              "could not write holder preprocessing fixture");
      const auto holder_key = identity(gst_geometry, policy);
      require(identity(gst_a, policy) != holder_key,
              "holder preprocessing authority must not be omitted");
      producer_geometry.original_width = 1920;
      require(simaai::neat::write_simaai_preprocess_meta(gst_sample_get_buffer(geometry_sample),
                                                         producer_geometry),
              "could not change holder original-size fixture");
      require(identity(gst_geometry, policy) == holder_key,
              "qualified original-size reuse must also cover actual holder metadata");
      producer_geometry.resize_mode = "stretch";
      require(simaai::neat::write_simaai_preprocess_meta(gst_sample_get_buffer(geometry_sample),
                                                         producer_geometry),
              "could not change holder resize-mode fixture");
      require(identity(gst_geometry, policy) != holder_key,
              "actual holder resize mode remains a checked preparation fact");
      pipeline_internal::reset_tensor_io_stats();
      require(identity(gst_a) == identity(gst_b),
              "Gst carrier identity must not include allocation addresses");
      const auto gst_stats = pipeline_internal::snapshot_tensor_io_stats();
      require(gst_stats.tensor_copy_count == 0 && gst_stats.gst_memory_map_calls == 0,
              "Gst metadata lookup must not map or copy");
      require(identity(gst_a) != identity(gst_input(80)),
              "actual Gst visible capacity must enter compatibility");
      require(identity(gst_a) != identity(gst_input(64, 8)),
              "physical binding offset must enter compatibility");

      const Sample gst_shared_wrapper = gst_six_heads(true);
      const Sample gst_separate_wrappers = gst_six_heads(false);
      require(gst_separate_wrappers.tensors[0].storage != gst_separate_wrappers.tensors[1].storage,
              "cold six-head fixture must have separate TensorBuffer wrappers");
      const auto& holder_a = gst_separate_wrappers.tensors[0].storage->holder;
      const auto& holder_b = gst_separate_wrappers.tensors[1].storage->holder;
      const std::owner_less<std::shared_ptr<void>> owner_less;
      require(owner_less(holder_a, holder_b) || owner_less(holder_b, holder_a),
              "cold six-head fixture must have separate holder control blocks");
      require(gst_sample_get_buffer(static_cast<GstSample*>(holder_a.get())) ==
                  gst_sample_get_buffer(static_cast<GstSample*>(holder_b.get())),
              "cold six-head wrappers must describe one actual GstBuffer carrier");
      pipeline_internal::reset_tensor_io_stats();
      require(identity(gst_shared_wrapper) == identity(gst_separate_wrappers),
              "incidental Gst TensorBuffer/holder partitions must not split equivalent carriers");
      const auto normalized_gst_stats = pipeline_internal::snapshot_tensor_io_stats();
      require(normalized_gst_stats.tensor_copy_count == 0 &&
                  normalized_gst_stats.gst_memory_map_calls == 0,
              "normalized Gst physical identity must remain metadata-only");
      const Sample gst_separate_buffers =
          gst_six_heads(false, GstCarrierTopology::SeparateBuffersSharedMemory);
      const Sample gst_independent_carriers =
          gst_six_heads(false, GstCarrierTopology::IndependentCarriers);
      require(identity(gst_shared_wrapper) != identity(gst_separate_buffers),
              "different actual GstBuffer partitions must remain conservative cache misses");
      require(identity(gst_separate_buffers) != identity(gst_independent_carriers),
              "shared and independent actual GstMemory partitions must not alias");

      const Model::Options defaults;
      const auto default_key = internal::model_stage_options_cache_identity(defaults);
      auto different_option = [&](const std::function<void(Model::Options&)>& change) {
        Model::Options changed = defaults;
        change(changed);
        require(internal::model_stage_options_cache_identity(changed) != default_key,
                "stage-affecting options must change the model identity");
      };
      different_option([](auto& o) { o.decode_type_option = BoxDecodeTypeOption::PackedPerHead; });
      different_option([](auto& o) { o.score_threshold = 0.25f; });
      different_option([](auto& o) { o.nms_iou_threshold = 0.5f; });
      different_option([](auto& o) { o.top_k = 10; });
      different_option([](auto& o) { o.num_classes = 80; });
      different_option([](auto& o) { o.superpoint.profile = SuperPointProfile::LightGlueV1; });
      different_option([](auto& o) { o.superpoint.nms_radius = 0; });
      different_option([](auto& o) { o.superpoint.border_margin = 0; });
      different_option([](auto& o) { o.superpoint.descriptor_output_dtype = TensorDType::Int8; });
      different_option([](auto& o) {
        o.superpoint.output_format = SuperPointOutputFormat::LegacyA65InterleavedV0;
      });
      different_option([](auto& o) { o.advanced_execution.preprocess_target = "EV74"; });
      different_option([](auto& o) { o.advanced_execution.postprocess_target = "A65"; });
      different_option([](auto& o) { o.advanced_execution.preprocess_async = false; });
      different_option([](auto& o) { o.advanced_execution.inference_async = false; });
      different_option([](auto& o) { o.advanced_execution.inference_output_buffers = 2; });
      different_option([](auto& o) { o.advanced_execution.defer_output_cache_sync = false; });
      different_option([](auto& o) { o.advanced_execution.internal_queue_depth = 2; });
      different_option([](auto& o) {
        PreparedRunnerOptions runner;
        runner.profile = true;
        o.advanced_execution.prepared_runner = runner;
      });
      Model::Options legacy = defaults;
      legacy.processmla.async = false;
      Model::Options overlay = defaults;
      overlay.advanced_execution.inference_async = false;
      require(internal::model_stage_options_cache_identity(legacy) ==
                  internal::model_stage_options_cache_identity(overlay),
              "equivalent effective legacy/advanced options should have equal identities");
      Model::Options quiet = defaults;
      quiet.verbose = VerboseOptions::quiet();
      require(internal::model_stage_options_cache_identity(quiet) == default_key,
              "verbosity must not alter stage preparation identity");
      require(default_key.find("\"score_threshold\":0.0") != std::string::npos &&
                  default_key.find("\"nms_iou_threshold\":0.0") != std::string::npos,
              "valid zero thresholds must be explicitly serialized");
    }));
