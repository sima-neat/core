#include "pipeline/gst/InputStreamInternal.h"
#include "pipeline/internal/SampleUtil.h"
#include "test_main.h"

#include <gstsimaaitensorbuffer_meta.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace {
using namespace simaai::neat;
using namespace simaai::gst;
TensorBufferPreparedMetaTemplate prepare() {
  TensorBufferPublishContract contract;
  contract.stage_key = "core-decode-test";
  std::vector<TensorBufferPhysicalBinding> bindings;
  for (int i = 0; i < 2; ++i) {
    const auto name = "arena" + std::to_string(i);
    contract.physical_outputs.push_back({i, 64, name});
    bindings.push_back({i, 0, 64});
    TensorBufferPublishLogicalOutput tensor;
    tensor.logical_index = i;
    tensor.physical_index = i;
    tensor.memory_index = i;
    tensor.backend_output_index = i;
    tensor.route_slot = i;
    tensor.logical_name = "tensor" + std::to_string(i);
    tensor.backend_name = tensor.logical_name;
    tensor.segment_name = name;
    tensor.byte_offset = 4;
    tensor.size_bytes = 4;
    tensor.dtype = SIMA_TENSOR_SET_DTYPE_INT8_V1;
    tensor.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
    tensor.shape = {2, 2};
    tensor.stride_bytes = {8, 1};
    tensor.quant = TensorBufferQuantView{1, 1, {0.25, 0.5}, {-3, 7}};
    contract.logical_outputs.push_back(tensor);
  }
  TensorBufferPreparedMetaTemplate result;
  std::string error;
  require(tensor_buffer_prepare_meta_template_from_contract(contract, bindings, &result, &error),
          error);
  return result;
}
std::shared_ptr<GstBuffer> buffer() {
  auto result = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
  for (int i = 0; i < 2; ++i)
    gst_buffer_append_memory(result.get(), gst_allocator_alloc(nullptr, 64, nullptr));
  return result;
}
Sample decode(InputStream::State& state, GstBuffer* buffer, int width, bool write_legacy = true) {
  InputOptions options;
  PreprocessMetaTemplate meta;
  meta.enabled = true;
  meta.target_width = 640;
  meta.target_height = 640;
  meta.resize_mode = "letterbox";
  options.preprocess_meta = meta;
  if (write_legacy)
    require(apply_simaai_preprocess_meta_template(buffer, options, width, 480), "dynamic metadata");
  auto caps = std::shared_ptr<GstCaps>(gst_caps_new_empty_simple("application/vnd.simaai.tensor"),
                                       gst_caps_unref);
  auto sample = std::shared_ptr<GstSample>(gst_sample_new(buffer, caps.get(), nullptr, nullptr),
                                           gst_sample_unref);
  return decode_sample_from_inputstream_state(state, sample.get(), "unit_compact_core_decode");
}
void check(const Sample& sample, int width) {
  const auto tensors = tensors_from_sample(sample);
  require(tensors.size() == 2, "logical output count");
  for (const auto& tensor : tensors) {
    require(tensor.dtype == TensorDType::Int8 && tensor.shape == std::vector<int64_t>({2, 2}),
            "shape/dtype");
    require(tensor.semantic.quant && tensor.semantic.quant->axis == 1 &&
                tensor.semantic.quant->scales == std::vector<float>({0.25F, 0.5F}) &&
                tensor.semantic.quant->zero_points == std::vector<int32_t>({-3, 7}),
            "quantization");
    require(tensor.semantic.preprocess && tensor.semantic.preprocess->original_width == width,
            "current-frame geometry was cached");
  }
}

void check_routing(const Sample& sample, int index, int slot, const std::string& context) {
  require(sample.output_index == index && sample.logical_output_index == index &&
              sample.route_slot == slot,
          context + ": envelope routing differs");
  const auto tensors = tensors_from_sample(sample);
  require(tensors.size() == 2, context + ": logical output count");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    require(tensors[i].route.logical_index == static_cast<int>(i) &&
                tensors[i].route.route_slot == static_cast<int>(i),
            context + ": current envelope overwrote descriptor routing");
  }
}

void check_current_frame_routing(const TensorBufferPreparedMetaTemplate& prepared,
                                 const TensorBufferPublicationOwner& owner) {
  struct LegacyCase {
    const char* name;
    std::optional<int> origin_slot;
    std::optional<std::int64_t> offset;
    int index;
    int slot;
  };
  const auto max_index = std::numeric_limits<int>::max();
  const LegacyCase legacy_cases[] = {
      {"cold zero", 0, 0, 0, 0},
      {"warm zero", 0, 0, 0, 0},
      {"explicit origin precedence", 9, 4, 4, 9},
      {"offset fallback", std::nullopt, 3, 3, 3},
      {"offset absent", 7, std::nullopt, -1, 7},
      {"all absent", std::nullopt, std::nullopt, -1, -1},
      {"negative origin fallback", -1, 5, 5, 5},
      {"negative offset", std::nullopt, -1, -1, -1},
      {"overflow offset", std::nullopt, std::int64_t(max_index) + 1, -1, -1},
      {"maximum valid offset", std::nullopt, max_index, max_index, max_index},
      {"explicit origin invalid offset", 0, -1, -1, 0},
  };
  InputStream::State legacy_cached;
  legacy_cached.opt.copy_output = false;
  std::optional<TensorSetOutputDecodeSignature> legacy_signature;
  std::string error;
  for (const auto& test : legacy_cases) {
    auto current = buffer();
    require(tensor_buffer_attach_prepared_meta(current.get(), prepared, &error), error);
    if (test.origin_slot || test.offset) {
      auto* meta = gst_buffer_add_custom_meta(current.get(), "GstSimaMeta");
      require(meta != nullptr, "legacy routing fixture");
      auto* structure = gst_custom_meta_get_structure(meta);
      if (test.origin_slot)
        gst_structure_set(structure, "origin_output_slot", G_TYPE_INT, gint(*test.origin_slot),
                          nullptr);
      if (test.offset)
        gst_structure_set(structure, "buffer-offset", G_TYPE_INT64, gint64(*test.offset), nullptr);
    }
    InputStream::State fresh;
    fresh.opt.copy_output = false;
    check_routing(decode(fresh, current.get(), 0, false), test.index, test.slot,
                  std::string(test.name) + " cold full");
    check_routing(decode(legacy_cached, current.get(), 0, false), test.index, test.slot,
                  std::string(test.name) + " reused decoder");
    require(legacy_cached.tensor_set_output_decode_cache.valid, "legacy routing cache invalid");
    const auto& signature = legacy_cached.tensor_set_output_decode_cache.signature;
    if (legacy_signature)
      require(signature == *legacy_signature, "current routing changed static legacy identity");
    else
      legacy_signature = signature;
  }

  // Compact offset zero is an intentional attached-frame default, not absence.
  // An absent frame and an offset outside Sample's range must not inherit it.
  const std::optional<std::int64_t> compact_offsets[] = {
      0, 0, 7, std::nullopt, 0, std::int64_t(max_index) + 1, max_index, std::nullopt};
  InputStream::State compact_cached;
  compact_cached.opt.copy_output = false;
  std::optional<TensorSetOutputDecodeSignature> compact_signature;
  for (const auto offset : compact_offsets) {
    auto current = buffer();
    require(tensor_buffer_attach_publication(current.get(), owner, &error), error);
    if (offset) {
      TensorBufferFrameMetadata frame;
      if (*offset != 0)
        frame.buffer_offset = *offset;
      require(tensor_buffer_attach_frame_metadata(current.get(), std::move(frame), &error), error);
    }
    const int index = offset && *offset <= max_index ? static_cast<int>(*offset) : -1;
    InputStream::State fresh;
    fresh.opt.copy_output = false;
    check_routing(decode(fresh, current.get(), 0, false), index, index, "compact cold full");
    check_routing(decode(compact_cached, current.get(), 0, false), index, index,
                  "compact reused decoder");
    require(compact_cached.tensor_set_output_decode_cache.valid, "compact routing cache invalid");
    const auto& signature = compact_cached.tensor_set_output_decode_cache.signature;
    if (compact_signature)
      require(signature == *compact_signature, "current routing changed compact publication");
    else
      compact_signature = signature;
  }
  auto invalid = buffer();
  require(tensor_buffer_attach_publication(invalid.get(), owner, &error), error);
  TensorBufferFrameMetadata negative;
  negative.buffer_offset = -1;
  require(!tensor_buffer_attach_frame_metadata(invalid.get(), std::move(negative), &error) &&
              !error.empty(),
          "negative compact correlation was accepted");
  check_routing(decode(compact_cached, invalid.get(), 0, false), -1, -1,
                "rejected compact frame left stale routing");
}
void check_source_values(const TensorBufferPublicationOwner& owner) {
  using V = TensorBufferSourceValues;
  struct Case {
    const char* name;
    V source;
    int64_t id, pts, dts, duration;
  };
  V zero;
  zero.present = V::AllFields;
  zero.valid = V::ValidityFields;
  V invalid = zero;
  invalid.valid = 0;
  V missing = zero;
  missing.present = V::ValidityFields;
  V numeric = zero;
  numeric.present = V::AllFields & ~V::ValidityFields;
  V range = zero;
  range.frame_id = -2;
  range.pts_ns = range.dts_ns = range.duration_ns = std::numeric_limits<uint64_t>::max();
  V maximum = zero;
  maximum.frame_id = std::numeric_limits<int64_t>::max();
  maximum.pts_ns = maximum.dts_ns = maximum.duration_ns = uint64_t(maximum.frame_id);
  const Case cases[] = {
      {"absent", {}, 71, 101, 102, 103},
      {"valid zero", zero, 0, 0, 0, 0},
      {"explicit invalidity", invalid, -1, -1, -1, -1},
      {"validity without numeric", missing, -1, 101, 102, 103},
      {"numeric without validity", numeric, 71, 101, 102, 103},
      {"out of public range", range, -1, 101, 102, 103},
      {"maximum public value", maximum, maximum.frame_id, maximum.frame_id, maximum.frame_id,
       maximum.frame_id},
  };
  InputStream::State reused;
  reused.opt.copy_output = false;
  const auto check_values = [](const Sample& sample, const Case& expected) {
    require(sample.frame_id == expected.id && sample.pts_ns == expected.pts &&
                sample.dts_ns == expected.dts && sample.duration_ns == expected.duration,
            std::string(expected.name) + ": source public policy differs");
    require(sample.input_seq == 53 && sample.orig_input_seq == 54,
            "source identity overwrote stage correlation");
  };
  std::string error;
  for (int repetition = 0; repetition != 2; ++repetition) {
    for (const auto& test : cases) {
      auto compact = buffer();
      GST_BUFFER_PTS(compact.get()) = 101;
      GST_BUFFER_DTS(compact.get()) = 102;
      GST_BUFFER_DURATION(compact.get()) = 103;
      require(tensor_buffer_attach_publication(compact.get(), owner, &error), error);
      TensorBufferFrameMetadata stage;
      stage.frame_id = 71;
      stage.timestamp = 777;
      stage.input_sequence = 53;
      stage.original_sequence = 54;
      require(tensor_buffer_attach_frame_metadata(compact.get(), stage, test.source, &error),
              error);
      InputStream::State cold;
      cold.opt.copy_output = false;
      check_values(decode(cold, compact.get(), 0, false), test);
      check_values(decode(reused, compact.get(), 0, false), test);
      auto legacy = std::shared_ptr<GstBuffer>(tensor_buffer_clone_envelope(compact.get(), &error),
                                               gst_buffer_unref);
      require(legacy && tensor_buffer_materialize_frame_metadata(legacy.get(), &error), error);
      check_values(decode(reused, legacy.get(), 0, false), test);
      // A wrong-typed numeric behaves as absent, not as zero; an absent or
      // wrong-typed validity leaves stage/header fallback unchanged.
      auto* fields =
          gst_custom_meta_get_structure(gst_buffer_get_custom_meta(legacy.get(), "GstSimaMeta"));
      gst_structure_set(fields, "sample-frame-id-valid", G_TYPE_BOOLEAN, TRUE, "sample-frame-id",
                        G_TYPE_STRING, "not-an-id", "sample-pts-valid", G_TYPE_BOOLEAN, TRUE,
                        "sample-pts-ns", G_TYPE_STRING, "not-a-time", "sample-dts-valid",
                        G_TYPE_STRING, "not-a-bool", "sample-duration-valid", G_TYPE_STRING,
                        "not-a-bool", nullptr);
      check_values(decode(reused, legacy.get(), 0, false),
                   {"legacy wrong types", {}, -1, 101, 102, 103});
    }
  }
}
} // namespace

RUN_TEST(
    "unit_compact_core_decode_test", ([] {
      gst_init(nullptr, nullptr);
      simaai::gst::tensor_buffer_register_meta();
      if (!gst_meta_get_info("GstSimaMeta")) {
        static const gchar* tags[] = {nullptr};
        gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
      }
      const char* static_env = std::getenv("SIMA_INPUTSTREAM_FAST_STATIC_PREPROC_META");
      const bool static_preprocess = static_env && std::string(static_env) == "1";
      auto prepared = prepare();
      std::string error;
      auto owner = tensor_buffer_freeze_publication(prepared, &error);
      require(bool(owner), error);
      check_current_frame_routing(prepared, owner);
      check_source_values(owner);
      InputStream::State state;
      state.opt.copy_output = false;
      auto legacy = buffer();
      require(tensor_buffer_attach_prepared_meta(legacy.get(), prepared, &error), error);
      check(decode(state, legacy.get(), 640), 640);
      if (!static_preprocess) {
        for (const int width : {1280, 1920, 640}) {
          auto next = buffer();
          require(tensor_buffer_attach_prepared_meta(next.get(), prepared, &error), error);
          check(decode(state, next.get(), width), width);
        }
        auto absent = buffer();
        require(tensor_buffer_attach_prepared_meta(absent.get(), prepared, &error), error);
        for (const auto& tensor : tensors_from_sample(decode(state, absent.get(), 0, false)))
          require(!tensor.semantic.preprocess, "legacy frame inherited absent preprocess facts");
      }
      for (const int width : {1280, 1920, 640}) {
        auto compact = buffer();
        require(tensor_buffer_attach_publication(compact.get(), owner, &error), error);
        check(decode(state, compact.get(), width), width);
        require(state.tensor_set_output_decode_cache.signature.publication == owner,
                "compact cache identity");
      }
      // Full v2 changes only current-frame facts, never static owner identity.
      // Compare direct facts decoding with explicit legacy serialization.
      for (const int width : {1280, 1920, 640}) {
        auto compact = buffer();
        require(tensor_buffer_attach_publication(compact.get(), owner, &error), error);
        TensorBufferPreprocessFacts facts;
        facts.original_width = width;
        facts.original_height = 480;
        facts.resized_width = facts.resized_height = 640;
        facts.primary = {0, 0, width, 480};
        facts.placed = {0, 80, 640, 480};
        facts.affine = {2.0, 0.0, 1.0, 0.0, 2.0, -80.0};
        facts.source_stride = width * 3;
        facts.resize_mode = "letterbox";
        facts.color_in = facts.color_out = "RGB";
        facts.roi_list = true;
        facts.has_roi = width != 1920;
        facts.counts = facts.has_roi ? std::array<std::uint64_t, 4>{1, 1, 0, 0}
                                     : std::array<std::uint64_t, 4>{0, 0, 0, 0};
        if (facts.has_roi) {
          facts.roi_batch_indices = {0};
          facts.roi_rects = {0, 0, width, 480};
          facts.roi_affines = {2.0, 0.0, 1.0, 0.0, 2.0, -80.0};
        }
        TensorBufferFrameMetadata frame;
        frame.frame_id = width;
        frame.input_sequence = width + 1;
        frame.original_sequence = width + 2;
        frame.stream_id = "camera";
        frame.buffer_name = "preproc";
        frame.preprocess = tensor_buffer_freeze_preprocess_facts(std::move(facts), &error);
        require(bool(frame.preprocess), error);
        require(tensor_buffer_attach_frame_metadata(compact.get(), std::move(frame), &error),
                error);
        require(!gst_buffer_get_custom_meta(compact.get(), "GstSimaMeta"),
                "v2 duplicated dynamic metadata");
        const auto direct = decode(state, compact.get(), width, false);
        check(direct, width);
        require(direct.frame_id == width && direct.input_seq == width + 1 &&
                    direct.orig_input_seq == width + 2 && direct.stream_id == "camera" &&
                    direct.stream_label == "preproc",
                "v2 correlation differs");
        auto repush = direct;
        repush.kind = SampleKind::TensorSet;
        repush.payload_type = PayloadType::Tensor;
        repush.media_type = "application/vnd.simaai.tensor";
        repush.tensors = tensors_from_sample(direct);
        std::reverse(repush.tensors.begin(), repush.tensors.end());
        auto repush_holder =
            pipeline_internal::make_sample_holder_from_bundle(repush, &error, true);
        require(bool(repush_holder), error);
        auto* repush_buffer = gst_sample_get_buffer(static_cast<GstSample*>(repush_holder.get()));
        require(!tensor_buffer_publication(repush_buffer, &error) && error.empty(),
                "public descriptor rewrite retained immutable authority");
        require(gst_buffer_get_custom_meta(repush_buffer, SIMA_TENSOR_SET_META_NAME),
                "public descriptor rewrite lost legacy authority");
        require(tensor_buffer_publication(compact.get(), &error) == owner &&
                    tensor_buffer_frame_metadata(compact.get(), &error) &&
                    !gst_buffer_get_custom_meta(compact.get(), "GstSimaMeta"),
                "public descriptor rewrite changed held source");
        auto legacy_clone = std::shared_ptr<GstBuffer>(
            tensor_buffer_clone_envelope(compact.get(), &error), gst_buffer_unref);
        require(bool(legacy_clone) &&
                    tensor_buffer_materialize_publication(legacy_clone.get(), &error),
                error);
        InputStream::State legacy_state;
        legacy_state.opt.copy_output = false;
        const auto restored = decode(legacy_state, legacy_clone.get(), width, false);
        check(restored, width);
        const auto direct_tensors = tensors_from_sample(direct);
        const auto restored_tensors = tensors_from_sample(restored);
        const auto& x = *direct_tensors.front().semantic.preprocess;
        const auto& y = *restored_tensors.front().semantic.preprocess;
        require(x.scaled_width == y.scaled_width && x.pad_top == y.pad_top &&
                    x.pad_bottom == y.pad_bottom && x.affine_m12 == y.affine_m12 &&
                    x.roi_valid_count == y.roi_valid_count && x.roi_capacity == y.roi_capacity &&
                    x.rois.size() == y.rois.size() && x.roi_affines.size() == y.roi_affines.size(),
                "direct facts and boundary materialization disagree");
        if (width == 640) {
          // Release decoded owners before mutation of the test envelope.
          auto mixed_frame = buffer();
          require(tensor_buffer_attach_publication(mixed_frame.get(), owner, &error), error);
          TensorBufferFrameMetadata other;
          require(tensor_buffer_attach_frame_metadata(mixed_frame.get(), std::move(other), &error),
                  error);
          require(gst_buffer_add_custom_meta(mixed_frame.get(), "GstSimaMeta"),
                  "mixed frame fixture");
          bool rejected_frame = false;
          try {
            (void)decode(state, mixed_frame.get(), width, false);
          } catch (const std::exception&) {
            rejected_frame = true;
          }
          require(rejected_frame, "mixed current-frame authorities accepted");
        }
      }
      for (const bool full_frame : {false, true}) {
        auto absent = buffer();
        require(tensor_buffer_attach_publication(absent.get(), owner, &error), error);
        if (full_frame) {
          TensorBufferFrameMetadata empty;
          require(tensor_buffer_attach_frame_metadata(absent.get(), std::move(empty), &error),
                  error);
        }
        for (const auto& tensor : tensors_from_sample(decode(state, absent.get(), 0, false)))
          require(!tensor.semantic.preprocess, "compact frame inherited absent preprocess facts");
      }
      // A mutable legacy preprocess writer must explicitly materialize the
      // cloned dynamic header, without changing the held source publication.
      auto held_frame = buffer();
      require(tensor_buffer_attach_publication(held_frame.get(), owner, &error), error);
      TensorBufferFrameMetadata empty_facts;
      empty_facts.frame_id = 987;
      require(tensor_buffer_attach_frame_metadata(held_frame.get(), std::move(empty_facts), &error),
              error);
      auto writable = std::shared_ptr<GstBuffer>(
          tensor_buffer_clone_envelope(held_frame.get(), &error), gst_buffer_unref);
      require(bool(writable), error);
      InputStream::State writer_state;
      writer_state.opt.copy_output = false;
      check(decode(writer_state, writable.get(), 640), 640);
      require(!tensor_buffer_frame_metadata(writable.get(), &error) && error.empty(),
              "legacy writer left a compact dynamic authority");
      require(gst_buffer_get_custom_meta(writable.get(), "GstSimaMeta"),
              "legacy writer missing metadata");
      const auto* unchanged = tensor_buffer_frame_metadata(held_frame.get(), &error);
      require(unchanged && unchanged->frame_id == 987 && !unchanged->preprocess &&
                  !gst_buffer_get_custom_meta(held_frame.get(), "GstSimaMeta"),
              "legacy writer mutated held compact source");
      auto small = buffer();
      require(tensor_buffer_attach_publication(small.get(), owner, &error), error);
      gst_buffer_remove_all_memory(small.get());
      for (int i = 0; i < 2; ++i)
        gst_buffer_append_memory(small.get(), gst_allocator_alloc(nullptr, 8, nullptr));
      bool rejected = false;
      try {
        (void)decode(state, small.get(), 640);
      } catch (const std::exception&) {
        rejected = true;
      }
      require(rejected, "cached compact storage shrink was accepted");
      auto mixed = buffer();
      require(tensor_buffer_attach_publication(mixed.get(), owner, &error), error);
      require(gst_buffer_add_custom_meta(mixed.get(), SIMA_TENSOR_SET_META_NAME), "mixed fixture");
      rejected = false;
      try {
        (void)decode(state, mixed.get(), 640);
      } catch (const std::exception&) {
        rejected = true;
      }
      require(rejected, "mixed canonical authorities were accepted");
    }))
