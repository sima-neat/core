#pragma once

#include <gst/gst.h>

#include <gstsimaaitensorbuffer.h>

#include "gst/ProcessMlaRuntimeConfig.h"
#include <ev/ev_tensor_abi.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define SIMA_PREPARED_RUNTIME_CONTEXT_TYPE "sima.model.prepared-runtime"
#define SIMA_PREPARED_RUNTIME_ABI_VERSION ((guint)1)

#define SIMA_PREPARED_RUNTIME_KEY_SESSION_ID "session_id"
#define SIMA_PREPARED_RUNTIME_KEY_MODEL_ID "model_id"
#define SIMA_PREPARED_RUNTIME_KEY_HANDLE "prepared_runtime_handle"

enum SimaPreparedRuntimeLookupStatus {
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK = 0,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_NO_CONTEXT = 1,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_WRONG_CONTEXT_TYPE = 2,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_MISSING_HANDLE = 3,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_ABI_MISMATCH = 4,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL = 5,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING = 6,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND = 7,
  SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_KIND_MISMATCH = 8,
};

namespace simaai::gst {

enum class PreparedStageKind : std::uint8_t {
  Unknown = 0,
  Cast = 1,
  ProcessMla = 2,
  ProcessCvu = 3,
  Dequant = 4,
  DetessDequant = 5,
};

struct CastPreparedStage {
  std::string stage_key;
  gint direction = 0;
  TensorBufferPublishContract identity_publish_contract;
  std::optional<TensorBufferPreparedMetaTemplate> prepared_meta_template;
  bool output_use_segment_allocator = false;
  std::uint64_t output_gst_flags = 0U;
};

struct ProcessMlaPreparedStage {
  std::string stage_key;
  ProcessMlaRuntimeConfig runtime_cfg;
  TensorBufferReadRequest input_request;
  TensorBufferPublishContract output_publish_contract;
  std::optional<TensorBufferPreparedMetaTemplate> output_meta_template;
  GstCaps* sink_caps = nullptr;
  GstCaps* src_caps = nullptr;

  ProcessMlaPreparedStage() = default;
  ~ProcessMlaPreparedStage() {
    reset();
  }

  ProcessMlaPreparedStage(const ProcessMlaPreparedStage&) = delete;
  ProcessMlaPreparedStage& operator=(const ProcessMlaPreparedStage&) = delete;

  ProcessMlaPreparedStage(ProcessMlaPreparedStage&& other) noexcept
      : stage_key(std::move(other.stage_key)), runtime_cfg(std::move(other.runtime_cfg)),
        input_request(std::move(other.input_request)),
        output_publish_contract(std::move(other.output_publish_contract)),
        output_meta_template(std::move(other.output_meta_template)), sink_caps(other.sink_caps),
        src_caps(other.src_caps) {
    other.sink_caps = nullptr;
    other.src_caps = nullptr;
  }

  ProcessMlaPreparedStage& operator=(ProcessMlaPreparedStage&& other) noexcept {
    if (this != &other) {
      reset();
      stage_key = std::move(other.stage_key);
      runtime_cfg = std::move(other.runtime_cfg);
      input_request = std::move(other.input_request);
      output_publish_contract = std::move(other.output_publish_contract);
      output_meta_template = std::move(other.output_meta_template);
      sink_caps = other.sink_caps;
      src_caps = other.src_caps;
      other.sink_caps = nullptr;
      other.src_caps = nullptr;
    }
    return *this;
  }

  void reset() {
    if (sink_caps) {
      gst_caps_unref(sink_caps);
      sink_caps = nullptr;
    }
    if (src_caps) {
      gst_caps_unref(src_caps);
      src_caps = nullptr;
    }
  }
};

struct Dims {
  int w = 0;
  int h = 0;
};

enum class PreparedTensorMaterializationKind : std::uint8_t {
  Unknown = 0,
  Direct = 1,
  OffsetView = 2,
  Bf16LaneSplitRepack = 3,
};

struct CvuInputMemoryBinding {
  int sink_pad_index = -1;
  int logical_input_index = -1;
  int local_physical_index = -1;
  int source_logical_index = -1;
  int source_output_slot = -1;
  int source_physical_index = -1;
  std::uint64_t source_size_bytes = 0U;
  std::int64_t source_byte_offset = 0;
  std::string group_name;
  std::string segment_name;
  std::string graph_input_name;
  std::vector<std::int64_t> shape;
  std::string dtype;
  std::string layout;
};

struct CvuOutputBinding {
  int output_slot = 0;
  int logical_output_index = -1;
  int physical_output_index = -1;
  std::int64_t byte_offset = 0;
  std::string dispatcher_name;
  std::string cm_output_name;
  std::string segment_name;
  enum class ContractKind {
    Dense,
    Packed,
  };
  ContractKind contract_kind = ContractKind::Dense;
  std::vector<std::int64_t> shape;
  std::optional<std::uint64_t> size_bytes;
  std::string dtype;
  std::string layout;
};

struct CvuRoutingContract {
  std::vector<CvuInputMemoryBinding> input_bindings;
  std::vector<CvuOutputBinding> runtime_output_bindings;
  std::vector<CvuOutputBinding> exposed_output_bindings;
  std::optional<Dims> global_input_dims;
  bool is_preproc_graph = false;
  bool preproc_single_output_handoff = false;
  bool single_input_mode = false;
};

struct ProcessCvuPreparedLogicalInput {
  int logical_index = -1;
  int backend_input_index = -1;
  int physical_index = -1;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride_bytes;
  std::int64_t byte_offset = 0;
  std::uint64_t size_bytes = 0U;
  std::string dtype;
  std::string layout;
  std::string logical_name;
  std::string backend_name;
  std::string segment_name;
  PreparedTensorMaterializationKind materialization_kind =
      PreparedTensorMaterializationKind::Direct;
  std::optional<TensorBufferQuantView> quant;
};

struct ProcessCvuPreparedPhysicalInput {
  int physical_index = -1;
  int allocator_index = -1;
  int source_physical_index = -1;
  std::uint64_t size_bytes = 0U;
  std::int64_t source_byte_offset = 0;
  std::string segment_name;
};

struct PreparedProcessCvuTypedConfig {
  std::string graph_name;
  std::string cpu;
  std::string requested_run_target = "AUTO";
  std::string run_target = "AUTO";
  std::string resolved_exec_backend = "EVXX";
  std::string run_target_resolution_reason;
  int32_t graph_id = -1;
  std::string default_input_name;

  int32_t scaled_width = -1;
  int32_t scaled_height = -1;
  int32_t input_stride = -1;
  int32_t output_stride = -1;
  int32_t input_offset = -1;
  int32_t batch_size = -1;
  int32_t round_off = -1;
  int32_t byte_align = -1;
  std::uint32_t opt_flags = 0U;

  int32_t aspect_ratio = -1;
  int32_t normalize = -1;
  int32_t tessellate = -1;

  bool has_q_scale = false;
  float q_scale = 1.0f;
  bool has_q_zp = false;
  int32_t q_zp = 0;
  int32_t num_in_tensor = -1;
  std::vector<sima_ev_tensor_desc> input_tensors;
  std::vector<sima_ev_tensor_desc> output_tensors;
  std::vector<int32_t> input_materialization_kind_array;
  std::vector<float> q_scale_array;
  std::vector<int32_t> q_zp_array;
  std::vector<float> dq_scale_array;
  std::vector<int32_t> dq_zp_array;
  std::vector<int32_t> round_off_array;
  std::vector<int32_t> byte_align_array;
  std::vector<std::string> input_dtype_array;
  std::vector<std::string> output_dtype_array;
  std::vector<std::string> out_dtype_array;
  std::vector<std::string> runtime_output_names;
  std::vector<std::string> published_output_names;
  std::string primary_output_name;
  bool single_output_handoff = false;

  bool has_channel_mean = false;
  std::array<float, 3> channel_mean = {0.0f, 0.0f, 0.0f};
  bool has_channel_stddev = false;
  std::array<float, 3> channel_stddev = {1.0f, 1.0f, 1.0f};

  std::string input_img_type;
  std::string output_img_type;
  std::string input_dtype;
  std::string output_dtype;
  std::string out_dtype;
  std::string scaling_type;
  std::string padding_type;
};

struct ProcessCvuPreparedStage {
  std::string stage_key;
  PreparedProcessCvuTypedConfig typed_config;
  std::vector<ProcessCvuPreparedLogicalInput> logical_inputs;
  std::vector<ProcessCvuPreparedPhysicalInput> physical_inputs;
  CvuRoutingContract routing_contract;
  TensorBufferPublishContract output_publish_contract;
  std::optional<TensorBufferPreparedMetaTemplate> output_meta_template;
  std::string primary_output_name;
  bool primary_output_packed_caps = false;
  GstCaps* sink_caps = nullptr;
  GstCaps* src_caps = nullptr;

  ProcessCvuPreparedStage() = default;
  ~ProcessCvuPreparedStage() {
    reset();
  }

  ProcessCvuPreparedStage(const ProcessCvuPreparedStage&) = delete;
  ProcessCvuPreparedStage& operator=(const ProcessCvuPreparedStage&) = delete;

  ProcessCvuPreparedStage(ProcessCvuPreparedStage&& other) noexcept
      : stage_key(std::move(other.stage_key)), typed_config(std::move(other.typed_config)),
        logical_inputs(std::move(other.logical_inputs)),
        physical_inputs(std::move(other.physical_inputs)),
        routing_contract(std::move(other.routing_contract)),
        output_publish_contract(std::move(other.output_publish_contract)),
        output_meta_template(std::move(other.output_meta_template)),
        primary_output_name(std::move(other.primary_output_name)),
        primary_output_packed_caps(other.primary_output_packed_caps), sink_caps(other.sink_caps),
        src_caps(other.src_caps) {
    other.sink_caps = nullptr;
    other.src_caps = nullptr;
  }

  ProcessCvuPreparedStage& operator=(ProcessCvuPreparedStage&& other) noexcept {
    if (this != &other) {
      reset();
      stage_key = std::move(other.stage_key);
      typed_config = std::move(other.typed_config);
      logical_inputs = std::move(other.logical_inputs);
      physical_inputs = std::move(other.physical_inputs);
      routing_contract = std::move(other.routing_contract);
      output_publish_contract = std::move(other.output_publish_contract);
      output_meta_template = std::move(other.output_meta_template);
      primary_output_name = std::move(other.primary_output_name);
      primary_output_packed_caps = other.primary_output_packed_caps;
      sink_caps = other.sink_caps;
      src_caps = other.src_caps;
      other.sink_caps = nullptr;
      other.src_caps = nullptr;
    }
    return *this;
  }

  void reset() {
    if (sink_caps) {
      gst_caps_unref(sink_caps);
      sink_caps = nullptr;
    }
    if (src_caps) {
      gst_caps_unref(src_caps);
      src_caps = nullptr;
    }
  }
};

struct DequantPreparedSpan {
  std::size_t input_elem_offset = 0U;
  std::size_t output_elem_offset = 0U;
  std::size_t elem_count = 0U;
  std::uint64_t input_byte_offset = 0U;
  std::uint64_t output_byte_offset = 0U;
  double q_scale = 0.0;
  std::int64_t q_zp = 0;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> input_stride_bytes;
  std::vector<std::int64_t> output_stride_bytes;
  bool input_contiguous = true;
  bool output_contiguous = true;
};

struct DequantPreparedStage {
  std::string stage_key;
  std::string input_dtype;
  std::vector<std::int64_t> tensor_shape;
  std::string tensor_layout;
  std::size_t input_elem_bytes = 0U;
  std::uint64_t required_input_bytes = 0U;
  std::uint64_t required_output_bytes = 0U;
  bool has_scalar_quant = false;
  double q_scale = 0.0;
  std::int64_t q_zp = 0;
  std::vector<DequantPreparedSpan> quant_spans;
  TensorBufferPublishContract identity_publish_contract;
  std::optional<TensorBufferPreparedMetaTemplate> prepared_meta_template;
  GstCaps* sink_caps = nullptr;
  GstCaps* src_caps = nullptr;

  DequantPreparedStage() = default;
  ~DequantPreparedStage() {
    reset();
  }

  DequantPreparedStage(const DequantPreparedStage&) = delete;
  DequantPreparedStage& operator=(const DequantPreparedStage&) = delete;

  DequantPreparedStage(DequantPreparedStage&& other) noexcept
      : stage_key(std::move(other.stage_key)), input_dtype(std::move(other.input_dtype)),
        tensor_shape(std::move(other.tensor_shape)), tensor_layout(std::move(other.tensor_layout)),
        input_elem_bytes(other.input_elem_bytes), required_input_bytes(other.required_input_bytes),
        required_output_bytes(other.required_output_bytes),
        has_scalar_quant(other.has_scalar_quant), q_scale(other.q_scale), q_zp(other.q_zp),
        quant_spans(std::move(other.quant_spans)),
        identity_publish_contract(std::move(other.identity_publish_contract)),
        prepared_meta_template(std::move(other.prepared_meta_template)), sink_caps(other.sink_caps),
        src_caps(other.src_caps) {
    other.input_elem_bytes = 0U;
    other.required_input_bytes = 0U;
    other.required_output_bytes = 0U;
    other.has_scalar_quant = false;
    other.q_scale = 0.0;
    other.q_zp = 0;
    other.sink_caps = nullptr;
    other.src_caps = nullptr;
  }

  DequantPreparedStage& operator=(DequantPreparedStage&& other) noexcept {
    if (this != &other) {
      reset();
      stage_key = std::move(other.stage_key);
      input_dtype = std::move(other.input_dtype);
      tensor_shape = std::move(other.tensor_shape);
      tensor_layout = std::move(other.tensor_layout);
      input_elem_bytes = other.input_elem_bytes;
      required_input_bytes = other.required_input_bytes;
      required_output_bytes = other.required_output_bytes;
      has_scalar_quant = other.has_scalar_quant;
      q_scale = other.q_scale;
      q_zp = other.q_zp;
      quant_spans = std::move(other.quant_spans);
      identity_publish_contract = std::move(other.identity_publish_contract);
      prepared_meta_template = std::move(other.prepared_meta_template);
      sink_caps = other.sink_caps;
      src_caps = other.src_caps;
      other.input_elem_bytes = 0U;
      other.required_input_bytes = 0U;
      other.required_output_bytes = 0U;
      other.has_scalar_quant = false;
      other.q_scale = 0.0;
      other.q_zp = 0;
      other.sink_caps = nullptr;
      other.src_caps = nullptr;
    }
    return *this;
  }

  void reset() {
    if (sink_caps) {
      gst_caps_unref(sink_caps);
      sink_caps = nullptr;
    }
    if (src_caps) {
      gst_caps_unref(src_caps);
      src_caps = nullptr;
    }
  }
};

struct DetessDequantPreparedStage {
  std::string stage_key;
  std::vector<std::vector<int>> output_shapes;
  std::vector<std::vector<int>> slice_shapes;
  std::vector<double> default_dq_scales;
  std::vector<double> default_dq_zps;
  TensorBufferPublishContract identity_publish_contract;
  std::optional<TensorBufferPreparedMetaTemplate> prepared_meta_template;
};

struct PreparedStageSpec {
  std::string stage_key;
  PreparedStageKind kind = PreparedStageKind::Unknown;
  std::optional<CastPreparedStage> cast;
  std::optional<ProcessMlaPreparedStage> processmla;
  std::optional<ProcessCvuPreparedStage> processcvu;
  std::optional<DequantPreparedStage> dequant;
  std::optional<DetessDequantPreparedStage> detessdequant;
};

} // namespace simaai::gst

typedef struct SimaPreparedRuntimeAccessor SimaPreparedRuntimeAccessor;

typedef struct SimaPreparedRuntimeHandle {
  guint abi_version;
  gpointer user_data;
  void (*ref)(gpointer user_data);
  void (*unref)(gpointer user_data);
  const SimaPreparedRuntimeAccessor* (*accessor)(gpointer user_data);
} SimaPreparedRuntimeHandle;

struct SimaPreparedRuntimeAccessor {
  guint abi_version;
  gpointer user_data;

  const gchar* (*session_id)(gpointer user_data);
  const gchar* (*model_id)(gpointer user_data);

  guint (*stage_count)(gpointer user_data);
  const simaai::gst::PreparedStageSpec* (*stage_by_index)(gpointer user_data, guint index);
  const simaai::gst::PreparedStageSpec* (*stage_by_key)(gpointer user_data, const gchar* stage_key);
  const simaai::gst::CastPreparedStage* (*cast_stage_by_key)(gpointer user_data,
                                                             const gchar* stage_key);
  const simaai::gst::ProcessMlaPreparedStage* (*processmla_stage_by_key)(gpointer user_data,
                                                                         const gchar* stage_key);
  const simaai::gst::ProcessCvuPreparedStage* (*processcvu_stage_by_key)(gpointer user_data,
                                                                         const gchar* stage_key);
  const simaai::gst::DequantPreparedStage* (*dequant_stage_by_key)(gpointer user_data,
                                                                   const gchar* stage_key);
  const simaai::gst::DetessDequantPreparedStage* (*detessdequant_stage_by_key)(
      gpointer user_data, const gchar* stage_key);
};

static inline const gchar*
sima_prepared_runtime_lookup_status_name(SimaPreparedRuntimeLookupStatus status) {
  switch (status) {
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK:
    return "ok";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_NO_CONTEXT:
    return "no_context";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_WRONG_CONTEXT_TYPE:
    return "wrong_context_type";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_MISSING_HANDLE:
    return "missing_handle";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_ABI_MISMATCH:
    return "handle_abi_mismatch";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL:
    return "handle_accessor_null";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING:
    return "handle_callback_missing";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND:
    return "stage_not_found";
  case SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_KIND_MISMATCH:
    return "stage_kind_mismatch";
  default:
    return "unknown";
  }
}

static inline gpointer sima_prepared_runtime_handle_boxed_copy(gpointer boxed) {
  auto* handle = static_cast<SimaPreparedRuntimeHandle*>(boxed);
  if (handle && handle->ref) {
    handle->ref(handle->user_data);
  }
  return boxed;
}

static inline void sima_prepared_runtime_handle_boxed_free(gpointer boxed) {
  auto* handle = static_cast<SimaPreparedRuntimeHandle*>(boxed);
  if (handle && handle->unref) {
    handle->unref(handle->user_data);
  }
}

static inline GType sima_prepared_runtime_handle_get_type(void) {
  static const gchar* kTypeName = "SimaPreparedRuntimeHandle";
  GType type = g_type_from_name(kTypeName);
  if (type != 0) {
    return type;
  }
  return g_boxed_type_register_static(kTypeName, sima_prepared_runtime_handle_boxed_copy,
                                      sima_prepared_runtime_handle_boxed_free);
}

static inline void sima_prepared_runtime_set_lookup_status(SimaPreparedRuntimeLookupStatus* status,
                                                           SimaPreparedRuntimeLookupStatus value) {
  if (status) {
    *status = value;
  }
}

static inline gboolean sima_prepared_runtime_context_matches(const GstContext* context) {
  if (!context) {
    return FALSE;
  }
  return g_strcmp0(gst_context_get_context_type(context), SIMA_PREPARED_RUNTIME_CONTEXT_TYPE) == 0;
}

static inline const GstStructure*
sima_prepared_runtime_context_structure(const GstContext* context) {
  if (!sima_prepared_runtime_context_matches(context)) {
    return NULL;
  }
  return gst_context_get_structure(context);
}

static inline const SimaPreparedRuntimeHandle*
sima_prepared_runtime_context_handle(const GstContext* context) {
  const GstStructure* structure = sima_prepared_runtime_context_structure(context);
  if (!structure) {
    return NULL;
  }
  const GValue* handle_val = gst_structure_get_value(structure, SIMA_PREPARED_RUNTIME_KEY_HANDLE);
  if (!handle_val || G_VALUE_TYPE(handle_val) != sima_prepared_runtime_handle_get_type()) {
    return NULL;
  }
  return static_cast<const SimaPreparedRuntimeHandle*>(g_value_get_boxed(handle_val));
}

static inline const SimaPreparedRuntimeAccessor*
sima_prepared_runtime_context_accessor_checked(const GstContext* context,
                                               SimaPreparedRuntimeLookupStatus* status) {
  if (!context) {
    sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_NO_CONTEXT);
    return nullptr;
  }
  if (!sima_prepared_runtime_context_matches(context)) {
    sima_prepared_runtime_set_lookup_status(status,
                                            SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_WRONG_CONTEXT_TYPE);
    return nullptr;
  }
  const auto* handle = sima_prepared_runtime_context_handle(context);
  if (!handle) {
    sima_prepared_runtime_set_lookup_status(status,
                                            SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_MISSING_HANDLE);
    return nullptr;
  }
  if (handle->abi_version != SIMA_PREPARED_RUNTIME_ABI_VERSION) {
    sima_prepared_runtime_set_lookup_status(
        status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_ABI_MISMATCH);
    return nullptr;
  }
  if (!handle->ref || !handle->unref || !handle->accessor) {
    sima_prepared_runtime_set_lookup_status(
        status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING);
    return nullptr;
  }
  const auto* accessor = handle->accessor(handle->user_data);
  if (!accessor) {
    sima_prepared_runtime_set_lookup_status(
        status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL);
    return nullptr;
  }
  if (accessor->abi_version != SIMA_PREPARED_RUNTIME_ABI_VERSION) {
    sima_prepared_runtime_set_lookup_status(
        status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_HANDLE_ABI_MISMATCH);
    return nullptr;
  }
  sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK);
  return accessor;
}

static inline const simaai::gst::PreparedStageSpec*
sima_prepared_runtime_context_stage_lookup_checked(const GstContext* context,
                                                   const gchar* stage_key,
                                                   SimaPreparedRuntimeLookupStatus* status) {
  const auto* accessor = sima_prepared_runtime_context_accessor_checked(context, status);
  if (!accessor || !accessor->stage_by_key || !stage_key || !*stage_key) {
    if (status && (!stage_key || !*stage_key)) {
      *status = SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND;
    }
    return nullptr;
  }
  const auto* stage = accessor->stage_by_key(accessor->user_data, stage_key);
  if (!stage) {
    sima_prepared_runtime_set_lookup_status(status,
                                            SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND);
    return nullptr;
  }
  sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK);
  return stage;
}

static inline const simaai::gst::CastPreparedStage*
sima_prepared_runtime_context_cast_stage_lookup_checked(const GstContext* context,
                                                        const gchar* stage_key,
                                                        SimaPreparedRuntimeLookupStatus* status) {
  const auto* accessor = sima_prepared_runtime_context_accessor_checked(context, status);
  if (!accessor || !accessor->cast_stage_by_key || !stage_key || !*stage_key) {
    if (status && (!stage_key || !*stage_key)) {
      *status = SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND;
    }
    return nullptr;
  }
  const auto* stage = accessor->cast_stage_by_key(accessor->user_data, stage_key);
  if (!stage) {
    const auto* generic =
        accessor->stage_by_key ? accessor->stage_by_key(accessor->user_data, stage_key) : nullptr;
    sima_prepared_runtime_set_lookup_status(
        status, generic ? SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_KIND_MISMATCH
                        : SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND);
    return nullptr;
  }
  sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK);
  return stage;
}

static inline const simaai::gst::ProcessMlaPreparedStage*
sima_prepared_runtime_context_processmla_stage_lookup_checked(
    const GstContext* context, const gchar* stage_key, SimaPreparedRuntimeLookupStatus* status) {
  const auto* accessor = sima_prepared_runtime_context_accessor_checked(context, status);
  if (!accessor || !accessor->processmla_stage_by_key || !stage_key || !*stage_key) {
    if (status && (!stage_key || !*stage_key)) {
      *status = SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND;
    }
    return nullptr;
  }
  const auto* stage = accessor->processmla_stage_by_key(accessor->user_data, stage_key);
  if (!stage) {
    const auto* generic =
        accessor->stage_by_key ? accessor->stage_by_key(accessor->user_data, stage_key) : nullptr;
    sima_prepared_runtime_set_lookup_status(
        status, generic ? SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_KIND_MISMATCH
                        : SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND);
    return nullptr;
  }
  sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK);
  return stage;
}

static inline const simaai::gst::ProcessCvuPreparedStage*
sima_prepared_runtime_context_processcvu_stage_lookup_checked(
    const GstContext* context, const gchar* stage_key, SimaPreparedRuntimeLookupStatus* status) {
  const auto* accessor = sima_prepared_runtime_context_accessor_checked(context, status);
  if (!accessor || !accessor->processcvu_stage_by_key || !stage_key || !*stage_key) {
    if (status && (!stage_key || !*stage_key)) {
      *status = SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND;
    }
    return nullptr;
  }
  const auto* stage = accessor->processcvu_stage_by_key(accessor->user_data, stage_key);
  if (!stage) {
    const auto* generic =
        accessor->stage_by_key ? accessor->stage_by_key(accessor->user_data, stage_key) : nullptr;
    sima_prepared_runtime_set_lookup_status(
        status, generic ? SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_KIND_MISMATCH
                        : SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND);
    return nullptr;
  }
  sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK);
  return stage;
}

static inline const simaai::gst::DequantPreparedStage*
sima_prepared_runtime_context_dequant_stage_lookup_checked(
    const GstContext* context, const gchar* stage_key, SimaPreparedRuntimeLookupStatus* status) {
  const auto* accessor = sima_prepared_runtime_context_accessor_checked(context, status);
  if (!accessor || !accessor->dequant_stage_by_key || !stage_key || !*stage_key) {
    if (status && (!stage_key || !*stage_key)) {
      *status = SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND;
    }
    return nullptr;
  }
  const auto* stage = accessor->dequant_stage_by_key(accessor->user_data, stage_key);
  if (!stage) {
    const auto* generic =
        accessor->stage_by_key ? accessor->stage_by_key(accessor->user_data, stage_key) : nullptr;
    sima_prepared_runtime_set_lookup_status(
        status, generic ? SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_KIND_MISMATCH
                        : SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND);
    return nullptr;
  }
  sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK);
  return stage;
}

static inline const simaai::gst::DetessDequantPreparedStage*
sima_prepared_runtime_context_detessdequant_stage_lookup_checked(
    const GstContext* context, const gchar* stage_key, SimaPreparedRuntimeLookupStatus* status) {
  const auto* accessor = sima_prepared_runtime_context_accessor_checked(context, status);
  if (!accessor || !accessor->detessdequant_stage_by_key || !stage_key || !*stage_key) {
    if (status && (!stage_key || !*stage_key)) {
      *status = SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND;
    }
    return nullptr;
  }
  const auto* stage = accessor->detessdequant_stage_by_key(accessor->user_data, stage_key);
  if (!stage) {
    const auto* generic =
        accessor->stage_by_key ? accessor->stage_by_key(accessor->user_data, stage_key) : nullptr;
    sima_prepared_runtime_set_lookup_status(
        status, generic ? SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_KIND_MISMATCH
                        : SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_STAGE_NOT_FOUND);
    return nullptr;
  }
  sima_prepared_runtime_set_lookup_status(status, SIMA_PREPARED_RUNTIME_LOOKUP_STATUS_OK);
  return stage;
}
