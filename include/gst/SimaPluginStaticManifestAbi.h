#pragma once

#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ev/ev_tensor_abi.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pipeline-level static manifest context contract for SIMA model plugins.
 * This is an ABI contract header; plugin/dispatcher repos should carry a synced
 * copy and must not link against framework internals.
 *
 * Context type:
 *   "sima.model.manifest"
 *
 * Required fields:
 *   - "manifest_handle" (SimaPluginStaticManifestHandle boxed value)
 *
 * Optional fields:
 *   - "session_id" (string)
 *   - "model_id" (string)
 */

#define SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE "sima.model.manifest"
#define SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION ((guint)20)

#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_SESSION_ID "session_id"
#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_MODEL_ID "model_id"
#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_HANDLE "manifest_handle"

typedef struct SimaPluginStaticManifestAbiHeader {
  guint abi_version;
} SimaPluginStaticManifestAbiHeader;

typedef struct SimaPluginStaticManifestAccessor SimaPluginStaticManifestAccessor;

typedef struct SimaPluginStaticManifestHandle {
  guint abi_version;
  gpointer user_data;
  void (*ref)(gpointer user_data);
  void (*unref)(gpointer user_data);
  const SimaPluginStaticManifestAccessor* (*accessor)(gpointer user_data);
} SimaPluginStaticManifestHandle;

typedef enum SimaPluginManifestLookupStatus {
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK = 0,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT = 1,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_WRONG_CONTEXT_TYPE = 2,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_MISSING_HANDLE = 3,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ABI_MISMATCH = 4,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL = 5,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING = 6,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_STAGE_NOT_FOUND = 7,
  SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_PAYLOAD_KIND_MISMATCH = 8
} SimaPluginManifestLookupStatus;

typedef enum SimaPluginStagePayloadKind {
  SIMA_PLUGIN_STAGE_PAYLOAD_NONE = 0,
  SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSCVU = 1,
  SIMA_PLUGIN_STAGE_PAYLOAD_PROCESSMLA = 2,
  SIMA_PLUGIN_STAGE_PAYLOAD_BOXDECODE = 3,
  SIMA_PLUGIN_STAGE_PAYLOAD_DETESSDEQUANT = 4,
  SIMA_PLUGIN_STAGE_PAYLOAD_QUANT = 5,
  SIMA_PLUGIN_STAGE_PAYLOAD_TESS = 6,
  SIMA_PLUGIN_STAGE_PAYLOAD_DEQUANT = 7,
  SIMA_PLUGIN_STAGE_PAYLOAD_QUANTTESS = 8
} SimaPluginStagePayloadKind;

typedef struct SimaPluginTensorSpec {
  gint tensor_index;
  const gint64* shape;
  guint shape_len;
  const gchar* dtype;
  const guint8* axis_semantics;
  guint axis_semantics_len;
  gint max_w;
  gint max_h;
  gint max_stride;
  const gchar* semantic_tag;
} SimaPluginTensorSpec;

typedef struct SimaPluginQuantSpec {
  gint granularity;
  gint axis;
  const gdouble* scales;
  guint scales_len;
  const gint64* zero_points;
  guint zero_points_len;
} SimaPluginQuantSpec;

typedef struct SimaPluginSinkRoute {
  gint sink_pad_index;
  gboolean required;
  const gchar* src_stage_id;
  gint src_output_slot;
  gint tensor_index;
  const gchar* cm_input_name;
  const gchar* source_segment_name;
} SimaPluginSinkRoute;

typedef struct SimaPluginOutputRoute {
  gint output_slot;
  const gchar* cm_output_name;
  const gchar* segment_name;
} SimaPluginOutputRoute;

typedef enum SimaPluginDeviceKind {
  SIMA_PLUGIN_DEVICE_KIND_UNKNOWN = 0,
  SIMA_PLUGIN_DEVICE_KIND_CPU = 1,
  SIMA_PLUGIN_DEVICE_KIND_MLA = 2,
  SIMA_PLUGIN_DEVICE_KIND_EVXX = 3
} SimaPluginDeviceKind;

typedef enum SimaPluginTensorMaterializationKind {
  SIMA_PLUGIN_TENSOR_MATERIALIZATION_UNKNOWN = 0,
  SIMA_PLUGIN_TENSOR_MATERIALIZATION_DIRECT = 1,
  SIMA_PLUGIN_TENSOR_MATERIALIZATION_OFFSET_VIEW = 2,
  SIMA_PLUGIN_TENSOR_MATERIALIZATION_BF16_LANE_SPLIT_REPACK = 3
} SimaPluginTensorMaterializationKind;

typedef struct SimaPluginPhysicalBuffer {
  gint physical_index;
  gint allocator_index;
  guint64 size_bytes;
  SimaPluginDeviceKind device_kind;
  guint64 memory_flags;
  gint segment_name_id;
  const gchar* segment_name;
  gint source_physical_index;
  gint64 source_byte_offset;
} SimaPluginPhysicalBuffer;

typedef struct SimaPluginLogicalTensor {
  gint logical_index;
  gint backend_output_index;
  gint physical_index;
  gint output_slot;
  gint tensor_index;
  gint64 byte_offset;
  guint64 size_bytes;
  const gint64* shape;
  guint shape_len;
  const gint64* stride_bytes;
  guint stride_bytes_len;
  const gchar* dtype;
  const guint8* axis_semantics;
  guint axis_semantics_len;
  gint logical_name_id;
  const gchar* logical_name;
  gint backend_name_id;
  const gchar* backend_name;
  gint segment_name_id;
  const gchar* segment_name;
  const SimaPluginQuantSpec* quant;
} SimaPluginLogicalTensor;

typedef struct SimaPluginLogicalInput {
  gint logical_index;
  gint backend_input_index;
  gint physical_index;
  const gint64* shape;
  guint shape_len;
  const gint64* stride_bytes;
  guint stride_bytes_len;
  gint64 byte_offset;
  guint64 size_bytes;
  const gchar* dtype;
  const guint8* axis_semantics;
  guint axis_semantics_len;
  gint logical_name_id;
  const gchar* logical_name;
  gint backend_name_id;
  const gchar* backend_name;
  gint segment_name_id;
  const gchar* segment_name;
  SimaPluginTensorMaterializationKind materialization_kind;
  const SimaPluginQuantSpec* quant;
} SimaPluginLogicalInput;

typedef struct SimaPluginInputBinding {
  gint sink_pad_index;
  gint local_logical_input_index;
  gint src_stage_index;
  const gchar* src_stage_id;
  gint src_logical_output_index;
  gint src_output_slot;
  gint src_physical_output_index;
  guint64 src_physical_size_bytes;
  gint64 src_physical_byte_offset;
  gboolean required;
  gint cm_input_name_id;
  const gchar* cm_input_name;
  gint source_segment_name_id;
  const gchar* source_segment_name;
} SimaPluginInputBinding;

typedef enum SimaPluginProcessCvuGraphFamily {
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_UNKNOWN = 0,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_PREPROC = 1,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_QUANT = 2,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_TESS = 3,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_QUANTTESS = 4,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_CASTTESS = 5,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DETESS = 6,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DEQUANT = 7,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DETESSCAST = 8,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_DETESSDEQUANT = 9,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_CAST = 10,
  SIMA_PLUGIN_PROCESSCVU_GRAPH_FAMILY_VISUAL_FRONTEND = 11
} SimaPluginProcessCvuGraphFamily;

typedef enum SimaPluginProcessCvuOutputTransportKind {
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_TRANSPORT_UNKNOWN = 0,
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_TRANSPORT_DENSE = 1,
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_TRANSPORT_PACKED = 2
} SimaPluginProcessCvuOutputTransportKind;

typedef enum SimaPluginProcessCvuOutputSemanticKind {
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_UNKNOWN = 0,
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_IMAGE = 1,
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_TESSELLATED_IMAGE = 2,
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_QUANTIZED_TENSOR = 3,
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_QUANTTESS_TENSOR = 4,
  SIMA_PLUGIN_PROCESSCVU_OUTPUT_SEMANTIC_TENSOR = 5
} SimaPluginProcessCvuOutputSemanticKind;

typedef struct SimaPluginProcessCvuStagePayload {
  const gchar* graph_family;
  SimaPluginProcessCvuGraphFamily graph_family_kind;
  const gchar* graph_name;
  const gchar* requested_run_target;
  const gchar* run_target;
  const gchar* resolved_exec_backend;
  const gchar* run_target_resolution_reason;
  const gchar* default_input_name;
  const gchar* const* default_output_names;
  guint default_output_names_len;
  const gchar* primary_output_name;
  SimaPluginProcessCvuOutputTransportKind primary_output_transport_kind;
  SimaPluginProcessCvuOutputSemanticKind primary_output_semantic_kind;
  const gchar* input_img_type;
  const gchar* output_img_type;
  const gchar* scaling_type;
  const gchar* padding_type;
  const gchar* input_dtype;
  const gchar* output_dtype;
  const gchar* out_dtype;

  gint pad_value;
  gint scaled_width;
  gint scaled_height;
  gint input_stride;
  gint output_stride;
  gint input_offset;
  gint batch_size;
  gint round_off;
  gint byte_align;
  gint graph_id;
  /* Native visual-frontend scalar fields for EV74 graphs 235-238. */
  gint width;
  gint height;
  gint threshold;
  gint max_features;
  gint grid_x;
  gint grid_y;
  gint min_px_dist;
  gint descriptor_words;
  gint num_points;
  gint win_half;
  gint max_iters;
  gint max_level;
  gint detect_new_features;
  gint fast_threshold;
  gint debug;
  guint32 opt_flags;
  gboolean canonical_contract;
  gboolean preproc_single_output_handoff;

  /* tri-state values: -1 unset, 0 false, 1 true */
  gint aspect_ratio;
  gint normalize;
  gint tessellate;

  gdouble q_scale;
  gint q_zp;
  gboolean has_q_scale;
  gboolean has_q_zp;
  const gdouble* q_scale_array;
  guint q_scale_array_len;
  const gint* q_zp_array;
  guint q_zp_array_len;
  gint num_in_tensor;
  const struct sima_ev_tensor_desc* input_tensors;
  guint input_tensors_len;
  const struct sima_ev_tensor_desc* output_tensors;
  guint output_tensors_len;
  const gint* runtime_output_logical_index_array;
  guint runtime_output_logical_index_array_len;
  const gint* runtime_output_output_slot_array;
  guint runtime_output_output_slot_array_len;
  const gint* runtime_output_physical_index_array;
  guint runtime_output_physical_index_array_len;
  const gchar* const* runtime_output_dtype_array;
  guint runtime_output_dtype_array_len;
  const gint* runtime_output_transport_kind_array;
  guint runtime_output_transport_kind_array_len;
  const gint* runtime_output_semantic_kind_array;
  guint runtime_output_semantic_kind_array_len;
  const gdouble* dq_scale_array;
  guint dq_scale_array_len;
  const gint* dq_zp_array;
  guint dq_zp_array_len;

  const gdouble* channel_mean;
  guint channel_mean_len;
  const gdouble* channel_stddev;
  guint channel_stddev_len;
} SimaPluginProcessCvuStagePayload;

typedef struct SimaPluginProcessMlaStagePayload {
  const gchar* model_path;
  gint batch_size;
  gint batch_sz_model;
  const gchar* const* dispatcher_output_names;
  guint dispatcher_output_names_len;
  const guint64* dispatcher_output_sizes;
  guint dispatcher_output_sizes_len;
} SimaPluginProcessMlaStagePayload;

typedef struct SimaPluginBoxDecodeStagePayload {
  const gchar* decode_type;
  const gchar* decode_type_option;
  gdouble detection_threshold;
  gdouble nms_iou_threshold;
  gint topk;
  gint num_classes;
  const struct sima_ev_shape_desc* slice_shapes;
  guint slice_shapes_len;
  const gchar* input_dtype;
  gint score_activation;
  gint tess_needed;
  gint quant_needed;
  gint model_owned_flags;
  gint quant_contract_required;
  const gint* tensor_storage_kind;
  guint tensor_storage_kind_len;
} SimaPluginBoxDecodeStagePayload;

typedef struct SimaPluginDetessDequantStagePayload {
  gint reserved;
} SimaPluginDetessDequantStagePayload;

typedef struct SimaPluginQuantLikeStagePayload {
  gint reserved;
} SimaPluginQuantLikeStagePayload;

typedef struct SimaPluginStageSpec {
  const gchar* element_name;
  const gchar* logical_stage_id;
  const gchar* plugin_kind;
  const gchar* kernel_kind;

  const gchar* const* name_table;
  guint name_table_len;
  const SimaPluginLogicalInput* logical_inputs;
  guint logical_inputs_len;
  const SimaPluginInputBinding* input_bindings;
  guint input_bindings_len;
  const SimaPluginPhysicalBuffer* physical_inputs;
  guint physical_inputs_len;
  const SimaPluginPhysicalBuffer* physical_outputs;
  guint physical_outputs_len;
  const SimaPluginLogicalTensor* logical_outputs;
  guint logical_outputs_len;
  const SimaPluginOutputRoute* output_order;
  guint output_order_len;

  const SimaPluginQuantSpec* output_quant;
  guint output_quant_len;

  const gchar* const* required_preprocess_meta_fields;
  guint required_preprocess_meta_fields_len;

  SimaPluginStagePayloadKind payload_kind;
  union {
    SimaPluginProcessCvuStagePayload processcvu;
    SimaPluginProcessMlaStagePayload processmla;
    SimaPluginBoxDecodeStagePayload boxdecode;
    SimaPluginDetessDequantStagePayload detessdequant;
    SimaPluginQuantLikeStagePayload quant;
    SimaPluginQuantLikeStagePayload tess;
    SimaPluginQuantLikeStagePayload dequant;
    SimaPluginQuantLikeStagePayload quanttess;
  } payload;
} SimaPluginStageSpec;

/*
 * ABI-safe accessor table carried via GstContext as a pointer.
 *
 * Lifetime:
 * - The owner of this table must outlive pipeline/plugin usage.
 * - Plugins must treat returned pointers as borrowed; copy if needed.
 */
struct SimaPluginStaticManifestAccessor {
  guint abi_version;
  gpointer user_data;

  const gchar* (*session_id)(gpointer user_data);
  const gchar* (*model_id)(gpointer user_data);

  guint (*stage_count)(gpointer user_data);
  const SimaPluginStageSpec* (*stage_by_index)(gpointer user_data, guint index);
  const SimaPluginStageSpec* (*stage_by_element_name)(gpointer user_data,
                                                      const gchar* element_name);
  const SimaPluginStageSpec* (*stage_by_logical_stage_id)(gpointer user_data,
                                                          const gchar* logical_stage_id);
};

static inline const gchar*
sima_plugin_manifest_lookup_status_name(SimaPluginManifestLookupStatus status) {
  switch (status) {
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK:
    return "ok";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT:
    return "no_context";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_WRONG_CONTEXT_TYPE:
    return "wrong_context_type";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_MISSING_HANDLE:
    return "missing_handle";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ABI_MISMATCH:
    return "handle_abi_mismatch";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL:
    return "handle_accessor_null";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING:
    return "handle_callback_missing";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_STAGE_NOT_FOUND:
    return "stage_not_found";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_PAYLOAD_KIND_MISMATCH:
    return "payload_kind_mismatch";
  default:
    return "unknown";
  }
}

static inline const gchar*
sima_plugin_manifest_lookup_status_error_message(SimaPluginManifestLookupStatus status) {
  switch (status) {
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT:
    return "Missing sima.model.manifest context";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_WRONG_CONTEXT_TYPE:
    return "Wrong sima.model.manifest context type";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_MISSING_HANDLE:
    return "Missing manifest handle in manifest context";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ABI_MISMATCH:
    return "Manifest handle ABI mismatch";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL:
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING:
    return "Manifest accessor unavailable";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_STAGE_NOT_FOUND:
    return "Missing stage entry in manifest context";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_PAYLOAD_KIND_MISMATCH:
    return "Manifest payload kind mismatch";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK:
  default:
    return "Manifest lookup status OK";
  }
}

static inline const gchar*
sima_plugin_manifest_lookup_status_missing_field(SimaPluginManifestLookupStatus status) {
  switch (status) {
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT:
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_WRONG_CONTEXT_TYPE:
    return "manifest_context";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_MISSING_HANDLE:
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ABI_MISMATCH:
    return "manifest_handle";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL:
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING:
    return "manifest_accessor";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_STAGE_NOT_FOUND:
    return "stage_spec";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_PAYLOAD_KIND_MISMATCH:
    return "typed_stage_payload";
  case SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK:
  default:
    return "none";
  }
}

static inline gpointer sima_plugin_static_manifest_handle_boxed_copy(gpointer boxed) {
  auto* handle = static_cast<SimaPluginStaticManifestHandle*>(boxed);
  if (handle && handle->ref) {
    handle->ref(handle->user_data);
  }
  return boxed;
}

static inline void sima_plugin_static_manifest_handle_boxed_free(gpointer boxed) {
  auto* handle = static_cast<SimaPluginStaticManifestHandle*>(boxed);
  if (handle && handle->unref) {
    handle->unref(handle->user_data);
  }
}

static inline GType sima_plugin_static_manifest_handle_get_type(void) {
  static const gchar* kTypeName = "SimaPluginStaticManifestHandle";
  static gsize type_id = 0;
  if (g_once_init_enter(&type_id)) {
    GType type = g_type_from_name(kTypeName);
    if (type == 0) {
      type = g_boxed_type_register_static(kTypeName, sima_plugin_static_manifest_handle_boxed_copy,
                                          sima_plugin_static_manifest_handle_boxed_free);
    }
    g_once_init_leave(&type_id, static_cast<gsize>(type));
  }
  return static_cast<GType>(type_id);
}

static inline void sima_plugin_manifest_set_lookup_status(SimaPluginManifestLookupStatus* status,
                                                          SimaPluginManifestLookupStatus value) {
  if (status) {
    *status = value;
  }
}

static inline gboolean sima_plugin_manifest_context_matches(const GstContext* context) {
  if (!context) {
    return FALSE;
  }
  return g_strcmp0(gst_context_get_context_type(context),
                   SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE) == 0;
}

static inline const GstStructure*
sima_plugin_manifest_context_structure(const GstContext* context) {
  if (!sima_plugin_manifest_context_matches(context)) {
    return NULL;
  }
  return gst_context_get_structure(context);
}

static inline const SimaPluginStaticManifestHandle*
sima_plugin_manifest_context_handle(const GstContext* context) {
  const GstStructure* structure = sima_plugin_manifest_context_structure(context);
  if (!structure) {
    return NULL;
  }
  const GValue* handle_val =
      gst_structure_get_value(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_HANDLE);
  if (!handle_val || G_VALUE_TYPE(handle_val) != sima_plugin_static_manifest_handle_get_type()) {
    return NULL;
  }
  const auto* handle =
      static_cast<const SimaPluginStaticManifestHandle*>(g_value_get_boxed(handle_val));
  if (!handle) {
    return NULL;
  }
  return handle;
}

static inline const SimaPluginStaticManifestAccessor*
sima_plugin_manifest_context_accessor_checked(const GstContext* context,
                                              SimaPluginManifestLookupStatus* status) {
  const bool debug_manifest_context = ([]() -> bool {
    const char* raw = getenv("SIMA_MANIFEST_CONTEXT_DEBUG");
    return raw && *raw && strcmp(raw, "0") != 0;
  })();
  if (!context) {
    sima_plugin_manifest_set_lookup_status(status, SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_NO_CONTEXT);
    return NULL;
  }
  if (!sima_plugin_manifest_context_matches(context)) {
    sima_plugin_manifest_set_lookup_status(status,
                                           SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_WRONG_CONTEXT_TYPE);
    return NULL;
  }
  const auto* handle = sima_plugin_manifest_context_handle(context);
  if (!handle) {
    sima_plugin_manifest_set_lookup_status(status,
                                           SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_MISSING_HANDLE);
    return NULL;
  }
  if (handle->abi_version != SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION) {
    if (debug_manifest_context) {
      fprintf(stderr,
              "[manifest-context-debug] lookup=handle_abi_mismatch context=%p "
              "handle=%p handle_abi=%u expected_abi=%u user_data=%p\n",
              static_cast<const void*>(context), static_cast<const void*>(handle),
              static_cast<unsigned>(handle->abi_version),
              static_cast<unsigned>(SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION), handle->user_data);
    }
    sima_plugin_manifest_set_lookup_status(status,
                                           SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ABI_MISMATCH);
    return NULL;
  }
  if (!handle->ref || !handle->unref || !handle->accessor) {
    sima_plugin_manifest_set_lookup_status(
        status, SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_CALLBACK_MISSING);
    return NULL;
  }
  const auto* accessor = handle->accessor(handle->user_data);
  if (!accessor) {
    sima_plugin_manifest_set_lookup_status(status,
                                           SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ACCESSOR_NULL);
    return NULL;
  }
  if (accessor->abi_version != SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION) {
    if (debug_manifest_context) {
      fprintf(stderr,
              "[manifest-context-debug] lookup=accessor_abi_mismatch context=%p "
              "handle=%p accessor=%p accessor_abi=%u expected_abi=%u user_data=%p\n",
              static_cast<const void*>(context), static_cast<const void*>(handle),
              static_cast<const void*>(accessor), static_cast<unsigned>(accessor->abi_version),
              static_cast<unsigned>(SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION), accessor->user_data);
    }
    sima_plugin_manifest_set_lookup_status(status,
                                           SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_HANDLE_ABI_MISMATCH);
    return NULL;
  }
  sima_plugin_manifest_set_lookup_status(status, SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK);
  return accessor;
}

static inline const SimaPluginStaticManifestAccessor*
sima_plugin_manifest_context_accessor(const GstContext* context) {
  return sima_plugin_manifest_context_accessor_checked(context, NULL);
}

static inline const gchar* sima_plugin_manifest_stage_key(const SimaPluginStageSpec* stage) {
  if (!stage) {
    return NULL;
  }
  if (stage->logical_stage_id && *stage->logical_stage_id) {
    return stage->logical_stage_id;
  }
  if (stage->element_name && *stage->element_name) {
    return stage->element_name;
  }
  return NULL;
}

static inline gboolean
sima_plugin_manifest_stage_matches_payload_kind(const SimaPluginStageSpec* stage,
                                                SimaPluginStagePayloadKind expected_payload_kind) {
  if (!stage) {
    return FALSE;
  }
  return expected_payload_kind == SIMA_PLUGIN_STAGE_PAYLOAD_NONE ||
         stage->payload_kind == expected_payload_kind;
}

static inline const SimaPluginStageSpec*
sima_plugin_manifest_stage_by_element_name(const SimaPluginStaticManifestAccessor* accessor,
                                           const gchar* element_name) {
  if (!accessor || !accessor->stage_by_element_name || !element_name || !*element_name) {
    return NULL;
  }
  return accessor->stage_by_element_name(accessor->user_data, element_name);
}

static inline const SimaPluginStageSpec*
sima_plugin_manifest_stage_by_logical_id(const SimaPluginStaticManifestAccessor* accessor,
                                         const gchar* logical_stage_id) {
  if (!accessor || !accessor->stage_by_logical_stage_id || !logical_stage_id ||
      !*logical_stage_id) {
    return NULL;
  }
  return accessor->stage_by_logical_stage_id(accessor->user_data, logical_stage_id);
}

static inline const SimaPluginStageSpec*
sima_plugin_manifest_stage_lookup(const SimaPluginStaticManifestAccessor* accessor,
                                  const gchar* stage_id_or_name,
                                  const gchar* element_name_fallback) {
  const SimaPluginStageSpec* out = NULL;
  if (stage_id_or_name && *stage_id_or_name) {
    out = sima_plugin_manifest_stage_by_logical_id(accessor, stage_id_or_name);
    if (!out) {
      out = sima_plugin_manifest_stage_by_element_name(accessor, stage_id_or_name);
    }
  }
  if (!out && element_name_fallback && *element_name_fallback) {
    out = sima_plugin_manifest_stage_by_element_name(accessor, element_name_fallback);
  }
  return out;
}

static inline const SimaPluginStageSpec* sima_plugin_manifest_stage_lookup_typed(
    const SimaPluginStaticManifestAccessor* accessor, const gchar* stage_id_or_name,
    const gchar* element_name_fallback, SimaPluginStagePayloadKind expected_payload_kind) {
  const SimaPluginStageSpec* stage =
      sima_plugin_manifest_stage_lookup(accessor, stage_id_or_name, element_name_fallback);
  if (!sima_plugin_manifest_stage_matches_payload_kind(stage, expected_payload_kind)) {
    return NULL;
  }
  return stage;
}

static inline const SimaPluginStageSpec* sima_plugin_manifest_context_stage_lookup_typed_checked(
    const GstContext* context, const gchar* stage_id_or_name, const gchar* element_name_fallback,
    SimaPluginStagePayloadKind expected_payload_kind, SimaPluginManifestLookupStatus* status);

static inline const SimaPluginStageSpec* sima_plugin_manifest_context_stage_lookup_typed(
    const GstContext* context, const gchar* stage_id_or_name, const gchar* element_name_fallback,
    SimaPluginStagePayloadKind expected_payload_kind) {
  return sima_plugin_manifest_context_stage_lookup_typed_checked(
      context, stage_id_or_name, element_name_fallback, expected_payload_kind, NULL);
}

static inline const SimaPluginStageSpec* sima_plugin_manifest_context_stage_lookup_typed_checked(
    const GstContext* context, const gchar* stage_id_or_name, const gchar* element_name_fallback,
    SimaPluginStagePayloadKind expected_payload_kind, SimaPluginManifestLookupStatus* status) {
  const SimaPluginStaticManifestAccessor* accessor =
      sima_plugin_manifest_context_accessor_checked(context, status);
  if (!accessor) {
    return NULL;
  }
  const SimaPluginStageSpec* stage =
      sima_plugin_manifest_stage_lookup(accessor, stage_id_or_name, element_name_fallback);
  if (!stage) {
    sima_plugin_manifest_set_lookup_status(status,
                                           SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_STAGE_NOT_FOUND);
    return NULL;
  }
  if (!sima_plugin_manifest_stage_matches_payload_kind(stage, expected_payload_kind)) {
    sima_plugin_manifest_set_lookup_status(
        status, SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_PAYLOAD_KIND_MISMATCH);
    return NULL;
  }
  sima_plugin_manifest_set_lookup_status(status, SIMA_PLUGIN_MANIFEST_LOOKUP_STATUS_OK);
  return stage;
}

static inline const gchar* sima_plugin_manifest_stage_name_from_id(const SimaPluginStageSpec* stage,
                                                                   gint name_id) {
  if (!stage || !stage->name_table || name_id < 0 ||
      static_cast<guint>(name_id) >= stage->name_table_len) {
    return NULL;
  }
  return stage->name_table[name_id];
}

static inline gint sima_plugin_stage_output_route_physical_index(const SimaPluginStageSpec* stage,
                                                                 gint logical_index,
                                                                 gint output_slot) {
  if (!stage || !stage->logical_outputs || stage->logical_outputs_len == 0U) {
    return -1;
  }
  for (guint i = 0; i < stage->logical_outputs_len; ++i) {
    const SimaPluginLogicalTensor* logical = &stage->logical_outputs[i];
    if (logical_index >= 0 && logical->logical_index == logical_index) {
      return logical->physical_index;
    }
    if (output_slot >= 0 && logical->output_slot == output_slot) {
      return logical->physical_index;
    }
  }
  return -1;
}

static inline gint sima_plugin_stage_output_route_logical_index(const SimaPluginStageSpec* stage,
                                                                gint output_slot,
                                                                const gchar* cm_output_name,
                                                                const gchar* segment_name) {
  if (!stage || !stage->logical_outputs || stage->logical_outputs_len == 0U) {
    return -1;
  }

  if (cm_output_name && *cm_output_name) {
    for (guint i = 0; i < stage->logical_outputs_len; ++i) {
      const SimaPluginLogicalTensor* logical = &stage->logical_outputs[i];
      if (logical->backend_name && g_strcmp0(logical->backend_name, cm_output_name) == 0) {
        return logical->logical_index;
      }
    }
  }

  if (segment_name && *segment_name) {
    for (guint i = 0; i < stage->logical_outputs_len; ++i) {
      const SimaPluginLogicalTensor* logical = &stage->logical_outputs[i];
      if (logical->segment_name && g_strcmp0(logical->segment_name, segment_name) == 0) {
        return logical->logical_index;
      }
    }
  }

  if (output_slot >= 0) {
    for (guint i = 0; i < stage->logical_outputs_len; ++i) {
      const SimaPluginLogicalTensor* logical = &stage->logical_outputs[i];
      if (logical->output_slot == output_slot) {
        return logical->logical_index;
      }
    }
  }

  return -1;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
