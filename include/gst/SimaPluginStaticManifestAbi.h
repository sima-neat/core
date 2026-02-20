#pragma once

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pipeline-level static manifest context contract for SIMA model plugins.
 * This is an ABI contract header; plugin/dispatcher repos should carry a synced
 * copy and must not link against framework internals.
 *
 * Context type:
 *   "sima.model.manifest.v1"
 *
 * Required fields in GstContext structure:
 *   - "manifest_version" (uint)
 *
 * Runtime transport fields:
 *   - "manifest_accessor_v1" (pointer to ABI-safe accessor table), preferred
 *   - "manifest_json" (string), legacy fallback payload
 *
 * Optional fields:
 *   - "session_id" (string)
 *   - "model_id" (string)
 */

#define SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE "sima.model.manifest.v1"
#define SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION ((guint)1)

#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_VERSION "manifest_version"
#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_JSON "manifest_json"
#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_SESSION_ID "session_id"
#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_MODEL_ID "model_id"
#define SIMA_PLUGIN_STATIC_MANIFEST_KEY_ACCESSOR_V1 "manifest_accessor_v1"

typedef struct SimaPluginStaticManifestAbiHeader {
  guint abi_version;
  guint manifest_version;
} SimaPluginStaticManifestAbiHeader;

/*
 * ABI-safe accessor table carried via GstContext as a pointer.
 *
 * Lifetime:
 * - The owner of this table must outlive pipeline/plugin usage.
 * - Plugins must treat returned string pointers as borrowed; copy if needed.
 */
typedef struct SimaPluginStaticManifestAccessorV1 {
  guint abi_version;
  gpointer user_data;

  guint (*manifest_version)(gpointer user_data);
  const gchar* (*manifest_json)(gpointer user_data);
  const gchar* (*session_id)(gpointer user_data);
  const gchar* (*model_id)(gpointer user_data);
  const gchar* (*stage_json_by_element_name)(gpointer user_data, const gchar* element_name);
  const gchar* (*stage_json_by_logical_stage_id)(gpointer user_data, const gchar* logical_stage_id);
} SimaPluginStaticManifestAccessorV1;

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

static inline const SimaPluginStaticManifestAccessorV1*
sima_plugin_manifest_context_accessor(const GstContext* context) {
  const GstStructure* structure = sima_plugin_manifest_context_structure(context);
  if (!structure) {
    return NULL;
  }
  const GValue* accessor_val =
      gst_structure_get_value(structure, SIMA_PLUGIN_STATIC_MANIFEST_KEY_ACCESSOR_V1);
  if (!accessor_val || !G_VALUE_HOLDS_POINTER(accessor_val)) {
    return NULL;
  }
  const SimaPluginStaticManifestAccessorV1* accessor =
      (const SimaPluginStaticManifestAccessorV1*)g_value_get_pointer(accessor_val);
  if (!accessor || accessor->abi_version != SIMA_PLUGIN_STATIC_MANIFEST_ABI_VERSION) {
    return NULL;
  }
  return accessor;
}

static inline const gchar* sima_plugin_manifest_lookup_stage_by_element_name(
    const SimaPluginStaticManifestAccessorV1* accessor, const gchar* element_name) {
  if (!accessor || !accessor->stage_json_by_element_name || !element_name || !*element_name) {
    return NULL;
  }
  return accessor->stage_json_by_element_name(accessor->user_data, element_name);
}

static inline const gchar*
sima_plugin_manifest_lookup_stage_by_logical_id(const SimaPluginStaticManifestAccessorV1* accessor,
                                                const gchar* logical_stage_id) {
  if (!accessor || !accessor->stage_json_by_logical_stage_id || !logical_stage_id ||
      !*logical_stage_id) {
    return NULL;
  }
  return accessor->stage_json_by_logical_stage_id(accessor->user_data, logical_stage_id);
}

#ifdef __cplusplus
} /* extern "C" */
#endif
