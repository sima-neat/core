#pragma once

#include "gst/GstInit.h"
#include "model_archive_test_utils.h"

#include <gst/gst.h>

namespace sima_test {

// Cold compiler/model-contract tests need real metadata and host GStreamer
// factories, but do not submit a model or qualify device-backend startup. Keep
// production startup policy in the isolated installed-startup consumer tests.
inline void initialize_cold_contract_gstreamer() {
  ScopedEnvVar manual_init("SIMA_ALLOW_GST_INIT", "1");
  gst_init(nullptr, nullptr);
  simaai::neat::gst_init_once();
  // CTest may disable initial system discovery; scan only Core's selected
  // real plugin cohort after its normal path setup, never fake backend plugins.
  if (const char* paths = g_getenv("GST_PLUGIN_SYSTEM_PATH_1_0")) {
    gchar** directories = g_strsplit(paths, G_SEARCHPATH_SEPARATOR_S, -1);
    for (gchar** directory = directories; *directory; ++directory) {
      if (**directory)
        gst_registry_scan_path(gst_registry_get(), *directory);
    }
    g_strfreev(directories);
  }
}

} // namespace sima_test
