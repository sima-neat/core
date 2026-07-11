#include "gst/GstInit.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

RUN_TEST(
    "unit_gst_data_adapter_edge_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal;

      gst_init_once();

      {
        SampleSpec spec;
        spec.kind = SampleMediaKind::Tensor;
        spec.media_type = "application/vnd.simaai.tensor";
        spec.format = "EVXX_FLOAT32";
        spec.dtype = TensorDType::Float32;
        spec.shape = {2, 3, 4};
        const std::string caps = caps_string_from_spec(spec);
        require_contains(caps, "rank=3", "tensor caps should contain rank");
        require_contains(caps, "dim0=2", "tensor caps should contain dim fields");
        require_contains(caps, "shape=(string)\"2,3,4\"",
                         "tensor caps should contain canonical shape");

        spec.tensor_envelope_transport = true;
        const std::string envelope_caps = caps_string_from_spec(spec);
        require_contains(envelope_caps, "representation=(string)tensor-set",
                         "tensor envelope caps should identify tensor-set representation");
        require_contains(envelope_caps, "storage=(string)tensorbuffer",
                         "tensor envelope caps should identify tensorbuffer storage");

        SampleSpec plain_spec = spec;
        plain_spec.tensor_envelope_transport = false;
        plain_spec.caps_key = capkey_from_spec(plain_spec);
        spec.caps_key = capkey_from_spec(spec);
        require(plain_spec.caps_key != spec.caps_key,
                "plain and envelope tensor caps must use distinct cache keys");
        GstCaps* plain_gst_caps = caps_from_spec(plain_spec);
        GstCaps* envelope_gst_caps = caps_from_spec(spec);
        require(
            !gst_structure_has_field(gst_caps_get_structure(plain_gst_caps, 0), "representation"),
            "plain tensor caps must not inherit envelope representation");
        const GstStructure* envelope_structure = gst_caps_get_structure(envelope_gst_caps, 0);
        require(g_strcmp0(gst_structure_get_string(envelope_structure, "representation"),
                          "tensor-set") == 0,
                "envelope tensor caps lost tensor-set representation");
        require(
            g_strcmp0(gst_structure_get_string(envelope_structure, "storage"), "tensorbuffer") == 0,
            "envelope tensor caps lost tensorbuffer storage");
        gst_caps_unref(plain_gst_caps);
        gst_caps_unref(envelope_gst_caps);
      }

      const Tensor rgb = make_color_tensor(4, 2, ImageSpec::PixelFormat::RGB, 0x12);
      require(tensor_bytes_tight(rgb) == 24, "tensor_bytes_tight should match RGB dense bytes");

      const Tensor nv12 = make_nv12_tensor(4, 2, 0x34);
      std::array<uint8_t, 10> too_small{};
      std::string copy_err;
      require(!copy_tensor_payload_to(nv12, too_small.data(), too_small.size(), &copy_err),
              "copy_tensor_payload_to should fail when composite plane data exceeds destination");
      require_contains(copy_err, "plane exceeds buffer",
                       "copy_tensor_payload_to failure reason mismatch");

      std::vector<uint8_t> exact(24, 0);
      copy_err.clear();
      require(copy_tensor_payload_to(rgb, exact.data(), exact.size(), &copy_err),
              "copy_tensor_payload_to should succeed for exact destination size");

      SampleSpec spec;
      spec.format = "gray";
      std::string spec_err;
      require(canonicalize_sample_spec(&spec, &spec_err),
              "canonicalize_sample_spec should succeed for valid SampleSpec");
      require(spec.format == "GRAY8", "canonicalize_sample_spec should normalize GRAY to GRAY8");

      Sample missing;
      SampleSpec field_spec;
      std::string field_err;
      require(!derive_field_spec(missing, &field_spec, &field_err),
              "derive_field_spec should fail for non-tensor Sample");
      require_contains(field_err, "missing tensor",
                       "derive_field_spec missing tensor error mismatch");

      Sample valid;
      valid.kind = SampleKind::Tensor;
      valid.tensor = rgb;
      valid.payload_type = PayloadType::Image;
      valid.format = "RGB";

      field_err.clear();
      require(derive_field_spec(valid, &field_spec, &field_err),
              "derive_field_spec should succeed for valid tensor field");
      require(!field_spec.caps_string.empty(),
              "derive_field_spec should produce non-empty caps_string");
    }));
