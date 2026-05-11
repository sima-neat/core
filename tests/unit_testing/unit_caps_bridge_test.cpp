#include "gst/GstInit.h"
#include "pipeline/TensorTypes.h"
#include "pipeline/internal/CapsBridge.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

RUN_TEST(
    "unit_caps_bridge_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal;

      gst_init_once();

      {
        GstCaps* caps = gst_caps_from_string(
            "video/x-raw,format=(string)RGB,width=(int)16,height=(int)8,framerate=(fraction)30/1");
        require(caps != nullptr, "failed to construct RGB caps");

        const TensorConstraint tc = tensor_constraint_from_caps(caps);
        require(tc.rank == 3, "RGB caps should map to rank 3 tensor");
        require(tc.shape.size() == 3, "RGB tensor shape should have 3 dimensions");
        require(tc.shape[0] == 8, "RGB tensor height mismatch");
        require(tc.shape[1] == 16, "RGB tensor width mismatch");
        require(tc.shape[2] == 3, "RGB tensor depth mismatch");
        require(!tc.dtypes.empty(), "RGB tensor should report at least one dtype");
        require(tc.dtypes.front() == TensorDType::UInt8, "RGB dtype should map to UInt8");
        require(tc.image_format.has_value(), "RGB image format should be populated");
        require(*tc.image_format == ImageSpec::PixelFormat::RGB, "RGB image format mismatch");

        const std::string debug = tensor_constraint_debug_string(tc);
        require_contains(debug, "rank=3", "RGB debug string rank mismatch");

        gst_caps_unref(caps);
      }

      {
        GstCaps* caps = gst_caps_from_string(
            "video/x-raw,format=(string)NV12,width=(int)12,height=(int)10,framerate=(fraction)24/"
            "1");
        require(caps != nullptr, "failed to construct NV12 caps");

        const TensorConstraint tc = tensor_constraint_from_caps(caps);
        require(tc.rank == 2, "NV12 caps should map to rank 2 tensor");
        require(tc.shape.size() == 2, "NV12 shape should have 2 dimensions");
        require(tc.shape[0] == 10, "NV12 tensor height mismatch");
        require(tc.shape[1] == 12, "NV12 tensor width mismatch");
        require(tc.allow_composite, "NV12 tensor should allow composite planes");

        gst_caps_unref(caps);
      }

      {
        GstCaps* caps = gst_caps_from_string(
            "application/"
            "vnd.simaai.tensor,format=(string)FP32,width=(int)4,height=(int)3,depth=(int)2");
        require(caps != nullptr, "failed to construct tensor caps");

        const TensorConstraint tc = tensor_constraint_from_caps(caps);
        require(tc.rank == 3, "tensor caps should map to rank 3");
        require(tc.shape.size() == 3, "tensor caps shape size mismatch");
        require(tc.shape[0] == 3 && tc.shape[1] == 4 && tc.shape[2] == 2,
                "tensor caps shape mismatch");
        require(!tc.dtypes.empty(), "tensor caps should report dtype");
        require(tc.dtypes.front() == TensorDType::Float32, "FP32 tensor caps dtype mismatch");

        gst_caps_unref(caps);
      }

      {
        GstCaps* caps = gst_caps_from_string(
            "application/vnd.simaai.tensor,dtype=(string)INT8,rank=(int)5,dim0=(int)2,"
            "dim1=(int)3,dim2=(int)4,dim3=(int)5,dim4=(int)6");
        require(caps != nullptr, "failed to construct rank-first tensor caps");

        const TensorConstraint tc = tensor_constraint_from_caps(caps);
        require(tc.rank == 5, "rank-first tensor caps rank mismatch");
        require(tc.shape == std::vector<int64_t>({2, 3, 4, 5, 6}),
                "rank-first tensor caps shape mismatch");
        require(!tc.dtypes.empty(), "rank-first tensor caps should report dtype");
        require(tc.dtypes.front() == TensorDType::Int8,
                "rank-first tensor caps dtype mismatch");

        gst_caps_unref(caps);
      }
    }));
