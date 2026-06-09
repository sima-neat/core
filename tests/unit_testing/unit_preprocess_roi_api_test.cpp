#include "pipeline/StageRun.h"
#include "test_main.h"
#include "test_utils.h"

#include <type_traits>
#include <vector>

namespace {

using PreprocRoiListFn =
    simaai::neat::TensorList (*)(const std::vector<cv::Mat>&, const simaai::neat::Model&,
                                 const std::vector<simaai::neat::PreprocessRoi>&);

} // namespace

RUN_TEST("unit_preprocess_roi_api_test", ([] {
           PreprocRoiListFn fn = &simaai::neat::stages::Preproc;
           (void)fn;

           simaai::neat::PreprocessRoi roi;
           roi.batch_index = 2;
           roi.x = -4;
           roi.y = 5;
           roi.width = 32;
           roi.height = 16;

           require(roi.batch_index == 2 && roi.x == -4 && roi.y == 5 && roi.width == 32 &&
                       roi.height == 16,
                   "PreprocessRoi API: field assignment mismatch");
           static_assert(std::is_aggregate_v<simaai::neat::PreprocessRoi>,
                         "PreprocessRoi should remain a simple aggregate");
         }));
