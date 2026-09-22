#include "boxdecode_yolox_seg_pose_raw_fixture.h"
#include "test_main.h"

RUN_TEST("unit_boxdecode_yolox_seg_pose_raw_test", ([] {
           for (bool threaded : {false, true}) {
             for (bool stretch : {false, true}) {
               yolox_test::RawHeads fixture(2.0f, stretch ? 1.125f : 2.0f, 0.0f,
                                            stretch ? 0.0f : -280.0f, threaded);
               const auto reference = fixture.decode();
               fixture.verify_reference();
               for (int run = 0; run < 100; ++run) {
                 require(fixture.decode() == reference, "raw-head results must repeat exactly");
               }
             }
           }
         }));
