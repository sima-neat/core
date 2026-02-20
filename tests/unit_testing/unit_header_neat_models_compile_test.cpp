#include "neat/models.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_models_compile_test", ([] {
           simaai::neat::Model::Options model_opt;
           simaai::neat::Model::SessionOptions session_opt;
           simaai::neat::stages::BoxDecodeOptions decode_opt;
           (void)model_opt;
           (void)session_opt;
           (void)decode_opt;
         }));
