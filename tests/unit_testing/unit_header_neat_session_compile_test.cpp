#include "neat/session.h"
#include "test_main.h"

RUN_TEST("unit_header_neat_session_compile_test", ([] {
           simaai::neat::Session session;
           simaai::neat::RunOptions run_opt;
           simaai::neat::Sample sample;
           (void)session;
           (void)run_opt;
           (void)sample;
         }));
