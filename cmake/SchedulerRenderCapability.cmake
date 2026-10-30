include_guard(GLOBAL)

# Core publishes int32 box coordinates. Standalone float-only Scheduler profiles
# remain supported, but are not the production provider for these Core graphs.
option(SIMANEAT_REQUIRE_INT32_RENDERER
  "Require Scheduler's typed int32 renderer for Core box/mask publication"
  ${SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS})
if(SIMANEAT_REQUIRE_INT32_RENDERER AND
  (NOT SimaScheduler_RENDER_ENABLED OR NOT SimaScheduler_RENDER_INT32_CAPABLE))
  message(FATAL_ERROR
    "Core box/mask publication requires a Scheduler package with typed int32 "
    "renderer capability. Build Scheduler against matching canonical Kernels "
    "and Runner packages; a float-only renderer cannot consume Core coordinates. "
    "SIMANEAT_REQUIRE_INT32_RENDERER=OFF is only for profiles that do not use "
    "Core box/mask rendering.")
endif()

