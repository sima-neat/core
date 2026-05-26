#include "neat/version.h"

#ifndef SIMA_NEAT_VERSION_STRING
#define SIMA_NEAT_VERSION_STRING "unknown"
#endif

#ifndef SIMA_NEAT_PLATFORM_VERSION_STRING
#define SIMA_NEAT_PLATFORM_VERSION_STRING "unknown"
#endif

#ifndef SIMA_NEAT_ABI_VERSION_STRING
#define SIMA_NEAT_ABI_VERSION_STRING "unknown"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SIMA_NEAT_EXPORT __attribute__((visibility("default")))
#define SIMA_NEAT_USED __attribute__((used))
#else
#define SIMA_NEAT_EXPORT
#define SIMA_NEAT_USED
#endif

extern "C" {

SIMA_NEAT_EXPORT const char* sima_neat_version(void) {
  return SIMA_NEAT_VERSION_STRING;
}

SIMA_NEAT_EXPORT const char* sima_neat_platform_version(void) {
  return SIMA_NEAT_PLATFORM_VERSION_STRING;
}

SIMA_NEAT_EXPORT const char* sima_neat_abi_version(void) {
  return SIMA_NEAT_ABI_VERSION_STRING;
}

SIMA_NEAT_EXPORT SIMA_NEAT_USED const char sima_neat_version_marker[] =
    "sima-neat-version=" SIMA_NEAT_VERSION_STRING;

SIMA_NEAT_EXPORT SIMA_NEAT_USED const char sima_neat_platform_version_marker[] =
    "sima-neat-platform-version=" SIMA_NEAT_PLATFORM_VERSION_STRING;

SIMA_NEAT_EXPORT SIMA_NEAT_USED const char sima_neat_abi_version_marker[] =
    "sima-neat-abi-version=" SIMA_NEAT_ABI_VERSION_STRING;

}
