#pragma once

#ifdef __cplusplus
extern "C" {
#endif

const char* sima_neat_version(void);
const char* sima_neat_platform_version(void);
const char* sima_neat_abi_version(void);

extern const char sima_neat_version_marker[];
extern const char sima_neat_platform_version_marker[];
extern const char sima_neat_abi_version_marker[];

#ifdef __cplusplus
}
#endif
