#pragma once

#if !defined(SIMA_DEPRECATED)
#if defined(SIMA_ENABLE_LEGACY_DEPRECATION)
#if defined(_MSC_VER)
#define SIMA_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(__GNUC__) || defined(__clang__)
#define SIMA_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define SIMA_DEPRECATED(msg)
#endif
#else
#define SIMA_DEPRECATED(msg)
#endif
#endif
