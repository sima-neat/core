/**
 * @file
 * @ingroup pipeline
 * @brief Compiler-portable deprecation macros used framework-wide.
 *
 * Defines `SIMA_DEPRECATED(msg)`. When `SIMA_ENABLE_LEGACY_DEPRECATION` is
 * defined the macro expands to the appropriate compiler attribute
 * (`__declspec(deprecated)` on MSVC, `__attribute__((deprecated))` on GCC/Clang)
 * so call sites generate deprecation warnings; otherwise it expands to nothing
 * so production builds remain warning-free. Headers can pre-define
 * `SIMA_DEPRECATED` themselves to override this default.
 */
#pragma once

/**
 * @def SIMA_DEPRECATED
 * @brief Mark a symbol as deprecated with an explanatory message.
 *
 * Usage: `SIMA_DEPRECATED("Use Foo() instead") void OldFoo();`. Expands to a
 * compiler-specific deprecation attribute when
 * `SIMA_ENABLE_LEGACY_DEPRECATION` is set, otherwise expands to nothing.
 *
 * @param msg String literal explaining what to use instead.
 */
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
