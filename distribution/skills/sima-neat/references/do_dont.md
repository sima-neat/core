# Do and Don't

## Do

- Use C++20.
- Include only public headers.
- Keep pipeline assembly explicit and readable.
- Validate assumptions (format, shape, pull status).
- Provide runnable build/test commands with generated code.
- Check hardware/plugin prerequisites for specialized flows (`element_exists(...)`).
- Handle skip/fallback paths explicitly when optional plugins are unavailable.

## Don't

- Don't use private/internal headers.
- Don't hardcode environment-specific plugin paths unless requested.
- Don't assume all outputs are tensor-only; handle bundles when applicable.
- Don't ignore timeout/closed/error branches in async pull loops.
- Don't bypass `find_package(SimaNeat REQUIRED)` in CMake examples.
- Don't hardcode fixed caps unless intentionally using `caps_override`.
- Don't mix stream metadata keys inconsistently in multistream joins/schedulers.
