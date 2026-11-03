# Installed Core startup contract

This independent consumer uses installed Core/Scheduler headers and libraries.
It does not use Core-private headers, initialize GStreamer manually, submit a
kernel, load a model, or claim numerical/device-retirement qualification.

Configure against one matching installed full-production cohort:

```sh
cmake -S tests/installed_startup -B build-installed-startup \
  -DCMAKE_PREFIX_PATH=/path/to/installed/usr \
  -DSIMA_STARTUP_PLUGIN_DIR=/path/to/installed/usr/lib/aarch64-linux-gnu/neat/gst-plugins
cmake --build build-installed-startup
```

For cross compilation, also pass the same SDK toolchain as Core. Inspect the
executables with `file`; execute AArch64 binaries only on the matching target,
using the configured CMake launcher/DevKit wrapper rather than running locally.
The same sources are registered in Core's regular guarded CTest suite.

Each case gets a new process, a temporary registry and a plugin directory that
exposes only the real installed allocator and MLA provider. System discovery is
redirected to an empty directory. The allocator remains a symlink to the installed
library, preserving the shared memory authority. Only the MLA plugin is copied
into the isolated discovery root, with a byte-for-byte check; its dependencies
still resolve from the installed cohort through paths set before process launch.
Cases prove:

- Public numerical node options cross the Core library boundary unchanged.
- Core registers the linked generic runner, resolves the exact MLA provider and
  admits a real canonical A65 BoxDecode request with no legacy numerical plugin.
- Omitting the otherwise-present MLA provider gives the exact missing-factory error.
- An executable-only registration double that returns success without registering
  the generic runner is rejected by actual Core startup validation.

The negative-control double is linked only into its dedicated test executable.
The current full-production Core profile requires MLA; this test does not change
that policy or require MLA in standalone optional Runner/Scheduler profiles.
Missing installed fixtures fail, not skip. The suite must not be reported as
passed on a host dependency profile that does not include the real MLA provider.
