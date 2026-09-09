# Pre-board validation

## Results

- Internals: ARM64 production targets built and installed into a new component staging tree.
- Core: ARM64 shared/static libraries, audit/PCIe tools and 15 affected configured test targets compiled against that installation.
- Package/source checks: **145 passed** (existing Internals packaging, isolation, direct MLA/TVM and decoder/encoder source checks).
- Staged contract audit: **58 ARM64 ELF files**; no dispatcher/ZeroMQ dependencies, no build-host runtime paths, no SiMa plugins in the system GStreamer directory. Neat-private dynamic dependencies resolve in the staged generation.
- Source/installed plugin ABI copies and pinned MLA/CVU/codec UAPI hashes agree. Core's tensor-meta shim resolves the freshly installed provider, not the older SDK header.
- Supplementary native DMA-BUF/CVU checks: **4 passed**, including the existing syscall and cross-DSO epoch tests.
- Supplementary native MLA: lifecycle and public-header checks passed without the optional golden fixture. The fixture-specific test is **not qualified**: its exact ResNet ELF is absent. Trying a different ResNet artifact produced golden-metadata mismatches and is not valid regression evidence.
- The graph-migration variant target is not enabled in this configuration.
- **No ARM64 binaries ran locally, no board deployment, no full CI/CD run, and no new FPS claim.**

## Reproduce the build

Run in the cross-compilation container, from a parent directory containing sibling
`core` and `internals` checkouts. `KERNEL` points to the unchanged kernel commit in
`runtime-lock.json`; `MLART` points to the existing companion MLA-RT source.
Use a new empty staging path. Do not install into the host SDK or the demo image.

```bash
ROOT=$PWD
: "${KERNEL:?set matching kernel source path}"
: "${MLART:?set companion MLA-RT source path}"
STAGE=$ROOT/stage
TOOLCHAIN=$ROOT/internals/cmake/toolchains/docker-modalix-aarch64.cmake

cmake -S internals -B build-internals -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/aarch64-linux-gnu \
  -DSIMANEAT_BUILD_PROFILE=docker -DBUILD_TESTING=OFF \
  -DSIMANEAT_JSONCPP_INCLUDE_DIR=/opt/toolchain/aarch64/modalix/usr/include/jsoncpp \
  -DSIMAAI_MLA_RT_DIR="$MLART" \
  -DSIMAAI_LINUX_UAPI_DIR="$KERNEL/include/uapi" \
  -DMLA_KERNEL_UAPI_DIR="$KERNEL/include/uapi" \
  -DNEAT_INSTALL_SYSTEM_GST_SYMLINKS=OFF
cmake --build build-internals --target installable_neat_internals qmla-package-audit -j8
for component in neat-runtime neat-gst-plugins neat-internals-dev; do
  DESTDIR="$STAGE" cmake --install build-internals --component "$component"
done

# Avoid a toolchain-environment -I.../usr/include/simaai overriding the
# selected package's imported system include directories.
CXXFLAGS="${CXXFLAGS//-I\/opt\/toolchain\/aarch64\/modalix\/usr\/include\/simaai/}" \
cmake -S core -B build-core -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/aarch64-linux-gnu \
  -DNeatInternals_DIR="$STAGE/usr/lib/aarch64-linux-gnu/cmake/NeatInternals" \
  -DSIMANEAT_USE_EXPORTED_NEAT_INTERNALS=ON \
  -DSIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS=ON \
  -DSIMANEAT_REQUIRE_LLIMA_ARTIFACTS=OFF -DSIMANEAT_BUILD_PYTHON=OFF
cmake --build build-core --target sima_neat sima_neat_static \
  neat-dmabuf-plan-audit pcie-pipeline-builder -j8
for component in core dev; do
  DESTDIR="$STAGE" cmake --install build-core --component "$component"
done
```

This validates the Graph/Core build, not GenAI or Python bindings. The package
launch payload is installed exactly as specified in Internals `debian/rules`:
`sima-neat-run`, its `sima-neat-gst` symlink and `build-filtered-system-path`.
A raw unqualified CMake install also includes standalone component outputs;
use the named package components for the candidate runtime view.
Firmware was pinned, not rebuilt or installed. No kernel version was changed.
