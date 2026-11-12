# Build And Validation

Validate at the highest level available on the current host. A successful compile or Python import
proves packaging and API use, but it does not prove PCIe transport or model execution.

## Package Checks

For a C++ application, the development package must provide the selected API header, static library,
and CMake package. The runtime package provides the host plugin and setup command.

```bash
dpkg-query -W sima-pcie-host sima-pcie-host-dev
command -v pcie-setup.sh
```

On Windows, verify the native installation and plugin in PowerShell:

```powershell
$root = "C:\Program Files\SimaPCIeHost"
$gst = "C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-inspect-1.0.exe"
$env:PATH = "$root\bin;$(Split-Path $gst);$env:PATH"
Test-Path "$root\include\simaai\neat\pcie\Model.h"
& "$root\bin\sima-pcie-host-doctor.cmd"
& $gst "$root\bin\gstreamer-1.0\gstneatpciehost.dll"
```

Before the first connected-card run on Windows, provision or verify passwordless
SSH with `& "$root\bin\pcie-setup.cmd"`. Pass `-Hosts 10.0.0.2` if automatic
management-link discovery is unavailable. Let OpenSSH prompt interactively;
never put a card password in source code or a command argument.

Check `Model.h` for `pcie::Model` or `genai/GenAIModel.h` for GenAI under
`/usr/include/simaai/neat/pcie/`. For `pcie::Model`, also run `gst-inspect-1.0 neatpciehost`.
GenAI requires matching GenAI-capable host and card packages; plugin inspection does not
validate that backend.

For Python, use the environment into which the PCIe wheel was installed:

```bash
~/pyneatpcie/bin/python -c \
  'import pyneatpcie as p; print(p.__version__); print(p.Model)'
```

Do not report Python validation as successful when only the system interpreter was checked but the
wheel lives in another environment.

For GenAI, verify the selected interpreter exposes the separate API:

```bash
~/pyneatpcie/bin/python -c \
  'from pyneatpcie import genai; print(genai.GenAIModel); print(genai.GenerationRequest)'
```

On Windows, this import check confirms only that the bindings are installed. Customer acceptance
also requires the release-qualified Platform package and the connected-card C++ and Python GenAI
matrix.

## C++ Build

Use C++20 and the installed CMake package:

```cmake
cmake_minimum_required(VERSION 3.16)
project(pcie_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaPCIeHost REQUIRED CONFIG)

add_executable(pcie_model main.cpp)
target_link_libraries(
  pcie_model
  PRIVATE SimaPCIeHost::sima_neat_pcie_host
)
```

Add OpenCV components only when the application uses `cv::Mat` or OpenCV image loading. Configure
and build natively on the host:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

With Visual Studio Build Tools on Windows, use the installed package prefix:

```powershell
cmake -S . -B build -A x64 -DCMAKE_PREFIX_PATH="C:\Program Files\SimaPCIeHost"
cmake --build build --config Release
```

## Connected-Card Validation

Run hardware checks only when a card, compatible model, and permission to run the workload are
available. Confirm passwordless SSH and card reachability before loading a model. Do not rerun
provisioning or alter SSH configuration unless the user requested setup.

For GenAI, follow `genai.md`: construction loads the card model immediately. Use connection
timeouts and close after validating a representative result; do not select a `pcie::Model` queue or
call `build()`. Check streaming, cancellation, or media when the application uses them.

For `pcie::Model`, use a bounded smoke sequence:

1. Confirm the model archive exists.
2. Construct `Model` and inspect `info()` without touching the card.
3. Select a free queue and call `build()` with a finite readiness timeout.
4. Run one representative request with a finite inference timeout.
5. Validate output count, route, dtype, shape, and size.
6. Close the model even when validation fails.

The host package and card-side Neat Library must come from compatible releases. A build failure can
also indicate SSH/SCP failure, an occupied queue, an invalid archive, or card-side pipeline startup
failure. Preserve the original error and report the selected card and, for `pcie::Model`, queue.

## Packaged Examples

When the PCIe extras bundle is present, prefer its matching-release tutorials over copied examples:

- `share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/`
- `share/sima-pcie-host/tutorials/025_run_pcie_inference_async/`
- `share/sima-pcie-host/tutorials/026_run_multiple_models/`
- `share/sima-pcie-host/tutorials/027_run_mla_only_int8/`
- `share/sima-pcie-host/tutorials/028_wrap_external_tensor_memory/`
- `share/sima-pcie-host/tutorials/029_run_genai_over_pcie/`

Tutorial 026 uses separate `Model` objects on distinct queues. It does not require the excluded
`pcie::Runtime` API. Tutorial 029 requires a Linux host; do not use it for native Windows.
