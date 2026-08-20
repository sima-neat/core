# Build And Validation

Validate at the highest level available on the current host. A successful compile or Python import
proves packaging and API use, but it does not prove PCIe transport or model execution.

## Package Checks

For a C++ application, the development package must provide the header, static library, and CMake
package. The runtime package provides the host plugin and setup command.

```bash
dpkg-query -W sima-pcie-host sima-pcie-host-dev
test -r /usr/include/simaai/neat/pcie/Model.h
command -v pcie-setup.sh
gst-inspect-1.0 neatpciehost
```

For Python, use the environment into which the PCIe wheel was installed:

```bash
~/pyneatpcie/bin/python -c \
  'import pyneatpcie as p; print(p.__version__); print(p.Model)'
```

Do not report Python validation as successful when only the system interpreter was checked but the
wheel lives in another environment.

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

## Connected-Card Validation

Run hardware checks only when a card, compatible compiled model, and permission to use a queue are
available. Confirm passwordless SSH and card reachability before occupying a queue. Do not rerun
provisioning or alter SSH configuration unless the user requested setup.

Use a bounded smoke sequence:

1. Confirm the model archive exists.
2. Construct `Model` and inspect `info()` without touching the card.
3. Select a free queue and call `build()` with a finite readiness timeout.
4. Run one representative request with a finite inference timeout.
5. Validate output count, route, dtype, shape, and size.
6. Close the model even when validation fails.

The host package and card-side Neat Library must come from compatible releases. A build failure can
also indicate SSH/SCP failure, an occupied queue, an invalid archive, or card-side pipeline startup
failure. Preserve the original error and report the selected card and queue.

## Packaged Examples

When the PCIe extras bundle is present, prefer its matching-release tutorials over copied examples:

- `share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/`
- `share/sima-pcie-host/tutorials/025_run_pcie_inference_async/`
- `share/sima-pcie-host/tutorials/026_run_multiple_models/`

Tutorial 026 uses separate `Model` objects on distinct queues. It does not require the excluded
`pcie::Runtime` API.
