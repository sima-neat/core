# SiMa NEAT PCIe Host

Host-side C++ runtime and API for NEAT PCIe coprocessor execution.

This package is the WP11 successor to the old PipelineSession `SimaPCIe` host
app. It keeps the useful control-plane ideas from that prototype, but uses the
production NEAT PCIe lifecycle:

- card process: one `/usr/bin/pcie-pipeline-builder --queue N` instance per
  active PCIe queue
- readiness: `/run/sima-neat/pcie/qN.status`
- logs: `/var/log/sima-neat/pcie/qN.log`
- host plugin: `neatpciehost`

## Public API

Public headers install under:

```text
/usr/include/simaai/neat/pcie/
```

Main type:

```cpp
#include <simaai/neat/pcie/SimaPCIeHost.h>

simaai::neat::pcie::SimaPCIeHost host;
```

The API uses PCIe-host-owned `ModelOptions`, `Tensor`, and `TensorList` types.
Applications that are already built only need the `sima-pcie-host` runtime
package; applications compiling against this API use `sima-pcie-host-dev`.
SimaPCIeHost serializes only the restricted WP9 model-options schema before
launching the card-side builder; no full NEAT core `Model`, `Run`, or `Graph`
API is part of this package surface.

## SSH Provisioning

The PCIe host expects passwordless SSH to the PCIe card during normal operation.
The package installs `pcie-setup.sh`, which follows the SDK DevKit pattern:
create or reuse an ed25519 key, add PCIe card host keys to `known_hosts`, and
install the public key with `ssh-copy-id`.

By default it discovers PCIe management links from local IPv4 addresses: each
host-side `10.0.N.1` address maps to card address `10.0.N.2`.

```bash
pcie-setup.sh
```

If discovery is not available, pass explicit hosts or a static management
address range:

```text
10.0.0.2
10.0.1.2
...
10.0.9.2
```

Useful variants:

```bash
pcie-setup.sh --hosts "10.0.0.2 10.0.1.2"
pcie-setup.sh --range 0-3
pcie-setup.sh --key ~/.ssh/sima_neat_pcie_ed25519
pcie-setup.sh --password '<bootstrap-password>'
```

Password/`sshpass` is only a bootstrap path for installing the key. Runtime
SimaPCIeHost APIs use passwordless SSH. If `~/.ssh/sima_neat_pcie_ed25519` exists,
the PCIe host automatically passes it to `ssh`/`scp`; otherwise it falls back to
the user's normal SSH configuration.

### ConnectionOptions

Configures SSH/SCP access to the card.

```cpp
struct ConnectionOptions {
  std::string card_host;      // Optional explicit card IP/host for SSH/SCP.
  int card_id = 0;            // PCIe card/plugin index; default host is 10.0.<card_id>.2.
  std::string user = "sima";
  int queue = 0;              // PCIe queue, 0..5.
  std::string card_gst_debug; // Optional card-side GST_DEBUG spec for pcie-pipeline-builder.
  std::string card_gst_debug_file;
  bool card_gst_debug_no_color = true;
};
```

`card_id` is the hardware/plugin card number passed to `neatpciehost` as
`card-number`. It is also used as the default SSH management endpoint when
`card_host` is empty: `card_id = N` maps to `10.0.N.2`. Set `card_host`
explicitly when the management address is known or non-standard.

Set `card_gst_debug` during PCIe bring-up to start the card-side
`pcie-pipeline-builder` under GStreamer debug logging. When enabled and
`card_gst_debug_file` is empty, logs are written to
`/var/log/sima-neat/pcie/qN.gst.log` on the card.

```cpp
pcie::ConnectionOptions conn;
conn.card_gst_debug = "neatpciesrc:7,neatpciesink:7,neatprocesscvu:5";
```

The normal production launch path is unchanged when `card_gst_debug` is empty.

### ModelInfo

Returned by `load_metadata()` and `init_pipeline()`.

```cpp
struct TensorInfo {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t size_bytes = 0;
};

struct ModelInfo {
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  bool has_preprocess = false;
  bool has_boxdecode = false;
};
```

### Payloads And Results

Tensor input always travels as tensor-set/tensorbuffer caps with
`GstSimaTensorSetMeta`, even for a single tensor. The public API uses PCIe host
transport tensors. `TensorList` is the real contract for tensor-set transport.
Single `Tensor` overloads are convenience wrappers. Image input is represented
as a `Tensor` with `image_format` and optional plane metadata set, not as a
separate PCIe-specific payload type.

### Status

```cpp
enum class PipelineState {
  Uninitialized,
  Starting,
  Ready,
  Failed,
  Stopping,
  Exited,
};

struct Status {
  PipelineState state = PipelineState::Uninitialized;
  int queue = -1;
  std::string message;
  std::string error_code;
};
```

### SimaPCIeHost Methods

```cpp
class SimaPCIeHost {
public:
  explicit SimaPCIeHost(ConnectionOptions connection = {});

  ModelInfo load_metadata(const std::string& model_path,
                          const ModelOptions& options = {});
  ModelInfo init_pipeline(const std::string& model_path,
                          const ModelOptions& options = {},
                          int readiness_timeout_ms = 180000);

  void stop();
  Status status() const;

  bool push(const Tensor& tensor);
  bool push(const TensorList& tensors);
  std::optional<TensorList> pull(int timeout_ms = -1);
  TensorList run(const Tensor& tensor, int timeout_ms = -1);
  TensorList run(const TensorList& tensors, int timeout_ms = -1);
};
```

`load_metadata(...)` parses the local model archive, validates representable
options, updates cached model information, and does not touch the card.
Metadata uses the logical full-model tensor contract, commonly `FP32` when
quant/cast and dequant/postcast stages are present. `init_pipeline(...)` does
the same metadata resolution plus model upload, card runtime startup, readiness
wait, and local host channel setup.

`run(...)` is the simplest synchronous API and is equivalent to `push(...)`
followed by `pull()`. For pipelined use, call `push(...)` and `pull()` directly.
The host channel receives asynchronously from `appsink` and stores results in an
internal queue.

During bring-up, `init_pipeline()` keeps an internal five-second stabilization delay
after the card status reaches `ready`, matching the old PipelineSession host
behavior. This is not exposed as a public option.

`stop()` sends `SIGTERM` to the card-side builder and waits briefly for a clean
exit. It does not send `SIGKILL`; if the remote process ignores `SIGTERM`, the
PID file remains and the queue stays visibly busy for diagnosis.

### Minimal Tensor Example

```cpp
namespace pcie = simaai::neat::pcie;

pcie::ConnectionOptions conn;
conn.card_host = "10.0.0.2";
conn.queue = 0;

pcie::SimaPCIeHost host(conn);
pcie::ModelInfo model_info = host.init_pipeline("model.tar.gz");
pcie::Tensor input = make_input_tensor_somehow(model_info.inputs.front());
pcie::TensorList output = host.run(input);
host.stop();
```

### Minimal Image Example

```cpp
namespace pcie = simaai::neat::pcie;

pcie::ModelOptions options;
options.preprocess.kind = pcie::InputKind::Image;

pcie::SimaPCIeHost host;
host.init_pipeline("model.tar.gz", options);

pcie::Tensor image = make_rgb_image_tensor_somehow(rgb, 1920, 1080);
pcie::TensorList output = host.run(image, 5000);
host.stop();
```

## Build

The host PCIe plugin is consumed as a prebuilt artifact from the internals repo.
By default `build.sh` uses `../deps/manifest.json`, resolves the `internals`
dependency with snap semantics, and downloads the matching Vulcan-hosted PCIe
host tarball from the `internals` artifact tree when the local artifact is
missing. For snap branch builds it tries the current core branch first, then
falls back to `develop`; tag builds require the matching tag artifact. Automatic
download reads `latest.tag`, verifies `pcie-host-artifact-<multiarch>.tar.gz`
with its `.sha256`, and extracts it locally.

The extracted/local layout is:

```text
artifacts/x86_64-linux-gnu/libgstneatpciehost.so
artifacts/x86_64-linux-gnu/include/gst/SimaTensorSetMetaAbi.h
artifacts/aarch64-linux-gnu/libgstneatpciehost.so
artifacts/aarch64-linux-gnu/include/gst/SimaTensorSetMetaAbi.h
```

`build.sh` detects the host multiarch with `dpkg-architecture`, copies the
plugin and header into `build/artifacts/neatpciehost/<multiarch>/`, and passes
those staged paths to CMake. CMake does not include headers from a sibling
`internals` checkout.

```bash
./build.sh --with-tests --no-deb
./build.sh --with-tests
```

`build.sh` writes the locally built packages and a copy of the installer into
`dist/`:

```text
dist/sima-pcie-host_<version>_<arch>.deb
dist/sima-pcie-host-dev_<version>_<arch>.deb
dist/install_pciehost.sh
```

Install the locally built packages on the host with:

```bash
dist/install_pciehost.sh
```

The installer looks for the PCIe host debs in the same directory as the script.
It runs `pcie-setup.sh` at the end. Setup is interactive by default
and can prompt while provisioning passwordless SSH for `init_pipeline()`. Pass
extra setup arguments when discovery is not available:

```bash
dist/install_pciehost.sh --setup-args "--hosts 10.0.0.2"
```

For `load_metadata()` investigation, no PCIe card or SSH setup is required, so
the setup step can be skipped:

```bash
dist/install_pciehost.sh --skip-setup
```

To print metadata on a laptop without a PCIe card, build examples and run:

```bash
./build.sh --with-examples
sima_pcie_metadata_dump --model /path/model.tar.gz
```

The package installs the real plugin under:

```text
/usr/lib/<multiarch>/sima-pcie-host/gst-plugins/libgstneatpciehost.so
```

and creates the GStreamer discovery symlink:

```text
/usr/lib/<multiarch>/gstreamer-1.0/libgstneatpciehost.so
```

The standalone CMake target is:

```text
sima_neat_pcie_host
```

Installed customer apps should consume it with:

```cmake
cmake_minimum_required(VERSION 3.16)
project(pcie_host_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaPCIeHost REQUIRED CONFIG)

add_executable(pcie_host_app main.cpp)
target_link_libraries(pcie_host_app PRIVATE SimaPCIeHost::sima_neat_pcie_host)
```

The package config is installed under:

```text
/usr/lib/<multiarch>/cmake/SimaPCIeHost/
```

The runtime Debian package is:

```text
sima-pcie-host
```

It contains the host GStreamer plugin, runtime setup script, and runtime
dependencies.

The development Debian package is:

```text
sima-pcie-host-dev
```

It contains `libsima_neat_pcie_host.a`, public headers, and the CMake package
config. Because the installed target is a static library, `sima-pcie-host-dev`
depends on the GStreamer development packages. Customer applications do not need
the NEAT core or internals source tree; `find_package(SimaPCIeHost)` rediscovers
the local GStreamer link flags with `pkg-config`.

## V1 Scope

Implemented in this first WP11 cut:

- model metadata through core archive/MPK parsing
- `ModelOptions` to WP9 JSON serialization
- SSH/SCP launch of `pcie-pipeline-builder`
- WP9 status-file readiness polling
- host `appsrc ! queue ! neatpciehost ! appsink` channel
- tensor push through tensor-set/tensorbuffer caps and `GstSimaTensorSetMeta`
- image tensor push for RGB/BGR/GRAY8/NV12/I420

Known follow-up:

- validate tensor-set metadata attachment on the target host runner
- hardware smoke must validate all routes on Modalix PCIe hardware
