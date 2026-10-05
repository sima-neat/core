---
title: Peripheral catalog
description: List the DevKit peripherals reported by the board-local catalog service
sidebar_position: 8
---

# Peripheral catalog

Use the peripheral catalog to inspect devices currently attached to the local
DevKit. The API is available in C++ and Python and returns the same typed
snapshot in both languages.

The catalog belongs to SiMa Sentinel (`simaai-sentinel.service`), the
board-local daemon that serves it as `GET /v1/peripherals`. Sentinel reports
hardware facts only. Each `list()` call performs one bounded request to
Sentinel and then Core decides which camera modes `CameraInput` supports. Core
does not scan hardware, cache a second catalog, or fall back to another
discovery path.

## List peripherals

Python:

```python
import pyneat

catalog = pyneat.peripherals.list()
for peripheral in catalog:
    print(peripheral.id, peripheral.type)
```

C++:

```cpp
#include <neat.h>

auto catalog = simaai::neat::peripherals::list();
for (const auto& peripheral : catalog) {
  // Use peripheral.id and peripheral.type.
}
```

The returned catalog is iterable. It also contains:

| Field | Meaning |
| --- | --- |
| `revision` | Changes whenever `devices` or `errors` change. Compare it for equality only. |
| `observed_at` | When the scan behind this snapshot started; unset until Sentinel's first scan completes. Until then the catalog is empty even if devices are attached. |
| `errors` | Providers that failed in the latest scan, each with `provider`, `code`, and `reason`. A failed provider's devices from its last successful scan stay in `devices`. |
| `devices` | The peripherals. |

A catalog may contain zero devices. That is a successful result.

## Camera details

When `peripheral.type == "camera"`, `peripheral.camera` contains:

| Field | Meaning |
| --- | --- |
| `camera_name` | Optional exact libcamera name accepted by `CameraInputOptions`. It is absent for a camera that the current input API cannot select. |
| `model` | Device model when the provider reports one. |
| `backend` | Discovery backend, such as `mipi` or `v4l2`. |
| `modes` | Discrete sizes or explicit size ranges, frame rate, support flag, and rejection reason. |

Each mode's `framerate_num`/`framerate_den` is the fastest rate among the
frame intervals Sentinel lists for the mode, or `0/1` when it lists none. The
DevKit's ISP lists none for MIPI modes; `CameraInput` sets the rate through
caps.

Core skips an interval entry whose `type` it does not know, or a stepwise or
continuous range whose maximum is shorter than its minimum. A skipped entry
neither sets the rate nor covers the default rate, but the mode still counts as
listing intervals.

Core classifies each mode for the `CameraInput` default libcamera profile
(`profile=Default`, which selects cameras by `camera_name`). A mode is
supported when all of these hold, checked in this order; `reason` gives the
first one that fails:

1. The camera's `backend` is `mipi`.
2. The format is `CameraInputOptions`' default format (`NV12`).
3. If the mode lists frame intervals, one of them covers the default frame rate
   (`30/1`): a discrete interval of 1/30 s, or a stepwise or continuous range
   that contains it. A mode that lists no intervals is not rejected on rate and
   has rate `0/1`. A mode whose listed intervals are all skipped is rejected on
   rate and also has rate `0/1`.
4. The mode is an ISP output size (`isp_output` is true).

Core does not probe, acquire, configure, or stream from the camera. Catalog
support therefore does not guarantee that exclusive acquisition will succeed
later.

These rules do not classify the Metoak SIMOR raw V4L2 profile
(`CameraProfile::MetoakSimor`, RAW8 1920×360, selected with `profile` and
optionally `device`). They have no per-sensor condition, so a SIMOR sensor
(name starting with `simor_metoak`) that Sentinel lists as a `mipi` camera gets
the same ISP-mode classification as any other MIPI sensor. For that camera,
`supported: true` means only that the mode matches the default profile's
backend, format, frame rate, and ISP output size; it does not mean that
libcamera is qualified for the sensor. Select a SIMOR camera with
`profile=MetoakSimor` as described in
[`CameraInput`](/reference/nodes/camera-input), not with its catalog
`camera_name`.

## Details for any peripheral type

Every peripheral, whatever its `type`, carries the whole device record that
Sentinel published in `details_json`, as compact JSON. Python also provides
`details`, which decodes `details_json` into a new `dict`.

`details_json` is the way to read a peripheral type for which Core has no
typed accessor, such as `microphone`. A new device type is usable as soon as
Sentinel reports it, without a Core update. Every field and value is
preserved, including fields this Core release does not know; the JSON is
re-serialized, so key order and whitespace may differ from the daemon
response. Sentinel documents the fields of each type.

Python:

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details["capture_target"])
```

C++:

```cpp
#include <nlohmann/json.hpp>

for (const auto& peripheral : simaai::neat::peripherals::list()) {
  if (peripheral.type == "microphone") {
    const auto details = nlohmann::json::parse(peripheral.details_json);
    // Read details["capture_target"] and other fields.
  }
}
```

Any JSON library can parse `details_json`; the example uses nlohmann/json.

Core validates only the fields it reads. Invalid catalog fields (`revision`,
`observed_at`, `errors`, `devices`) or a device without a valid `id` and `type`
make `list()` fail with a parse error. A camera record whose camera fields Core
cannot read, for example one from a newer Sentinel, does not: that device keeps
its `id`, `type` and `details_json`, its `camera` is left unset, and every
other device is returned as usual.

## Failures and scope

`list()` raises `NeatError` with a stable code when the service is missing,
too old to serve the catalog, or has peripheral discovery disabled or stopped,
when permission is denied, the request times out, or the response is
malformed or oversized. The message includes the next operational action and,
when Sentinel gives one, Sentinel's own error text. See the
[error code catalog](./error-codes.md).

If Sentinel is not installed, or is too old to serve the peripheral catalog,
install or update it with `sima-cli neat install sentinel`. If it is installed
but not running, start `simaai-sentinel.service`.

This API connects only to `/run/simaai-sentinel/api.sock` on the local
DevKit. It does not use SSH or select a remote board. Insight and future CLI
clients connect to the daemon as sibling clients rather than through Core.

This release provides a one-shot catalog read. Refresh requests and daemon
lifecycle control are not part of the public Core API.
