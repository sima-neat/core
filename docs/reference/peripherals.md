---
title: Peripheral catalog
description: List the DevKit peripherals reported by the board-local catalog service
sidebar_position: 8
---

# Peripheral catalog

Use the peripheral catalog to inspect devices currently attached to the local
DevKit. The API is available in C++ and Python and returns the same typed
snapshot in both languages.

The authoritative catalog belongs to SiMa Sentinel (`simaai-sentinel.service`),
the board-local daemon that serves it as `GET /v1/peripherals`. Each `list()`
call performs one bounded request to Sentinel. Core does not scan hardware,
cache a second catalog, or fall back to another discovery path.

## List peripherals

Python:

```python
import pyneat

catalog = pyneat.peripherals.list()
for peripheral in catalog:
    print(peripheral.id, peripheral.type, peripheral.provider)
```

C++:

```cpp
#include <neat.h>

auto catalog = simaai::neat::peripherals::list();
for (const auto& peripheral : catalog) {
  // Use peripheral.id, peripheral.type, and peripheral.provider.
}
```

The returned catalog is iterable. It also contains the daemon `instance_id`,
catalog `revision`, event `sequence`, completed `scan_sequence`, freshness
state and timestamps, current structured error, provider issues, and the
`devices` collection.

A ready catalog may contain zero devices. That is a successful result. A
degraded catalog may contain the last successful device list; in that case
`stale` is `true`, and `error` and/or `issues` explain the refresh failure.

## Camera details

When `peripheral.type == "camera"`, `peripheral.camera` contains:

| Field | Meaning |
| --- | --- |
| `camera_name` | Optional exact libcamera name accepted by `CameraInputOptions`. It is absent for a camera that the current input API cannot select. |
| `model` | Device model when the provider reports one. |
| `backend` | Discovery backend, such as `mipi` or `v4l2`. |
| `modes` | Discrete sizes or explicit size ranges, frame rate, support flag, and rejection reason. |

Sentinel classifies support by applying the rules this Core package installs at
`/usr/share/simaai-sentinel/support/neat-core.json`, so the result matches the
installed `CameraInput`. Catalog support does not guarantee that exclusive
acquisition will succeed later.

## Details for any peripheral type

Every peripheral, whatever its `type`, carries its type-specific details in
`details_json`: the compact JSON object that Sentinel publishes under the
record key named by `type` (for example `camera`, `microphone`, or `lidar`).
When that key is absent or `null`, `details_json` is `"{}"`. Python also
provides `details`, which decodes `details_json` into a new `dict`.

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details.get("channels"))
```

In C++, parse `details_json` with any JSON library. A new device type is
usable as soon as Sentinel reports it, without a Core update. For a type other
than `camera`, a details value that is not a JSON object does not make `list()`
fail: that peripheral stays in the catalog with `details_json` set to `"{}"`.
Invalid `camera` details make `list()` fail with a parse error.

## Failures

`list()` raises `NeatError` with a stable code when the service is missing or
too old to serve the catalog, permission is denied, the request times out, the
daemon is not ready, or the response is malformed, oversized, or incompatible. The message includes the
next operational action. See the [error code catalog](./error-codes.md).

If Sentinel is not installed, or is too old to serve the peripheral catalog,
install or update it with `sima-cli neat install sentinel`. If it is installed
but not running, start `simaai-sentinel.service`.
