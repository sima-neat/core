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
installed `CameraInput`. The client preserves that result and does not
probe, reclassify, acquire, configure, or stream from the camera. Catalog
support therefore does not guarantee that exclusive acquisition will succeed
later.

## Details for any peripheral type

Every peripheral, whatever its `type`, carries its type-specific details in
`details_json`: the compact JSON object that Sentinel publishes under the
record key named by `type` (for example `camera`, `microphone`, or `lidar`).
When that key is absent or `null`, `details_json` is `"{}"`. Python also
provides `details`, which decodes `details_json` into a new `dict`.

`details_json` is the authoritative way to read a peripheral type for which
Core has no typed accessor. A new device type is usable as soon as Sentinel
reports it, without a Core update. Every field and value is preserved,
including fields this Core release does not know; the JSON is re-serialized, so
key order and whitespace may differ from the daemon response. Cameras carry
both `details_json` and the typed `camera` field.

Python:

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details.get("channels"))
```

C++:

```cpp
#include <nlohmann/json.hpp>

for (const auto& peripheral : simaai::neat::peripherals::list()) {
  if (peripheral.type == "microphone") {
    const auto details = nlohmann::json::parse(peripheral.details_json);
    // Read details.value("channels", 0) and other provider fields.
  }
}
```

Any JSON library can parse `details_json`; the example uses nlohmann/json.

For a type other than `camera`, a details value that is not a JSON object does
not make `list()` fail: that peripheral stays in the catalog with
`details_json` set to `"{}"`. Core does not otherwise validate the details of
types it has no typed accessor for. Invalid `camera` details are a protocol
defect and make `list()` fail with a parse error. Typed fields such as `camera`
ignore optional protocol v1 fields they do not know; those fields remain
available in `details_json`. Core also accepts the top-level `changes` log and
`support` status that Sentinel publishes with each snapshot, but does not
expose them.

## Failures and scope

`list()` raises `NeatError` with a stable code when the service is missing or
too old to serve the catalog, permission is denied, the request times out, the
daemon is not ready, or the response is malformed, oversized, or incompatible. The message includes the
next operational action. See the [error code catalog](./error-codes.md).

If Sentinel is not installed, or is too old to serve the peripheral catalog,
install or update it with `sima-cli neat install sentinel`. If it is installed
but not running, start `simaai-sentinel.service`.

This API connects only to `/run/simaai-sentinel/api.sock` on the local
DevKit. It does not use SSH or select a remote board. Insight and future CLI
clients connect to the daemon as sibling clients rather than through Core.

This release provides a one-shot catalog read. Event subscriptions, refresh
requests, and daemon lifecycle control are not part of the public Core API.
