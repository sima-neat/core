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
| `backend` | Discovery backend, such as `libcamera` or `v4l2`. |
| `modes` | Discrete sizes or explicit size ranges, frame rate, support flag, and rejection reason. |

The daemon classifies support. The client preserves that result and does not
probe, reclassify, acquire, configure, or stream from the camera. Catalog
support therefore does not guarantee that exclusive acquisition will succeed
later.

Unknown peripheral types remain in the catalog with their common `id`, `type`,
and `provider`. Optional fields added to protocol v1 are ignored by older
clients. This includes the top-level `changes` log that Sentinel publishes with
each snapshot; Core accepts it but does not expose it.

## Failures and scope

`list()` raises `NeatError` with a stable code when the service is missing,
permission is denied, the request times out, the daemon is not ready, or the
response is malformed, oversized, or incompatible. The message includes the
next operational action. See the [error code catalog](./error-codes.md).

If Sentinel is not installed, install it with `sima-cli neat install sentinel`.
If it is installed but not running, start `simaai-sentinel.service`.

This API connects only to `/run/simaai-sentinel/api.sock` on the local
DevKit. It does not use SSH or select a remote board. Insight and future CLI
clients connect to the daemon as sibling clients rather than through Core.

This release provides a one-shot catalog read. Event subscriptions, refresh
requests, and daemon lifecycle control are not part of the public Core API.
