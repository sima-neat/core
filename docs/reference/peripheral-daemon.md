---
title: Peripheral Catalog Daemon
---

# Peripheral catalog daemon

`simaai-peripherals` maintains one current peripheral catalog for all local
clients on a DevKit. The full Core distribution includes its separate
`sima-neat-peripherals` component package, managed by
`simaai-peripherals.service`.

## Architecture

```text
Linux/udev notification or explicit refresh
                  |
                  v
       simaai-peripherals daemon
       - debounce refresh triggers
                  |
                  v
       private provider registry
       - isolate provider failures
       - retain last-good provider results
       - normalize and classify capabilities
                  |
                  v
       daemon catalog and event replay
                  |
                  v
       Unix socket clients
```

The daemon owns monitoring, providers, refresh scheduling, the current snapshot,
scan and catalog sequences, and replayable events. A udev notification only
tells the daemon to refresh; provider code performs the capability query and
support classification without acquiring or streaming from the camera.

The initial private provider discovers MIPI/libcamera cameras. No connected
camera produces a ready empty catalog. Providers run independently: a failing
provider contributes a structured issue and retains only its own last-good
records, while current results from healthy providers remain visible. This
allows future microphone, LiDAR, and other providers to be added without one
backend failure erasing unrelated devices.

## Local API

The service exposes HTTP/JSON only through this Unix socket:

```text
/run/simaai-peripherals/api.sock
```

It does not listen on TCP. The systemd unit runs as the unprivileged `sima`
user, creates the runtime directory for the `sima` group, and sets the socket
mode to `0660`.

| Method and path | Behavior |
| --- | --- |
| `GET /v1/health` | Returns readiness, freshness, instance ID, revision, event sequence, scan sequence, and the latest error. |
| `GET /v1/catalog` | Returns one internally consistent current snapshot. |
| `GET /v1/events?after_sequence=N&wait_ms=M&instance_id=ID` | Returns events after a client cursor; `wait_ms` is limited to 30000. |
| `POST /v1/refresh` | Schedules a daemon-owned discovery refresh and returns `202` with a `target_scan_sequence` completion token. |

Requests are limited to 8 KiB, response bodies to 4 MiB, and concurrent
connections to 64. An oversized catalog fails with `response_too_large`
instead of sending an unbounded response.

### V1 response schema

Health and catalog responses share these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer catalog schema version; currently `1`. |
| `instance_id` | UUID created for this daemon process. |
| `state` | `starting`, `ready`, or `degraded`. |
| `ready` | Whether at least one provider has completed successfully, including a valid empty result. |
| `stale` | Whether devices come from the last successful scan after a newer failure. |
| `revision` | Device-snapshot revision; errors and recovery alone do not advance it. |
| `sequence` | Latest replayable event sequence. |
| `scan_sequence` | Number of completed discovery attempts; success and failure both advance it. |
| `last_success_at` | UTC timestamp of the latest successful scan, or `null`. |
| `last_attempt_at` | UTC timestamp of the latest attempted scan, or `null`. |
| `error` | `null`, or an object containing `code` and actionable `reason`. |
| `issues` | Provider-scoped failures, including whether that provider retained last-good records. |

`GET /v1/health` also returns `api_version: "v1"` and `device_count`.
`GET /v1/catalog` instead returns `devices`, whose type-specific capabilities
are nested under their device type:

```json
{
  "schema_version": 1,
  "instance_id": "7ff4c8a1-...",
  "state": "ready",
  "ready": true,
  "stale": false,
  "revision": 1,
  "sequence": 0,
  "scan_sequence": 1,
  "last_success_at": "2026-10-01T01:23:45.678Z",
  "last_attempt_at": "2026-10-01T01:23:45.678Z",
  "error": null,
  "issues": [],
  "devices": [
    {
      "id": "camera:imx477 5-001a",
      "type": "camera",
      "provider": "daemon.camera.libcamera",
      "camera": {
        "camera_name": "imx477 5-001a",
        "model": "imx477",
        "backend": "libcamera",
        "modes": [
          {
            "format": "NV12",
            "width": 1920,
            "height": 1080,
            "framerate_num": 30,
            "framerate_den": 1,
            "supported": true,
            "reason": ""
          }
        ]
      }
    }
  ]
}
```

`model` is omitted when the provider does not report it. A range mode contains
`size_range` with `min_width`, `min_height`, `max_width`, `max_height`,
`step_width`, and `step_height` instead of discrete `width` and `height`.

An events response contains `schema_version`, `instance_id`, `revision`, the
latest `sequence`, `scan_sequence`, `resync_required`, `shutting_down`, and an
`events` array.
Each event contains `sequence`, `revision`, and `kind`. Device events add
`device_id`, `device_type`, and `previous` and/or `current`. An `error` event
contains the structured `error`; a `recovered` event has no device and means a
successful scan cleared degraded state without necessarily changing devices.

For example:

```bash
curl --unix-socket /run/simaai-peripherals/api.sock \
  http://localhost/v1/catalog
```

An accepted explicit refresh returns a token such as:

```json
{"accepted":true,"target_scan_sequence":2}
```

Poll health or catalog until `scan_sequence` is at least that token. Requests
accepted before the same pending scan begins may share a token. A request that
arrives while discovery is running targets the following scan, so observing the
token always means a scan accepted after that request has completed.

### Catalog identity

Every daemon start creates a new `instance_id`. A successful initial scan sets
catalog revision `1` without generating synthetic `added` events. A later scan
that changes one or more devices advances the revision once; all events from
that scan share the new revision.

Each completed discovery attempt advances `scan_sequence`, including a failed
attempt that preserves the last good snapshot. Each event has its own
monotonically increasing `sequence`. Event kinds are
`added`, `removed`, `changed`, `error`, and `recovered`. Camera identities use
the provider's exact libcamera camera name rather than an unstable `/dev/videoN`
index. Camera details preserve its model, modes or explicit size ranges, frame
rate, support classification, and rejection reason.

### Client resynchronization

Clients should retain both `instance_id` and `sequence`:

1. Read `/v1/catalog` and store its `instance_id` and `sequence`.
2. Long-poll `/v1/events` with those values.
3. Apply returned events in sequence order.
4. If `resync_required` is true, discard local state and read `/v1/catalog`
   again.

Resynchronization is required when the daemon restarted, the cursor is newer
than the daemon, or the cursor fell behind the bounded replay buffer. Slow or
disconnected clients do not consume a shared queue and do not block monitoring
or other clients.

## Monitoring and refresh

The daemon builds its udev filter from the subsystems declared by registered
providers. The camera provider currently watches `media` and `video4linux`.
It combines a burst of notifications using a 250 ms debounce period and then
runs each private provider once. `POST /v1/refresh` and `systemctl reload`
cover software or configuration changes that do not emit a device event.

Do not physically connect or disconnect a MIPI ribbon camera while the board
is powered. Validate add/remove behavior with peripherals that are safe to
hot-plug.

## Operations

```bash
sudo systemctl status simaai-peripherals
sudo systemctl reload simaai-peripherals
sudo systemctl restart simaai-peripherals
journalctl -u simaai-peripherals
```

The service starts automatically when `sima-neat-peripherals` is installed and
restarts after an unexpected failure. On graceful shutdown it wakes long-poll
clients and removes the socket before waiting for in-progress discovery. Client
I/O has absolute deadlines, and systemd enforces a 10-second final stop timeout
if an external provider does not return.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Socket is missing | Check `systemctl status simaai-peripherals` and the service journal. |
| Socket access is denied | Confirm the client belongs to the `sima` group and reconnect its session. |
| State is `degraded` | Read `error` and provider-scoped `issues` in `/v1/health`; any retained provider records are marked by `retained_last_good`. |
| `resync_required` is true | Read a fresh catalog, replace local state, and continue from its sequence. |
| A configuration change was not detected | Call `POST /v1/refresh` or reload the service. |
