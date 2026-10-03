# Peripheral protocol fixture

`catalog.json` is a v1 peripheral catalog document in the shape SiMa Sentinel
(`sima-neat/sentinel`) serves from `GET /v1/peripherals` on
`/run/simaai-sentinel/api.sock`. It is hand-written, not captured from a
board:

- `devices[0]` is a MIPI IMX477 camera from the `daemon.camera.mipi`
  provider. Its record matches Sentinel's MIPI provider test
  `real_imx477_devkit_board_produces_one_camera_with_isp_modes`, whose media
  graph and ISP output sizes are transcribed from a DevKit capture.
- `devices[1]` is a USB camera from the `daemon.camera.v4l2` provider.
  Sentinel's V4L2 provider tests keep a copy of this record, without
  `supported` and `reason`, and check their output against it.
- Each mode's `supported` and `reason` are what Sentinel's support stage
  produces with the rules this Core installs
  (`src/peripherals/sentinel-support-rules.json.in`).
- The top-level `changes` log and `support` status are fields Sentinel
  publishes with every snapshot. Core accepts both and exposes neither.

Core tests parse this data through the public client without linking or
invoking Sentinel. Update the fixture only with a compatible v1 Sentinel
document or alongside a new protocol version.
