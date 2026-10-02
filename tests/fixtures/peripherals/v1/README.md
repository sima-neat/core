# Peripheral protocol fixture

`catalog.json` is a v1 peripheral catalog document in the shape SiMa Sentinel
(`sima-neat/sentinel`) serves from `GET /v1/peripherals` on
`/run/simaai-sentinel/api.sock`. It includes the optional top-level `changes`
log, which Core accepts and ignores. Core tests parse this data through the
public client without linking or invoking Sentinel. Update the fixture only
with a compatible v1 Sentinel document or alongside a new protocol version.
