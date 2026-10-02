# Peripheral protocol fixture

`catalog.json` is a byte-for-byte snapshot of the canonical v1 catalog fixture
owned by `sima-neat/internals` under `peripherals/protocol/v1/catalog.json`.
Core tests parse this data through the public client without linking or invoking
the daemon implementation. Update the snapshot only with a compatible v1
daemon fixture or alongside a new protocol version.
