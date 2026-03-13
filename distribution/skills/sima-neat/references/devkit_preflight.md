# DevKit Preflight

Use this before coding or testing from eLxr SDK.

## Required checks

1. Check the runner:

```bash
bash -lic 'type dk || type devkit-run'
```

2. Confirm the build will target ARM64.
3. Confirm required assets exist before coding around them:
   - model pack
   - test image or video
4. If running from SDK, ensure the binary or script will be under `/workspace`.

## Run order

1. Build the ARM64 target.
2. Run with `dk /workspace/...` if available.
3. Fall back to direct SSH only if `dk` is unavailable.

## Recovery

- `dk` missing:
  - check `bash -lic 'type dk || type devkit-run'`
  - if unavailable, use direct SSH only when the user still wants an on-device run
- target not under `/workspace`:
  - rebuild or copy it into the shared workspace path first
- non-ARM64 binary:
  - fix the toolchain/build configuration before attempting DevKit execution
- missing model/image:
  - stop and ask for the asset path, or use the user-provided default if one was specified
