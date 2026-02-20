# Tensor and Sample Contracts

`Tensor` essentials:

- Carries dtype/layout/shape/device/storage metadata.
- Prefer explicit shape/layout assumptions in code.
- Validate expected dimensions before inference/postprocess.

`Sample` essentials:

- `SampleKind::Tensor` carries single tensor in `sample.tensor`.
- `SampleKind::Bundle` carries multiple fields in `sample.fields`.
- Metadata fields (`stream_id`, `frame_id`, `pts_ns`, `port_name`) are used for routing/joins/diagnostics.

Input rules:

- Ensure `InputAppSrcOptions` (caps and optional max bounds) match produced tensor/image format.
- For dynamic input modes, set reasonable max bounds to avoid pool/caps churn.

Output handling:

- Check `sample.kind` before reading `sample.tensor`/`sample.fields`.
- For bundle outputs, iterate `fields` and select by `port_name` or `payload_tag`.

Defensive checks to generate:

- Null/empty tensor checks before processing.
- Dtype/layout assertions before reinterpretation.
- Timeout and closed-state handling for `pull(...)`.
