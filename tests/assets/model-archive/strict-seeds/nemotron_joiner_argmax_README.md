# Nemotron joiner manifest fixture

Compiler-generated MPK manifest from the Nemotron 3.5 ASR 0.6B INT8-weight,
BF16-activation joiner with MLA ArgMax. Only `model_path` is normalized to a
portable filename; stage order, names, sizes, typed ports and resources are
unchanged. No weights or compiled executable is included or needed for metadata
regressions. Tests generate a non-executable placeholder ELF header for archive inventory
and structural executable evidence separately.

Original manifest SHA256: `a413e124afa9608645f9ce14f19482ff28d7695a42f6a1d478cc06060bccf7f8`.

This fixture covers the scalar crop → INT32 argument-only host identity →
`[1,1]` reshape tail. The runtime's published tensor and cold `output_specs()`
metadata must agree. A real INT32-to-INT64 conversion remains a distinct host job.
