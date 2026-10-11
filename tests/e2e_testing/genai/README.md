# Gemma4 MTP regression

Build Core against the Gemma4 MTP-capable LLiMa development package and runtime.
The manual `genai_mtp_run_test` accepts the parent model package containing both
the target and assistant directories, a free local HTTP port, and an optional
image file:

```bash
timeout 240s ./build/tests/genai_mtp_run_test \
  /media/nvme/llima/models/Gemma-4-E2B-it-TextOnly-MTP-4k-LLiMa 19099

timeout 240s ./build/tests/genai_mtp_run_test \
  /media/nvme/llima/models/Gemma-4-E2B-it-MTP-4k-480-LLiMa 19099 \
  tests/images/people.jpg
```

It checks matching direct generation and streaming, a one-token output limit,
HTTP chat generation, and matching HTTP streaming with a final `[DONE]` event.
With an image, direct generation also exercises vision encoding; HTTP requests
exercise text generation on the same VLM target.

The executable is built with the GenAI tests but remains outside automatic CTest
registration until a compiled MTP pair is available in the CI fixture manifest.
Missing models and runtime failures fail the test.
