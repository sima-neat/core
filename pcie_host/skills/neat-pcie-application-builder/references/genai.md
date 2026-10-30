# PCIe GenAI APIs

Use `simaai::neat::pcie::genai::GenAIModel` in C++ or `pyneatpcie.genai.GenAIModel`
in Python for LLM, VLM, and ASR applications. These APIs consume prepared LLiMa
model directories on the host, not MPK archives or card-local paths.

## Model And Connection

- Pass an absolute host directory or a path relative to the application's working directory.
  Keep its files readable and unchanged while the model is open.
- Use GenAI's own `ConnectionOptions`, not `pcie::ConnectionOptions`.
  Set `card_id`; an empty `card_host` uses `10.0.<card_id>.2`. Override the address when needed.
- Construction loads the remote model. There is no `build()`, `info()`, `pcie::Model` queue,
  `max_inflight`, or `push()`/`pull()` workflow.
- Configure `startup_timeout_ms` and `request_timeout_ms` on the connection.
  `run()` does not take a per-call timeout argument.
- Use `task()` and `accepts_text()`, `accepts_image()`, `accepts_audio()` to check capabilities.
- Close the model on normal and error paths; prefer the Python context manager.

```python
from pyneatpcie import genai

connection = genai.ConnectionOptions()
connection.card_id = 0
with genai.GenAIModel("./models/my-llm", connection) as model:
    request = genai.GenerationRequest()
    request.prompt = "Explain PCIe in one sentence."
    request.max_new_tokens = 64
    print(model.run(request).text)
```

## Requests And Results

- Use `prompt` for a simple request, optionally with `system_prompt`.
  For history, use `messages` with explicit roles and content instead; do not combine
  `messages` with `prompt` or top-level `system_prompt`.
- Requests do not inherit conversation history. Include the history needed for each turn.
- VLM images are UInt8 HWC RGB PCIe tensors, shape `[H, W, 3]`; Python also accepts NumPy arrays.
  Load image files in the application and convert BGR to RGB when using OpenCV.
  Attach images to `request.images` with a prompt, or to each `ChatMessage.images` with history.
  There is no `image_files` field or Core `ImageList` on the PCIe API.
- For ASR, supply exactly one of `audio_file` (host path) or `audio` (Float32 mono `[N]` PCIe
  tensor; Python also accepts NumPy arrays). For in-memory audio, set `sample_rate`, normally
  16000. Use `language` and `ASRTask.Transcribe` or `ASRTask.Translate` as appropriate.
  Do not add text/chat/image fields to ASR requests.
- Use `tools` and `tool_choice` only with compatible tool-calling models. Read parsed
  `tool_calls`; the application executes tools and supplies their results in later history.
  Inspect supported `tool_choice` values in the matching release rather than inventing options.
- `enable_thinking` is model-dependent. Keep returned `reasoning` separate from answer `text`.
- `run()` returns a complete `GenerationResult`. For ASR, also inspect `language`,
  `no_speech_prob`, and `avg_logprob` when relevant.

## Streaming And Multiple Models

Use `stream(request)` for incremental `TokenSample` output. Iterate in C++ or Python, or use
`next()` until no sample remains. Preserve text, reasoning, and tool calls across samples;
the terminal sample carries final metrics and the finish reason.

Only one request may be active per model. After `cancel()`, consume the stream through its
terminal sample and end before reusing the model. A timeout is not evidence that remote
generation stopped; close a failed model rather than blindly submitting another request.

Use separate handles for independent models. GenAI handles do not reserve `pcie::Model` queue numbers
and can coexist with `pcie::Model` objects, subject to card resources. Do not promise a fixed
concurrent-model count or throughput without measuring the selected workload.

For speculative decoding in supporting releases, pass the parent directory containing a
prepared target/draft pair. Compiled configuration identifies the roles; both models load in
one card worker. No extra request flag is needed, and both models must fit in card memory.

## Boundaries And Validation

Use the PCIe `GenerationRequest`, `ChatMessage`, and tensor types, not their local Core
counterparts. The shared request pattern does not imply identical overloads or connection fields.
Do not substitute `pyneat.genai`, local `VisionLanguageModel`/`ASRModel`, `GenAIServer`,
GenAI graph fragments, or private transport APIs.

Verify the installed header and bindings before generating code. With authorized card access,
run one bounded representative request, check useful output and cleanup, then test streaming or
media if the application uses them. Compile/import checks alone do not validate inference.

Use tutorial `029_run_genai_over_pcie` from the matching PCIe extras bundle for runnable
C++ and Python LLM, VLM, and Whisper examples and model download instructions.
