# 021 Serve GenAI Models

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4, Qwen3-VL-4B-Instruct-GPTQ-a16w4, whisper-small-a16w8 |
| Labels | genai, server, llm, vlm, asr, http |

## Concept

Host multiple GenAI models behind the Neat GenAI server so a UI, service, or remote client can call LLM, VLM, and ASR endpoints from one process.

## Walkthrough

Direct `model.run(request)` is the best starting point for embedded application logic. Use `GenAIServer` when the application boundary is HTTP: a browser UI, a companion service, or a remote client that should not link against the Neat runtime.

### Configure the server {#step-configure-server}

Choose the host and port. The default host is `0.0.0.0`, which accepts connections from other machines that can reach the Modalix device.

### Register model directories {#step-register-models}

Add each deployed model directory with a served name. This tutorial registers `llm`, `vlm`, and `asr`; the served name is what clients send in the `model` field.

### Start serving {#step-start-serving}

Call `serve()` for a blocking foreground process or `start()` when your application owns the rest of the process lifetime.

After the server starts, verify the registered model names with `GET /v1/models`:

```bash
curl http://<modalix-ip>:9998/v1/models
```

The response should include the served names registered in this tutorial: `llm`, `vlm`, and `asr`.

## Run


On the Modalix DevKit, download the LLM, VLM, and ASR models from Hugging Face using the LLiMa CLI:

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
llima pull whisper-small-a16w8
```

Start the server on Modalix with all three DevKit-local model directories:

**Python:**
```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/serve_genai_models.py \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_021_serve_genai_models
./build/tutorials-standalone/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

Drop `--vlm` or `--asr` if you only want to serve a subset during development.

After the server is running, first verify that all served names are registered:

```bash
curl http://<modalix-ip>:9998/v1/models
```

Then call the endpoints from a client. Replace `<modalix-ip>` with the IP address or hostname of your Modalix device.
The request clients below use Python `requests`, stream the response, and print server-side TTFT plus average, minimum, and maximum per-token TPS when reported.

### Text request to the LLM

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_text.py \
  --server-ip <modalix-ip> \
  --model llm \
  "Give me three tips for designing a small REST API."
```

### Text and image request to the VLM

The request script base64-encodes the image and sends it as an OpenAI-compatible `image_url` content part.

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_image.py \
  --server-ip <modalix-ip> \
  --model vlm \
  image.jpg \
  "What is the main subject of this image?"
```

### Audio request to the ASR model

The transcription client defaults to automatic source-language detection. Use
`--language` when the source language is known:

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  speech.wav
```

To translate speech into English, add `--translate`. The client sends the same
multipart request to `POST /v1/audio/translations`:

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  --translate \
  speech-in-another-language.wav
```

Transcription uses `POST /v1/audio/transcriptions`. Both routes support
`stream=true`; the supplied client streams text and prints the detected source
language, `no_speech_prob`, and `avg_logprob` from the final event. A higher
`no_speech_prob` indicates that Whisper considers the input more likely to
contain no speech. `avg_logprob` is the mean log probability of generated
tokens, where a higher (less negative) value indicates a more confident decode.

## In Practice

Use the server when a network boundary is useful. Use direct `GenAIModel`, `VisionLanguageModel`, and `ASRModel` calls for lower-overhead application code inside the same process.

Run one `GenAIServer` process with multiple served model names for normal applications. Multiple server processes can bind different ports if the DevKit has enough memory, but they load their own model instances and still share the same MLA hardware gatekeeper, so they should not be treated as a way to multiply hardware throughput.

The `/v1/models` endpoint is the quickest smoke check: if it returns the served names, the server is reachable and the model registry is populated.

## Source Files
- C++: `tutorials/021_serve_genai_models/serve_genai_models.cpp`
- Python: `tutorials/021_serve_genai_models/serve_genai_models.py`
- Request clients:
  - `tutorials/021_serve_genai_models/request_chat_completion_text.py`
  - `tutorials/021_serve_genai_models/request_chat_completion_image.py`
  - `tutorials/021_serve_genai_models/request_audio_transcription.py`
