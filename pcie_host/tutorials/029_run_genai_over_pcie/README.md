# 029 Run GenAI Models over PCIe

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | Qwen3-0.6B-Autoround-a16w4, LFM2-VL-450M-Autoround-a16w4, whisper-small-a16w8 |
| Labels | PCIe, GenAI, LLM, VLM, Whisper |

## Concept

Run text generation, image questions, and speech transcription from a host application on a Modalix PCIe Card. Model files, requests, and results travel over PCIe; the application uses the public GenAI API, not a GStreamer pipeline.

## Walkthrough

Each example loads one prepared LLiMa model directory on the host, runs a bounded request, and closes the model. Run the examples separately first. They use the same connection settings but different request inputs.

### Load an LLM {#step-load-llm}

Construct `pcie::genai::GenAIModel` in C++ or `pyneatpcie.genai.GenAIModel` in Python. Construction loads the model on the card; there is no `build()` or queue selection. Set `connection.card_id` to select the card. Absolute model paths and paths relative to the current working directory are accepted.

### Send a text request {#step-text-request}

Set `prompt` and `max_new_tokens`, then call `run()`. The result contains answer text and generation metrics. The examples default to 128 generated tokens; use `--max-tokens` to change that budget.

### Stream the answer {#step-stream-answer}

Pass `--stream` to the LLM example to iterate `TokenSample` values instead. Print each text fragment as it arrives; the final sample carries the completed generation metrics. Only one request may be active per model.

### Close the model {#step-close-model}

C++ calls `close()` explicitly after inference and uses destructor cleanup if an exception interrupts the example. Python uses a context manager, which closes the model when the block exits, including on errors. Both connection timeouts are finite: 15 minutes for startup and 5 minutes for a request.

### Prepare an RGB image {#step-prepare-image}

The VLM example reads the image on the host with OpenCV, converts BGR to RGB, and produces contiguous UInt8 pixels with shape `[H, W, 3]`. C++ wraps an owning pixel vector in a PCIe tensor; Python supplies a NumPy array. Do not normalize the pixels or substitute a card-local image path.

### Ask about the image {#step-image-request}

Load a VLM and attach the RGB input to `request.images` alongside a prompt. The model handles its own image preprocessing. The returned text answers the question; there is no image-file field on the PCIe request.

### Transcribe speech with Whisper {#step-audio-request}

Load the Whisper directory and set `audio_file` to a host audio path. Use `language="auto"` for language detection or specify a source language with `--language`. By default the request transcribes; `--translate` selects translation into English. Read the transcript, language, and optional no-speech probability. Do not add a chat prompt to an ASR request.

## Run

Follow [Install PCIe Host](/getting-started/neat-library/pcie-host/) and [Tutorial Setup](/tutorials/before-you-run). Use matching host and card packages that expose the PCIe GenAI API. Run all commands below on the host, from the extracted PCIe extras root.

### Download prepared models

These SiMa.ai Model Zoo repositories already contain compiled LLiMa assets: [Qwen3](https://huggingface.co/simaai/Qwen3-0.6B-Autoround-a16w4), [LFM2-VL](https://huggingface.co/simaai/LFM2-VL-450M-Autoround-a16w4), and [Whisper](https://huggingface.co/simaai/whisper-small-a16w8). Download only the models you plan to run. Review their model cards and use assets compatible with your installed platform.

```bash
source ~/pyneatpcie/bin/activate
python -m pip install huggingface_hub opencv-python
mkdir -p models
hf download simaai/Qwen3-0.6B-Autoround-a16w4 \
  --local-dir models/Qwen3-0.6B-Autoround-a16w4
hf download simaai/LFM2-VL-450M-Autoround-a16w4 \
  --local-dir models/LFM2-VL-450M-Autoround-a16w4
hf download simaai/whisper-small-a16w8 \
  --local-dir models/whisper-small-a16w8
```

Keep each complete directory, including its tokenizer, configuration, and compiled assets. No unpacking or manual copy to the card is required. A raw Hugging Face checkpoint is not a prepared LLiMa directory; for custom models, follow [GenAI with LLiMa](/genai-llima/) before running this tutorial.

### Prepare the media

The VLM commands use the bundled street-scene image. For Whisper, record or provide a short speech clip on the host. With FFmpeg installed, convert an existing recording into a 16 kHz mono WAV:

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le speech.wav
```

Do not use silence to judge transcription quality. The examples accept another image or audio path through `--image` or `--audio`.

**Python:**

```bash
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_llm.py \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_vlm.py \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_whisper.py \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./lib/sima-pcie-host/tutorials/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./lib/sima-pcie-host/tutorials/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_029_run_llm
./build.sh --target tutorial_029_run_vlm
./build.sh --target tutorial_029_run_whisper
./build/tutorials-standalone/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./build/tutorials-standalone/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./build/tutorials-standalone/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

The C++ source build needs the PCIe development package and OpenCV development files. All examples default to card 0 and the connection's default SSH user, sima. Add `--card N`, `--user USER`, `--ssh-key PATH`, or `--card-host ADDRESS` when your installation differs. Use a provisioned SSH account; the examples do not configure access.

Expect an LLM answer, a description of the scene, or a transcript matching the speech recording. Wording depends on the model and input; successful execution alone is not a quality check.

## In Practice

Use separate handles for independent models, subject to card memory and compute resources. For chat, supply history through `messages` instead of `prompt`; independent calls do not remember previous turns. For stream cancellation, call `cancel()` and consume the terminal sample before reusing the model.

See [PCIe GenAI APIs](/develop-apps/development-workflow/genai-model/pcie-api) for the API overview. The standalone GenAI tutorials use a different namespace and local model execution; do not substitute their model handles here.

## Source Files

- `run_llm.cpp` / `run_llm.py`
- `run_vlm.cpp` / `run_vlm.py`
- `run_whisper.cpp` / `run_whisper.py`
- `tutorial_args.h` (shared C++ command-line parsing)
- `../assets/street-scene.png`
