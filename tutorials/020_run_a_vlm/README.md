# 020 Run a VLM

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Model | Qwen3-VL-4B-Instruct-GPTQ-a16w4 |
| Labels | genai, vlm, image, cache, multimodal |

## Concept

Ask repeated questions about the same image without re-encoding the image for every request.

## Walkthrough

Vision-language models can accept text plus image tensors. For one question, attach the image directly to `GenerationRequest.images`. For repeated questions, encode the image once and reuse the cached image embeddings in follow-up requests.

### Load the VLM and image {#step-load-inputs}

Load a `VisionLanguageModel` from a deployed LLiMa model directory and decode an image from disk.

**C++:** Use OpenCV to read the image. Neat treats three-channel `cv::Mat` inputs as BGR and converts them to RGB internally.

**Python:** Decode with OpenCV, convert BGR to RGB, and pass the NumPy array to the request.

### Ask with a direct image {#step-direct-image}

Attach the image directly to the first request. This is the simplest path and is often enough for one-shot visual questions.

### Cache the image embedding {#step-cache-image}

Call `encode(...)` to cache image embeddings in the model. The call returns `true` when the image was accepted and cached.

### Ask follow-up questions {#step-follow-up}

Set `use_cached_images = true` on each request that should reuse the cached image. You can ask multiple questions about the same cached image. Requests without that flag behave normally: text-only requests use no image, direct-image requests use their own `images`, and another `encode(...)` call replaces the cached image.

### Attach an image to a chat message {#step-message-image}

When you use `messages`, attach images to the user message that needs them. This keeps the image next to the exact text it belongs to.

## Run

First, download a VLM such as Qwen3-VL 4B from Hugging Face using the LLiMa CLI:

```bash
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

Run the tutorial on Modalix with the deployed model directory and a local image:

**Python:**
```bash
python3 share/sima-neat/tutorials/020_run_a_vlm/run_a_vlm.py \
  --model /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --image tests/images/people.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --image tests/images/people.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_020_run_a_vlm
./build/tutorials-standalone/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --image tests/images/people.jpg
```

Expected output is one answer from a direct image request, multiple follow-up answers that reuse the cached image, and one answer from a message-level image request.

## In Practice

Use image caching when the user asks several questions about the same frame, product image, diagram, or document page. Avoid caching when each request uses a different image because the direct-image path is simpler and keeps prompt state obvious.

Some model families may not support cached reuse. In that case, use direct images on each request.

Use `ChatMessage.images` when you are building a conversation and only one message should carry the image. Use top-level `GenerationRequest.images` for the simpler one-prompt shape.

## Source Files
- C++: `tutorials/020_run_a_vlm/run_a_vlm.cpp`
- Python: `tutorials/020_run_a_vlm/run_a_vlm.py`
