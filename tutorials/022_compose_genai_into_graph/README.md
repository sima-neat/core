# 022 Compose GenAI into a Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, graph, composition, streaming, advanced |

## Concept

Use GenAI graph fragments when LLM, VLM, or ASR work is one stage in a larger Neat graph.

## Walkthrough

Most GenAI applications should start with direct model APIs. Graph composition becomes useful when GenAI needs to sit beside other Neat stages, named inputs, named outputs, routing, or application-level orchestration.

### Create a GenAI graph fragment {#step-create-fragment}

Create a task-specific model handle, configure graph-fragment options, and build a public `Graph` fragment.

The vision-language fragment exposes `prompt`, `image`, and `use_cached_image` inputs plus `tokens`, `done`, `encoded`, and `error` outputs. The speech transcriber fragment exposes `audio` and `audio_path` inputs plus `tokens`, `done`, and `error` outputs.

### Add the fragment to an app graph {#step-compose-graph}

Add the fragment to a larger application graph. The fragment keeps its public endpoint names, so application code can push and pull by name.

### Build and push graph inputs {#step-push-prompt}

Build the graph into a `Run`, push an image sample to the `image` input, then push a text sample to the `prompt` input and let the GenAI stage produce tokens.

### Pull tokens and completion metadata {#step-pull-results}

Pull from `tokens` until a `done` sample arrives. The `done` sample is a bundle with fields such as generated token count and finish reason.

## Run

On the Modalix DevKit, download the LFM2-VL 1.6B VLM from Hugging Face using the LLiMa CLI:

```bash
llima pull LFM2-VL-1.6B-a16w4
```

Run the tutorial on Modalix with the DevKit-local model directory and a local image:

**Python:**
```bash
python3 share/sima-neat/tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_022_compose_genai_into_graph
./build/tutorials-standalone/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

Expected output prints the graph description and a streamed answer pulled from the `tokens` output.

## In Practice

Use this pattern when GenAI is part of a larger application graph. Keep direct `GenAIModel`, `VisionLanguageModel`, and `ASRModel` calls for simple request/response application code.

## Source Files
- C++: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.cpp`
- Python: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py`
