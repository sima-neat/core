# 019 Run an LLM

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 10 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4 |
| Labels | genai, llm, chat, history, streaming |

## Concept

Load a GenAI model directory, send one simple prompt, add a system prompt, then grow the same pattern into chat history and streaming.

## Walkthrough

The classic `Model` tutorials use `.tar.gz` MPK archives. GenAI models use LLiMa model directories and the `neat::genai` API instead. Start with the smallest request: load a model, set `request.prompt`, run it, and print the answer. Once that works, switch to `request.messages` when you need conversation state.

### Load the model directory {#step-load-model}

Point `GenAIModel` at a deployed LLiMa model directory. This tutorial uses `GenAIModel` because it auto-detects whether the directory is an LLM, VLM, or ASR model.

**C++:** Construct `simaai::neat::genai::GenAIModel` from the model path.

**Python:** Construct `pyneat.genai.GenAIModel` from the model path.

### Send one prompt {#step-send-prompt}

Build a `GenerationRequest` with `prompt` and a token budget. This is the shortest path for one-off questions, tests, and scripts.

### Define a system prompt {#step-system-prompt}

Use a short system instruction to steer the model's behavior. You can attach it to a simple prompt request with `system_prompt`; when you switch to chat history, carry the same instruction into the message list as a `system` message.

### Switch to messages {#step-store-history}

For chat-style requests, use `messages` instead of `prompt`: start with a system message and a user message, run the request, then store the assistant response. The model does not remember earlier `run()` calls by itself; your application owns the message history.

### Ask a follow-up with history {#step-follow-up}

Append another user message, send the updated message list, and read the answer. The model now sees the full conversation your application kept.

### Stream an answer {#step-stream-answer}

For UI-style output, call `stream()` and iterate the returned `GenerationStream`. Each token sample contains the latest text fragment.

## Run

On the Modalix DevKit, download an LLM such as Qwen3 4B from Hugging Face using the LLiMa CLI:

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

Run the tutorial on Modalix with the DevKit-local model directory:

**Python:**
```bash
python3 share/sima-neat/tutorials/019_run_an_llm/run_an_llm.py \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_019_run_an_llm
./build/tutorials-standalone/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

Expected output is a simple prompt answer, a system-prompted answer, a context-aware follow-up, and a streamed final response.

## In Practice

Keep only the amount of message history your application needs. Long histories consume context tokens and increase time to first token. For persistent chat applications, store the conversation outside the model object and rebuild `GenerationRequest.messages` for each turn.

## Source Files
- C++: `tutorials/019_run_an_llm/run_an_llm.cpp`
- Python: `tutorials/019_run_an_llm/run_an_llm.py`
