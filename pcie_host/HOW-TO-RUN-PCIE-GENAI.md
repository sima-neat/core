# PCIe GenAI: direct LLM, VLM and ASR APIs

Each `pcie::genai::GenAIModel` / `pyneatpcie.genai.GenAIModel` owns one
`neat-pcie-genai-worker` process on the card. That worker uses Core's local
GenAI API and LLiMa. Multiple handles have independent workers and sessions;
there is no shared application-level model registry or implicit chat history.

This implementation has been compiled in the SDK. Hardware validation,
multi-model resource limits and mixed GenAI/YOLO performance are still pending.

## Prerequisites

- Matching Core worker and PCIe host packages, including the GenAI API.
- Platform host/card drivers, service libraries and running daemons that support
  tagged notifications, relative destination file transfers and host
  `simaai_svc_open_card`. NEAT does not install or upgrade those daemons.
- Passwordless SSH on port 22 to the selected card user. SSH starts/stops the
  worker only; model assets, images, audio, prompts and results use real PCIe.
- The model runtime directory, containing `devkit/` and `elf_files/`, under
  a configured host daemon serve root (default root name: `models`).
  If the package wraps this in `sima_files/`, include that suffix in the model
  name passed to the constructor.
- For image/audio inputs: a writable host staging directory matching the host
  daemon's `data` serve root. Set `media_directory` to its actual path.
  Text-only requests do not require a host media directory.
- `card_receive_directory` must match the card daemon's configured default receive
  root and be writable by the SSH user and daemon. The defaults are
  `/srv/simaai/data` on the host and `/srv/simaai/incoming` on the card;
  configure them to match your installation, not the other way around.

Each session uses private `neat-genai/<session>/` subdirectories. Do not change
model assets while a worker is loading. Serve/receive roots must be trusted;
session tags provide routing isolation, not authentication against a malicious
local daemon client.

## Python

```python
from pyneatpcie import genai

connection = genai.ConnectionOptions()
connection.card_host = "10.0.0.2"
connection.user = "sima"
connection.media_directory = "/srv/simaai/data"
connection.card_receive_directory = "/srv/simaai/incoming"

request = genai.GenerationRequest()
request.prompt = "Explain PCIe in one sentence."
request.max_new_tokens = 64

with genai.GenAIModel("my-llm/sima_files", connection) as model:
    print(model.run(request).text)
    # A new request is independent; use messages for explicit conversation history.
    for sample in model.stream(request):
        print(sample.text, end="", flush=True)
```

For a VLM, set `request.image_files = ["image.jpg"]`, or set `request.images`
to a list of NumPy `uint8[H,W,3]` RGB arrays / PCIe tensors. OpenCV images must
be converted from BGR to RGB first. Request values are staged before
`stream()` returns, so the caller may reuse its arrays afterward.

For ASR, use a compatible Whisper runtime directory and set
`request.audio_file = "speech.wav"` instead of a prompt. Alternatively:

```python
import numpy as np

request = genai.GenerationRequest()
request.audio = np.zeros(16000, dtype=np.float32)  # replace with mono samples
request.sample_rate = 16000
request.language = "en"
request.asr_task = genai.ASRTask.Transcribe
```

Results preserve Core's text, reasoning, tool calls, metrics and ASR fields
(`language`, `no_speech_prob`, `avg_logprob`). Compatibility of tools and
media remains model-dependent and is validated by Core.

## C++

```cpp
#include <simaai/neat/pcie/genai/GenAIModel.h>
#include <iostream>

int main() {
  namespace genai = simaai::neat::pcie::genai;
  genai::ConnectionOptions connection;
  connection.user = "sima";
  genai::GenAIModel model("my-llm/sima_files", connection);
  genai::GenerationRequest request;
  request.prompt = "Explain PCIe in one sentence.";
  request.max_new_tokens = 64;
  for (const auto& sample : model.stream(request))
    std::cout << sample.text << std::flush;
  model.close();
}
```

Build with the installed `find_package(SimaPCIeHost REQUIRED)` and link
`SimaPCIeHost::sima_neat_pcie_host`. GenAI uses C++20 and the installed
`nlohmann-json3-dev` dependency; it needs no full Core or LLiMa library on the host.

## Several models and YOLO

Create one handle per LLM/VLM/ASR model. Use separate application threads for
simultaneous generation. A handle accepts one active request; another request
on that handle is rejected until the terminal result has arrived.

Existing vision `pcie::Model` / `pcie::Runtime` APIs continue using their
own queues 0–3. GenAI uses the daemon-managed service channels, not one of
those queue claims. The platform daemon owns channel selection; do not
manually open its channels. Independent workers avoid LLiMa's process-wide
lock, but memory, MLA/EV74 capacity and platform scheduling still constrain
the supported model combinations. No aggregate concurrency/performance
guarantee is implied until hardware validation.

## Cancellation and failure

Call `stream.cancel()` to cancel a request and continue draining `next()`
to its terminal result. Destroying a stream requests cancellation. Call
`model.close()` explicitly, or use the Python context manager.

Every event carries session, request and sequence IDs. The worker retains up
to 32 unacknowledged events plus a terminal event, retries them, and waits at
most 30 seconds for acknowledgement. Requests and messages are deduplicated.
Missing events never silently become a successful truncated answer.

The host's unread sample queue is bounded to 32 entries. A consumer that
does not drain it fails the session instead of growing memory indefinitely.
A worker without host heartbeats expires after 30 seconds; the host reports
a worker that stops replying after 15 seconds. Startup/request timeouts default
to 15 minutes and are configurable. Blocking driver/service calls and model
loading are not immediately cancellable: orderly close asks the session's
worker to stop, then escalates only for that PID/start-time/session identity.

Normal completion removes owned media. Failed SSH shutdown retains host media
for a retry of `close()`; forced termination can leave card session files.
Worker logs and PID/start-time records remain in
`~/.cache/neat-genai/<session>/` for diagnosis. Never sweep a shared receive
root while another worker or file transfer may be active.

LoRA and speculative-decoding packages are not supported by this remote API.
No old `pcie-genai-backend`, `llima run --pcie`, queue status file or
implicit conversation state is used.

## CLI

The CLI is a small consumer of the same public API:

```bash
pcie-genai my-llm/sima_files "Hello" --host 10.0.0.2 --user sima
pcie-genai my-vlm/sima_files "Describe this" --user sima --image image.jpg
pcie-genai my-whisper/sima_files "" --user sima --audio speech.wav
```

It performs one request; applications manage multi-turn history through
`GenerationRequest.messages`.

## Building this branch

- Build LLiMa's local core/dev DEBs first and extract them into the SDK sysroot.
  Do not download a remote LLiMa package over that local build.
- Core's Internals dependency is pinned to `release-3.0.0-prep`.
- The card build installs `neat-pcie-genai-worker` with Core.
- Host builds default `SIMAPCIE_BUILD_GENAI=ON` and require the platform
  `simaai_svc.h` development header. Set `SIMAPCIE_SVC_INCLUDE_DIR` if it is
  not installed in a standard include location. Package-build runners need
  this header as well; no platform library is bundled.
- Vision-only builds may set `SIMAPCIE_BUILD_GENAI=OFF`.
- Hardware tests are a separate acceptance step; compiling does not establish
  daemon compatibility, cancellation latency or multi-model correctness.
