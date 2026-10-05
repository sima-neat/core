# 029 PCIe를 통해 GenAI 모델 실행하기

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | Qwen3-0.6B-Autoround-a16w4, LFM2-VL-450M-Autoround-a16w4, whisper-small-a16w8 |
| Labels | PCIe, GenAI, LLM, VLM, Whisper |

## Concept

호스트 애플리케이션에서 Modalix PCIe 카드로 텍스트 생성, 이미지 질문, 음성 전사를 실행합니다. 모델 파일, 요청, 결과는 PCIe를 통해 전송되며, 애플리케이션은 GStreamer 파이프라인 대신 공개 GenAI API를 사용합니다.

## Walkthrough

각 예제는 호스트의 준비된 LLiMa 모델 디렉터리를 로드하고, 한도가 있는 요청을 실행한 뒤 모델을 닫습니다. 먼저 예제를 각각 실행하세요. 연결 설정은 같지만 요청 입력은 다릅니다.

### LLM 로드하기 {#step-load-llm}

C++에서는 `pcie::genai::GenAIModel`, Python에서는 `pyneatpcie.genai.GenAIModel`을 생성합니다. 생성 시 카드에 모델을 로드하므로 `build()` 호출이나 큐 선택이 필요하지 않습니다. `connection.card_id`로 카드를 선택합니다. 절대 모델 경로와 현재 작업 디렉터리 기준 상대 경로를 사용할 수 있습니다.

### 텍스트 요청 보내기 {#step-text-request}

`prompt`와 `max_new_tokens`를 설정하고 `run()`을 호출합니다. 결과에는 답변 텍스트와 생성 지표가 포함됩니다. 예제의 기본 생성 한도는 128토큰이며, `--max-tokens`로 변경할 수 있습니다.

### 답변 스트리밍하기 {#step-stream-answer}

LLM 예제에 `--stream`을 전달하면 대신 `TokenSample` 값을 순회합니다. 각 텍스트 조각이 도착하는 즉시 출력하세요. 마지막 샘플에는 완료된 생성 지표가 포함됩니다. 모델 하나당 한 번에 하나의 요청만 활성화할 수 있습니다.

### 모델 닫기 {#step-close-model}

C++는 추론 후 `close()`를 명시적으로 호출하고, 예외로 예제가 중단되면 소멸자가 정리합니다. Python은 컨텍스트 관리자를 사용하여 오류가 발생하더라도 블록을 나갈 때 모델을 닫습니다. 연결 제한 시간은 시작에 15분, 요청에 5분입니다.

### RGB 이미지 준비하기 {#step-prepare-image}

VLM 예제는 호스트에서 OpenCV로 이미지를 읽고, BGR을 RGB로 변환하여 형태가 `[H, W, 3]`인 연속 UInt8 픽셀을 만듭니다. C++는 픽셀을 소유하는 벡터를 PCIe 텐서로 감싸고, Python은 NumPy 배열을 제공합니다. 픽셀을 정규화하거나 카드 로컬 이미지 경로로 대체하지 마세요.

### 이미지에 대해 질문하기 {#step-image-request}

VLM을 로드하고 프롬프트와 함께 RGB 입력을 `request.images`에 첨부합니다. 이미지 전처리는 모델이 처리합니다. 반환된 텍스트가 질문에 대한 답변이며, PCIe 요청에는 이미지 파일 필드가 없습니다.

### Whisper로 음성 전사하기 {#step-audio-request}

Whisper 디렉터리를 로드하고 `audio_file`에 호스트 오디오 경로를 설정합니다. `language="auto"`로 언어를 감지하거나 `--language`로 원본 언어를 지정합니다. 기본 동작은 전사이며, `--translate`는 영어 번역을 선택합니다. 전사문, 언어, 선택적 비음성 확률을 확인하세요. ASR 요청에는 채팅 프롬프트를 추가하지 마세요.

## Run

[PCIe 호스트 설치](/getting-started/neat-library/pcie-host/)와 [튜토리얼 준비](/tutorials/before-you-run)를 따르세요. PCIe GenAI API를 제공하는 서로 호환되는 호스트 및 카드 패키지를 사용합니다. 아래 모든 명령은 호스트에서 압축 해제한 PCIe extras 루트를 기준으로 실행하세요.

### 준비된 모델 다운로드하기

이 SiMa.ai Model Zoo 저장소에는 이미 컴파일된 LLiMa 아티팩트가 포함되어 있습니다: [Qwen3](https://huggingface.co/simaai/Qwen3-0.6B-Autoround-a16w4), [LFM2-VL](https://huggingface.co/simaai/LFM2-VL-450M-Autoround-a16w4), [Whisper](https://huggingface.co/simaai/whisper-small-a16w8). 실행할 모델만 다운로드하세요. 모델 카드를 확인하고 설치된 플랫폼과 호환되는 아티팩트를 사용하세요.

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

토크나이저, 설정, 컴파일된 아티팩트를 포함하여 각 디렉터리를 온전히 유지하세요. 압축 해제나 카드로의 수동 복사는 필요하지 않습니다. 원본 Hugging Face 체크포인트는 준비된 LLiMa 디렉터리가 아닙니다. 사용자 지정 모델은 이 튜토리얼을 실행하기 전에 [LLiMa로 GenAI 사용하기](/genai-llima/)를 따르세요.

### 미디어 준비하기

VLM 명령은 번들에 포함된 거리 장면 이미지를 사용합니다. Whisper에는 호스트에서 짧은 음성 클립을 녹음하거나 준비하세요. FFmpeg가 설치되어 있다면 기존 녹음을 16 kHz 모노 WAV로 변환할 수 있습니다:

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le speech.wav
```

무음으로 전사 품질을 판단하지 마세요. 예제는 `--image` 또는 `--audio`를 통해 다른 이미지나 오디오 경로를 받을 수 있습니다.

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

C++ 소스 빌드에는 PCIe 개발 패키지와 OpenCV 개발 파일이 필요합니다. 모든 예제는 기본적으로 카드 0과 연결의 기본 SSH 사용자 root를 사용합니다. 설치 환경이 다르면 `--card N`, `--user sima`, `--ssh-key PATH`, `--card-host ADDRESS`를 추가하세요. 설정된 SSH 계정을 사용하세요. 예제는 접근 권한을 설정하지 않습니다.

LLM 답변, 장면 설명, 또는 녹음된 음성에 맞는 전사문이 출력됩니다. 표현은 모델과 입력에 따라 달라지며, 실행 성공만으로 품질을 확인할 수는 없습니다.

## In Practice

독립적인 모델에는 별도 핸들을 사용하고 카드 메모리와 연산 자원 한도를 고려하세요. 채팅에서는 `prompt` 대신 `messages`로 기록을 제공합니다. 독립적인 호출은 이전 대화를 기억하지 않습니다. 스트림을 취소할 때는 `cancel()`을 호출하고 마지막 샘플까지 소비한 후 모델을 재사용하세요.

API 개요는 [PCIe GenAI API](/develop-apps/development-workflow/genai-model/pcie-api)를 참조하세요. 독립형 GenAI 튜토리얼은 다른 네임스페이스와 로컬 모델 실행을 사용합니다. 해당 모델 핸들로 대체하지 마세요.

## 소스 파일

- `run_llm.cpp` / `run_llm.py`
- `run_vlm.cpp` / `run_vlm.py`
- `run_whisper.cpp` / `run_whisper.py`
- `tutorial_args.h` (공통 C++ 명령줄 파싱)
- `../assets/street-scene.png`
