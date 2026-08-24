# 021 GenAI 모델 배포

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Beginner |
| Estimated Read Time | 15-20 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4, Qwen3-VL-4B-Instruct-GPTQ-a16w4, whisper-small-a16w8 |
| Labels | genai, server, llm, vlm, asr, http |

## Concept

Neat GenAI 서버 뒤에 여러 개의 GenAI 모델을 배포하여 UI, 서비스 또는 원격 클라이언트가 단일 프로세스에서 LLM, VLM 및 ASR 엔드포인트를 호출할 수 있도록 합니다.

## Walkthrough

대부분의 애플리케이션의 경우 `GenAIServer`와 OpenAI 호환 `POST /v1/chat/completions` 엔드포인트로 시작합니다. 임베디드 애플리케이션 로직이 동일한 프로세스에서 모델 호출을 소유해야 하는 경우 직접 `model.run(request)` 호출을 사용합니다.

전체 엔드포인트 및 요청 계약은 [GenAI 서버 참조](/develop-apps/development-workflow/genai-model/genai-server)를 참조하십시오.

### 서버 구성 {#step-configure-server}

호스트와 포트를 선택합니다. 기본 호스트는 `0.0.0.0`이며, 이는 Modalix 장치에 연결할 수 있는 다른 시스템의 연결을 허용합니다.

### 모델 디렉터리 등록 {#step-register-models}

배포된 각 모델 디렉터리를 배포된 이름과 함께 추가합니다. 이 튜토리얼에서는 `llm`, `vlm` 및 `asr`을 등록합니다. 배포된 이름은 클라이언트가 `model` 필드에 보내는 이름입니다.

### 서비스 시작 {#step-start-serving}

차단되는 전경 프로세스의 경우 `serve()`를 호출하고, 애플리케이션이 프로세스의 나머지 수명 주기를 소유하는 경우 `start()`를 호출합니다.

서버가 시작된 후 `GET /v1/models`를 사용하여 등록된 모델 이름을 확인합니다.

```bash
curl http://<modalix-ip>:9998/v1/models
```

응답에는 이 튜토리얼에서 등록된 배포된 이름인 `llm`, `vlm` 및 `asr`이 포함되어야 합니다.

## Run

Modalix DevKit에서 Hugging Face에서 LLM, VLM 및 ASR 모델을 LLiMa CLI를 사용하여 다운로드합니다.

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
llima pull whisper-small-a16w8
```

Modalix에서 세 개의 DevKit 로컬 모델 디렉터리가 모두 있는 상태로 서버를 시작합니다.

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

개발 중에 하위 집합만 배포하려는 경우 `--vlm` 또는 `--asr`을 제거합니다.

서버가 실행 중인 후에는 먼저 모든 배포된 이름이 등록되었는지 확인합니다.

```bash
curl http://<modalix-ip>:9998/v1/models
```

그런 다음 클라이언트에서 엔드포인트를 호출합니다. `<modalix-ip>`를 Modalix 장치의 IP 주소 또는 호스트 이름으로 바꿉니다.
아래의 요청 클라이언트는 Python `requests`를 사용하여 응답을 스트리밍하고, 서버 측 TTFT와 평균, 최소 및 최대 토큰당 TPS를 보고할 때 출력합니다.

### LLM에 대한 텍스트 요청

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_text.py \
  --server-ip <modalix-ip> \
  --model llm \
  "Give me three tips for designing a small REST API."
```

### LLM에 대한 도구 호출 요청

OpenAI와 호환되는 `POST /v1/chat/completions` 엔드포인트와 Ollama와 호환되는 `POST /api/chat` 엔드포인트는 `tools` 배열에서 함수 정의를 허용합니다. 각 항목에는 `type: "function"`, `function` 객체 및 비어 있지 않은 문자열 `function.name`이 있어야 합니다. 함수 설명과 JSON 스키마 매개변수는 `function` 객체 내에 포함될 수 있습니다.

```bash
curl http://<modalix-ip>:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "stream": false
  }'
```

`tool_choice`를 `"auto"`로 설정하여 모델이 선언된 도구를 선택하도록 하거나, `"none"`으로 설정하여 도구 프롬프트 및 구문 분석을 비활성화합니다. `tool_choice`를 생략하거나 `null`로 설정하면 `tools`가 비어 있지 않은 경우 `"auto"`와 동일하게 작동합니다. 잘못된 형식의 도구 정의, 배열이 아닌 `tools` 및 지원되지 않는 `tool_choice` 값 또는 유형은 HTTP 400 오류와 함께 `invalid_request_error`를 반환합니다. 동일한 도구 정의 및 도구 선택 검증은 추론이 시작되기 전에 직접 `GenerationRequest` 호출에도 적용됩니다.

### VLM에 대한 텍스트 및 이미지 요청

요청 스크립트는 이미지를 base64로 인코딩하고 OpenAI와 호환되는 `image_url` 콘텐츠 부분으로 보냅니다.

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_image.py \
  --server-ip <modalix-ip> \
  --model vlm \
  image.jpg \
  "What is the main subject of this image?"
```

### ASR 모델에 대한 오디오 요청

트랜스크립션 클라이언트는 기본적으로 자동 소스 언어 감지를 사용합니다. 소스 언어를 알고 있는 경우 `--language`를 사용합니다.

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  speech.wav
```

음성을 영어로 번역하려면 `--translate`를 추가합니다. 클라이언트는 동일한 멀티파트 요청을 `POST /v1/audio/translations`로 보냅니다.

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  --translate \
  speech-in-another-language.wav
```

트랜스크립션은 `POST /v1/audio/transcriptions`를 사용합니다. 두 경로 모두 `stream=true`를 지원합니다. 제공된 클라이언트는 텍스트를 스트리밍하고, 감지된 소스 언어, `no_speech_prob` 및 `avg_logprob`를 최종 이벤트에서 출력합니다. `no_speech_prob` 값이 높을수록 Whisper는 입력에 음성이 없을 가능성이 더 높다고 판단합니다. `avg_logprob`는 생성된 토큰의 평균 로그 확률이며, 값이 높을수록(음수가 작을수록) 디코딩에 대한 신뢰도가 높다는 것을 나타냅니다.

## In Practice

네트워크 경계가 유용할 때 서버를 사용합니다. 동일한 프로세스 내에서 오버헤드가 적은 애플리케이션 코드를 위해 직접 `GenAIModel`, `VisionLanguageModel` 및 `ASRModel`를 호출합니다.

일반적인 애플리케이션의 경우 여러 모델 이름을 사용하여 하나의 `GenAIServer` 프로세스를 실행합니다. DevKit에 충분한 메모리가 있는 경우 여러 서버 프로세스가 다른 포트에 바인딩될 수 있지만, 각 프로세스는 자체 모델 인스턴스를 로드하고 여전히 동일한 MLA 하드웨어 게이트키퍼를 공유하므로 하드웨어 처리량을 늘리는 방법으로 간주해서는 안 됩니다.

`/v1/models` 엔드포인트는 가장 빠른 간단한 테스트입니다. 이 엔드포인트에서 제공되는 모델 이름을 반환하면 서버에 연결할 수 있고 모델 레지스트리가 채워진 것입니다.

## 소스 파일
- C++: `tutorials/021_serve_genai_models/serve_genai_models.cpp`
- Python: `tutorials/021_serve_genai_models/serve_genai_models.py`
- 요청 클라이언트:
  - `tutorials/021_serve_genai_models/request_chat_completion_text.py`
  - `tutorials/021_serve_genai_models/request_chat_completion_image.py`
  - `tutorials/021_serve_genai_models/request_audio_transcription.py`
