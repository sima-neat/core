# 019 직접 API를 사용하여 LLM 실행

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Intermediate |
| Estimated Read Time | 10 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4 |
| Labels | genai, llm, chat, history, streaming |

## Concept

GenAI 모델 디렉터리를 로드하고, 간단한 프롬프트를 보내고, 시스템 프롬프트를 추가한 다음, 동일한 패턴을 대화 기록 및 스트리밍으로 확장합니다.

## Walkthrough

클래식 `Model` 튜토리얼은 `.tar.gz` MPK 아카이브를 사용합니다. GenAI 모델은 대신 LLiMa 모델 디렉터리와 `neat::genai` API를 사용합니다. 가장 작은 요청으로 시작합니다. 모델을 로드하고, `request.prompt`를 설정하고, 실행한 다음, 답변을 출력합니다. 작동하면 대화 상태가 필요할 때 `request.messages`로 전환합니다.

### 모델 디렉터리 로드 {#step-load-model}

배포된 LLiMa 모델 디렉터리를 `GenAIModel`에 지정합니다. 이 튜토리얼에서는 `GenAIModel`을 사용하는데, 이는 디렉토리가 LLM, VLM 또는 ASR 모델인지 자동으로 감지하기 때문입니다.

**C++:** 모델 경로에서 `simaai::neat::genai::GenAIModel`을 생성합니다.

**Python:** 모델 경로에서 `pyneat.genai.GenAIModel`을 생성합니다.

### 단일 프롬프트 보내기 {#step-send-prompt}

`prompt` 및 토큰 예산과 함께 `GenerationRequest`를 빌드합니다. 이것은 일회성 질문, 테스트 및 스크립트에 대한 가장 간단한 방법입니다.

### 시스템 프롬프트 정의 {#step-system-prompt}

짧은 시스템 지침을 사용하여 모델의 동작을 제어합니다. `system_prompt`를 사용하여 간단한 프롬프트 요청에 연결할 수 있습니다. 대화 기록으로 전환하면 동일한 지침을 메시지 목록의 `system` 메시지로 전달합니다.

### 메시지로 전환 {#step-store-history}

대화 스타일 요청의 경우 `prompt` 대신 `messages`를 사용합니다. 시스템 메시지와 사용자 메시기로 시작하고, 요청을 실행한 다음, 어시스턴트 응답을 저장합니다. 모델은 이전 `run()` 호출을 자체적으로 기억하지 않습니다. 애플리케이션이 메시지 기록을 소유합니다.

### 기록과 함께 후속 질문 {#step-follow-up}

다른 사용자 메시지를 추가하고, 업데이트된 메시지 목록을 보내고, 답변을 읽습니다. 이제 모델은 애플리케이션이 유지한 전체 대화를 볼 수 있습니다.

### 답변 스트리밍 {#step-stream-answer}

UI 스타일 출력의 경우 `stream()`을 호출하고 반환된 `GenerationStream`을 반복합니다. 각 토큰 샘플에는 최신 텍스트 조각이 포함됩니다.

## Run

Modalix DevKit에서 Hugging Face의 LLiMa CLI를 사용하여 Qwen3 4B와 같은 LLM을 다운로드합니다.

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

Modalix에서 DevKit 로컬 모델 디렉터리를 사용하여 튜토리얼을 실행합니다.

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

예상 출력은 간단한 프롬프트 응답, 시스템 프롬프트 응답, 상황에 맞는 후속 응답, 그리고 스트리밍 방식으로 제공되는 최종 응답입니다.

## In Practice

애플리케이션에 필요한 메시지 기록의 양만 유지합니다. 긴 기록은 컨텍스트 토큰을 소모하고 첫 번째 토큰을 생성하는 데 걸리는 시간을 늘립니다. 지속적인 채팅 애플리케이션의 경우, 대화 내용을 모델 객체 외부에 저장하고 각 단계마다 `GenerationRequest.messages`를 다시 구성합니다.

## 소스 파일
- C++: `tutorials/019_run_an_llm/run_an_llm.cpp`
- Python: `tutorials/019_run_an_llm/run_an_llm.py`
