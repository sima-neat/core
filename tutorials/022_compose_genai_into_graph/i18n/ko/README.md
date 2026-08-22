# 022 GenAI를 그래프로 구성

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, graph, composition, streaming, advanced |

## Concept

LLM, VLM 또는 ASR 작업이 더 큰 작업의 한 단계일 때 GenAI 그래프 조각을 사용합니다. Neat 그래프.

## Walkthrough

대부분의 생성형 AI 애플리케이션은 직접 모델 API로 시작해야 합니다. 생성형 AI가 다른 모델과 함께 사용되어야 할 때 그래프 구성을 활용하면 유용합니다. Neat 단계, 명명된 입력, 명명된 출력, 라우팅 또는 애플리케이션 수준 오케스트레이션.

### GenAI 그래프 조각 생성 {#step-create-fragment}

작업에 특화된 모델 핸들을 생성하고, 그래프 조각 옵션을 구성한 다음, 공개 모델을 구축합니다. `Graph` 일부 내용.

시각-언어 일부 내용은 다음을 보여줍니다. `prompt`, `image`그리고 `use_cached_image` 입력값에 더하기 `tokens`, `done`, `encoded`그리고 `error` 출력합니다. 음성-텍스트 변환 기능의 일부는 다음을 노출합니다. `audio` 그리고 `audio_path` 입력값에 더하기 `tokens`, `done`그리고 `error` 출력.

`SpeechTranscriberOptions` 기본적으로 자동 언어 감지 및
음성 인식 기능을 사용합니다. 설정을 변경합니다. `task` ~로 `ASRTask::Translate` C++ 또는
`ASRTask.Translate` Python에서 음성을 영어로 번역합니다. `done`
번들은 감지된 원본 언어를 보고하며, 가능한 경우
`no_speech_prob` 그리고 `avg_logprob`.

### 앱 그래프에 조각을 추가합니다. {#step-compose-graph}

이 조각을 더 큰 애플리케이션 그래프에 추가합니다. 이 조각은 공개 엔드포인트 이름을 유지하므로 애플리케이션 코드는 이름을 사용하여 데이터를 푸시하고 가져올 수 있습니다.

### 그래프 입력 생성 및 푸시 {#step-push-prompt}

그래프를 기반으로 `Run`이미지 샘플을 푸시하여 `image` 입력한 후 텍스트 샘플을 다음으로 전송합니다. `prompt` 입력을 제공하고 GenAI 단계에서 토큰을 생성합니다.

### 토큰과 완료 메타데이터를 가져옵니다. {#step-pull-results}

다음에서 가져오기 `tokens` ~까지 `done` 샘플이 도착합니다. `done` 샘플은 생성된 토큰 수 및 완료 사유와 같은 필드를 포함하는 묶음입니다.

## Run

 Modalix DevKitLFM2-VL 1.6B VLM을 다운로드하세요. Hugging Face 다음 텍스트를 사용하여 LLiMa CLI:

```bash
llima pull LFM2-VL-1.6B-a16w4
```

튜토리얼을 실행하여 Modalix 다음과 함께 DevKit-로컬 모델 디렉터리와 로컬 이미지:

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

예상 출력은 그래프 설명을 출력하고, `tokens` 출력에서 가져온 스트리밍된 답변을 표시합니다.

## In Practice

GenAI가 더 큰 애플리케이션 그래프의 일부일 때 이 패턴을 사용합니다. 간단한 요청/응답 애플리케이션 코드에는 직접적인 `GenAIModel`, `VisionLanguageModel` 및 `ASRModel` 호출을 유지합니다.

## 소스 파일
- C++: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.cpp`
- Python: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py`
