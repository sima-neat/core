# 020 직접 API를 사용하여 VLM 실행

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, vlm, image, cache, multimodal |

## Concept

각 요청에 대해 이미지를 다시 인코딩하지 않고 동일한 이미지에 대해 반복적인 질문을 합니다.

## Walkthrough

비전-언어 모델은 텍스트와 이미지 텐서를 모두 입력으로 받을 수 있습니다. 하나의 질문에 대해 이미지를 `GenerationRequest.images`에 직접 첨부합니다. 반복적인 질문의 경우, 이미지를 한 번 인코딩하고 후속 요청에서 캐시된 이미지 임베딩을 재사용합니다.

### VLM 및 이미지 로드 {#step-load-inputs}

배포된 LLiMa 모델 디렉터리에서 `VisionLanguageModel`을 로드하고 디스크에서 이미지를 디코딩합니다.

**C++:** OpenCV를 사용하여 이미지를 읽습니다. Neat은 3채널 `cv::Mat` 입력을 BGR로 처리하고 내부적으로 RGB로 변환합니다.

**Python:** OpenCV로 디코딩하고, BGR을 RGB로 변환한 다음 NumPy 배열을 요청에 전달합니다.

### 직접 이미지를 사용하여 질문 {#step-direct-image}

첫 번째 요청에 이미지를 직접 첨부합니다. 이것은 가장 간단한 방법이며, 단일 시각적 질문에 충분한 경우가 많습니다.

### 이미지 임베딩 캐시 {#step-cache-image}

`encode(...)`를 호출하여 모델에 이미지 임베딩을 캐시합니다. 호출은 이미지가 수락되고 캐시되었을 때 `true`를 반환합니다.

### 후속 질문 {#step-follow-up}

캐시된 이미지를 재사용해야 하는 각 요청에서 `use_cached_images = true`를 설정합니다. 동일한 캐시된 이미지에 대해 여러 질문을 할 수 있습니다. 해당 플래그가 없는 요청은 정상적으로 작동합니다. 텍스트 전용 요청은 이미지를 사용하지 않고, 직접 이미지 요청은 자체 `images`를 사용하며, 다른 `encode(...)` 호출은 캐시된 이미지를 대체합니다.

### 채팅 메시지에 이미지 첨부 {#step-message-image}

`messages`를 사용하는 경우, 필요한 사용자 메시지에 이미지를 첨부합니다. 이렇게 하면 이미지가 해당 텍스트와 함께 유지됩니다.

## Run

Modalix DevKit에서 LLiMa CLI를 사용하여 Hugging Face에서 LFM2-VL 1.6B VLM을 다운로드합니다.

```bash
llima pull LFM2-VL-1.6B-a16w4
```

Modalix에서 DevKit 로컬 모델 디렉터리와 로컬 이미지를 사용하여 튜토리얼을 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/020_run_a_vlm/run_a_vlm.py \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_020_run_a_vlm
./build/tutorials-standalone/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

예상되는 출력은 직접 이미지 요청에 대한 하나의 답변, 캐시된 이미지를 재사용하는 여러 개의 후속 답변, 그리고 메시지 수준의 이미지 요청에 대한 하나의 답변입니다.

## In Practice

사용자가 동일한 프레임, 제품 이미지, 다이어그램 또는 문서 페이지에 대해 여러 질문을 할 때 이미지 캐싱을 사용합니다. 각 요청이 다른 이미지를 사용할 때는 캐싱을 피하십시오. 왜냐하면 직접 이미지 경로가 더 간단하고 프롬프트 상태를 명확하게 유지하기 때문입니다.

일부 모델 제품군은 캐시된 재사용을 지원하지 않을 수 있습니다. 이 경우 각 요청에 대해 직접 이미지를 사용하십시오.

대화를 구축하고 단 하나의 메시지만 이미지를 포함해야 할 때 `ChatMessage.images`를 사용합니다. 더 간단한 단일 프롬프트 형식에는 최상위 `GenerationRequest.images`를 사용합니다.

## 소스 파일
- C++: `tutorials/020_run_a_vlm/run_a_vlm.cpp`
- Python: `tutorials/020_run_a_vlm/run_a_vlm.py`
