---
title: "Model Zoo"
sidebar_position: 5
---

# Model Zoo

Model Zoo는 SiMa 장치에서 바로 실행할 수 있도록 미리 컴파일되고 양자화된 모델을 엄선한 컬렉션입니다.

다음과 같은 경우에 사용하세요.

- Modalix 하드웨어에서 모델의 정확도와 성능을 평가합니다.
- 이미 알려진 모델에 대해서는 수동 컴파일 및 양자화를 피합니다.
- 검증된 모델 아티팩트부터 시작합니다.
- 특정 하드웨어 대상에 맞게 구축된 모델을 선택합니다.

Model Zoo는 GenAI 모델을 제외한 Neat C++ 및 PyNeat 애플리케이션을 위한 미리 컴파일된 모델 아티팩트를 제공합니다.

사용 가능한 모델 목록:

```bash
sima-cli modelzoo list
```

다운로드하기 전에 모델을 검토하세요.

```bash
sima-cli modelzoo describe yolov5
```

모델 아티팩트를 다운로드하세요:

```bash
sima-cli modelzoo get yolov5s
```

모델 이름은 릴리스에 따라 다를 수 있습니다. 어떤 모델 식별자를 사용해야 할지 확실하지 않은 경우 `sima-cli modelzoo list`를 먼저 사용하세요.

명령어에 대한 자세한 내용은 해당 [`sima-cli modelzoo`](/tools/sima-cli/modelzoo/) 참조 자료를 참조하십시오.

## 생성형 AI 모델

GenAI용으로 SiMa.ai는 [Hugging Face](https://huggingface.co/simaai)에 사전 컴파일된 LLM, VLM 및 ASR 모델 컬렉션을 제공합니다. LLiMa CLI를 사용하여 다운로드하십시오.

```bash
llima pull <model_name>
```

예를 들어:

```bash
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

GenAI 모델을 다운로드한 후에는 LLiMa 런타임을 사용하여 DevKit에서 실행합니다. 설정 및 런타임 명령은 [LLiMa를 활용한 생성형 AI](/genai-llima/)를 참조하십시오.
