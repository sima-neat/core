---
title: "Model Browser"
sidebar_position: 4
---

# Model Browser

[Palette Model Browser](https://developer.sima.ai/models/)는 SiMa.ai가 Modalix용으로 컴파일하고 측정한 모델을 나열합니다. 각 모델 페이지는 Modalix DevKit에서 측정한 성능과 정확도를 보여 주며, 컴파일된 모델을 다운로드로 제공합니다.

다음과 같은 경우에 사용하십시오.

- 객체 감지나 텍스트 생성과 같은 작업에 맞는 모델을 찾을 때
- 모델과 그 변형의 성능 및 정확도를 비교할 때
- Neat에서 바로 실행할 수 있는 컴파일된 모델을 다운로드할 때
- 공개된 스크립트로 모델을 직접 다시 빌드할 때

## sima-cli로 모델 다운로드

`sima-cli`는 터미널에서 동일한 모델을 읽습니다. 계정은 필요하지 않습니다.

모델을 검색합니다.

```bash
sima-cli models list -q rf_detr_nano
```

모델을 선택한 다음 변형을 선택합니다. 변형은 모델을 컴파일하는 한 가지 방식이며, 가중치와 활성화의 비트 폭을 따서 이름이 지정됩니다. 명령은 측정된 정확도와 성능이 포함된 모델 카드를 표시한 다음 모델을 다운로드할지 묻습니다.

ID와 변형을 알고 있다면 모델을 직접 다운로드합니다.

```bash
sima-cli models download --id rf_detr_nano --variant w8a8_w16a16
```

명령은 모델을 `models/<model id>/<variant>/` 아래에 저장합니다. 다른 위치에 저장하려면 `--output <folder>`를 추가하십시오.

라이선스가 재배포를 금지하는 모델은 다운로드할 수 없습니다. 다음 섹션에서 설명하는 대로 스크립트로 다시 빌드하십시오.

명령에 대한 자세한 내용은 [`sima-cli models`](/tools/sima-cli/models/) 참조 문서를 참조하십시오.

## 모델 재현

모든 모델에는 해당 모델을 빌드하고 측정한 스크립트가 함께 제공되며, [models 리포지토리](https://github.com/sima-neat/models)에 공개되어 있습니다. 각 모델에는 `models/<model id>/` 폴더가 있고, 단계마다 스크립트가 하나씩 있습니다.

| 단계 | 스크립트 | 실행 위치 |
| --- | --- | --- |
| 공식 모델을 ONNX로 내보내기 | `convert.py` | 사용자 컴퓨터 |
| 양자화 및 컴파일 | `compile.py` | Palette SDK의 Model Compiler |
| 성능 측정 | `performance.py` | Modalix DevKit |
| 정확도 측정 | `accuracy.py` | Modalix DevKit |

같은 폴더의 `model.yaml` 파일에는 빌드할 수 있는 변형이 `build_profiles` 아래에 나열되어 있으며, 각 단계에 필요한 데이터 세트도 나열되어 있습니다.

[시작 가이드](https://developer.sima.ai/models/docs/getting-started)는 복사해서 사용할 수 있는 명령과 함께 모든 단계를 안내하고, 결과를 게시된 값과 비교하는 방법을 보여 줍니다.
