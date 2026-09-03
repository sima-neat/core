---
title: "CVU 커널 및 그래프 카탈로그"
description: "프레임워크의 CVU 측 커널이 수행하는 작업과 이를 기반으로 전처리/후처리 그래프가 어떻게 구성되는지 설명합니다."
sidebar_position: 3
slug: /develop-apps/advanced-concepts/cvu_kernels
---

# CVU 커널 및 그래프 카탈로그

이 프레임워크는 CVU(EV74) 커널의 작은 카탈로그를 제공하며, 이 커널들은 각 모델에 대해 플래너가 선택하는 전처리 및 후처리 그래프로 결합됩니다. 이 페이지에서는 커널과 이를 구성하는 그래프 패밀리에 대해 설명합니다.

## 커널 패밀리

### 전처리 커널

EV74에서 입력과 MLA 사이에서 실행됩니다. 표준 패밀리:

- **Resize** — 양선형/최근접 스케일링, 선택적으로 레터박스 또는 중앙 자르기 적용.
- **색상 변환** — RGB ↔ BGR, NV12 → RGB, I420 → RGB, GRAY8 ↔ 패킹.
- **Layout convert** — HWC ↔ CHW, 축 순열.
- **Normalize** — 채널별 평균/표준 편차 (FP32 입력, FP32 출력).

### 경계 커널

MLA 경계를 넘어 FP32 / BF16 / INT8을 연결합니다.

- **Quant** — FP32 → INT8, 스케일 + 제로 포인트 적용.
- **Dequant** — INT8 → FP32, 스케일 + 제로 포인트 적용.
- **Cast** — FP32 ↔ BF16 (스케일/제로 포인트 없음).
- **Tess** / **Detess** — MLA 타일 지오메트리로 레이아웃을 재배치하거나 재배치 해제. 동일한 바이트, 다른 순서.

### 융합 커널

모델의 계약이 경계 커널을 요구하지만 MLA 단계에 포함하지 않는 경우 플래너가 선택하는 조합:

- **QuantTess** — Quant + Tess 융합.
- **DetessDequant** — Detess + Dequant 융합.
- **CastTess** / **DetessCast** — BF16 경로에서 Cast와 Tess 융합.

### 일반 전처리

애플리케이션에서 임의의 사용자 정의 변환을 제공하는 경우 플래너는 전처리 그래프를 일반 변형으로 업그레이드하여 해당 변환을 단일 CVU 커널로 융합합니다. MLA 경계에서의 계약은 변경되지 않습니다.

### BoxDecode

탐지 모델에 대한 NMS / 디코드를 융합하는 후처리 커널입니다. 출력 샘플에 `DetectionMeta`를 생성합니다. [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h)를 참조하십시오.

## 그래프 구성 방법

네 개의 `PreprocessGraphFamily` 값은 네 개의 커널 체인에 매핑됩니다.

| 그래프 패밀리 | 체인 (입력 → MLA) |
|--------------|---------------------|
| `Preproc` | Resize → ColorConvert → Normalize → MLA (내부적으로 타일링) |
| `Quant` | Resize → ColorConvert → Normalize → Quant → MLA (내부적으로 타일링) |
| `Tess` | Resize → ColorConvert → Normalize → Tess → MLA |
| `QuantTess` | Resize → ColorConvert → Normalize → QuantTess → MLA |

출력 측의 대응되는 요소 — `Postproc` / `Detess` / `DetessDequant` / 패스 스루 — 는 MLA의 컴파일된 출력 커널에 detess/dequant가 포함되어 있는지 여부에 따라 달라집니다.

이 네 가지 패밀리가 존재하는 이유는 [데이터 유형 계약](/develop-apps/advanced-concepts/dtype_contract)를 참조하십시오.

## 커널 명명 규칙

프레임워크 내에서 커널은 `RoutePlanner` 결정 및 `MeasureReport` 플러그인/커널 타이밍 행에 표시되는 안정적인 문자열 이름으로 참조됩니다.

- `cvu/preproc/<variant>` — 전처리 커널.
- `cvu/quant/<dtype>` — 양자화 변형.
- `cvu/tess/<geometry>` — 테셀레이션 변형.
- `cvu/postproc/box_decode_<type>` — BoxDecode 변형.

정확한 카탈로그는 `core/src/pipeline/internal/sima/` (프레임워크의 직접 액세스 레이어)에 있습니다.

## 추가 정보

- “CVU 커널 및 그래프 카탈로그” — 디자인 심층 분석의 §86, §87.
- “테셀레이션, 양자화, 캐스팅” — 디자인 심층 분석의 §17.
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — 네 개의 모서리를 갖는 열거형.
- [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h) — 후처리 박스 디코딩.
