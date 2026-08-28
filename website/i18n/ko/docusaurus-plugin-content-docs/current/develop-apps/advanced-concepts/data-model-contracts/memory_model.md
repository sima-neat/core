---
title: "메모리 모델"
description: "제로 복사 버퍼, 세그먼트, (버퍼 ID, 물리 주소, 가상 주소) 튜플, 그리고 캐시 일관성."
sidebar_position: 3
slug: /develop-apps/advanced-concepts/memory_model
---

# 메모리 모델

Neat 프레임워크의 런타임은 인코딩된 비디오 프레임, 디코딩된 YUV 평면, FP32 입력 텐서, INT8 양자화된 타일, MLA 스크래치패드 이미지 등 많은 바이트를 이동합니다. 명시적인 메모리 모델 없이 이를 수행하면 각 단계 경계에서 복사가 발생합니다. 이 페이지에서는 프레임워크가 이러한 복사를 어떻게 방지하는지 설명합니다.

## 버퍼 삼중: `(buffer_id, paddr, vaddr)`

프레임워크가 이동하는 모든 버퍼는 세 가지 요소로 식별됩니다.

- **`buffer_id`** — 런타임에서 버퍼의 수명 주기를 추적하는 데 사용하는 안정적인 정수(참조 횟수, 세그먼트 소유권).
- **`paddr`** — 물리 주소, IOMMU가 버퍼를 보는 방식. MLA / EV74 / DMA 하드웨어가 이를 확인합니다.
- **`vaddr`** — 가상 주소, 애플리케이션이 버퍼를 보는 방식. CPU 코드가 이를 참조합니다.

이 삼중은 복사 없이 양쪽(CPU 또는 가속기)에서 버퍼를 주소 지정할 수 있도록 합니다. 단일 할당은 *둘 다* 커널 페이지 테이블(소프트웨어가 읽을 수 있도록 함)과 IOMMU 페이지 테이블(하드웨어가 DMA를 수행할 수 있도록 함)에 나타납니다.

단계 간 핸드오프는 바이트가 아닌 삼중을 전달합니다.

## 세그먼트

버퍼는 명명된 **세그먼트**에서 가져옵니다. 세그먼트는 특정 할당자(DMA-BUF, CMA, ION, 일반 힙)에 의해 지원되고 누가 액세스할 수 있는지에 대한 메타데이터로 태그가 지정된 연속적인 메모리 영역입니다(CPU만, MLA만, 둘 다 등). 런타임은 각 버퍼에 대해 어떤 단계에서 해당 버퍼에 액세스할지 여부에 따라 적절한 세그먼트를 선택합니다.

예:

- `nv12_decode` 세그먼트에는 H.264에서 디코딩된 YUV가 포함됩니다. CPU는 진단 탭을 위해 읽을 수 있고, IOMMU는 크기 조정 노드를 위해 읽을 수 있습니다.
- `mla_input` 세그먼트에는 MLA에 전달되는 테셀레이션된 텐서가 포함됩니다. MLA 하드웨어만 읽을 수 있습니다. CPU 액세스를 위해서는 명시적인 매핑이 필요합니다.
- `model_output` 세그먼트에는 디테셀레이션 후의 FP32 텐서가 포함됩니다. CPU는 읽을 수 있으므로 애플리케이션에서 이를 가져올 수 있습니다.

`Tensor`는 삼중과 함께 해당 세그먼트를 전달하므로 프레임워크는 CPU 코드에서 피크/포크가 유효한지 여부를 알 수 있습니다.

## 캐시 일관성

MLA, EV74 및 CPU는 모두 자체 캐시를 가지고 있습니다. 버퍼가 하나에 의해 작성되고 다른 하나에 의해 읽히는 경우 프레임워크는 경계에서 캐시 플러시/무효화 호출을 삽입합니다. 애플리케이션 코드는 이에 대해 생각할 필요가 없습니다. 이는 버퍼가 단계를 교차할 때 세그먼트 수준에서 처리됩니다.

애플리케이션 코드가 이에 대해 생각해야 하는 유일한 경우는 `Mapping`을 통해 직접 CPU 읽기 또는 쓰기를 위해 `TensorBuffer`를 **매핑**할 때입니다. 프레임워크는 언매핑 시점에 적절한 무효화(읽기 매핑) 또는 플러시(쓰기 매핑)를 삽입합니다. [`MapMode`](/reference/cppapi/namespaces/simaai-neat) 및 [`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer)를 참조하십시오.

## 실제 제로 복사

일반적인 추론 파이프라인:

```
file → demux → H.264 decode → resize → preproc → MLA → postproc → app
```

제로 복사 방식을 사용하지 않으면 7번 복사가 발생합니다. 버퍼 트리플과 세그먼트를 사용하면 복사가 0번이 됩니다. 즉, 각 단계에서 `(buffer_id, paddr, vaddr)`를 전달하고 다음 단계에서는 해당 버퍼를 직접 사용합니다.

프레임워크의 플래너는 연속된 단계가 공유할 수 있도록 세그먼트를 선택하는 역할을 합니다. 인접한 두 단계의 세그먼트 요구 사항이 호환되지 않으면 플래너는 `Transfer` `ConversionKind`를 삽입하고 활성 `ConversionTraceCollector`에 기록합니다. 이 부분을 주의 깊게 확인하십시오. 런타임에 실제 바이트가 이동하는 유일한 지점입니다.

## 카메라 소스 및 적응형 메모리

실시간 카메라 프레임은 플랫폼 카메라 스택을 통해 입력되므로 메모리 유형은 설치된 커널, 드라이버 및 `libcamerasrc` 경로에 따라 달라집니다. `CameraInput`은 먼저 장치/SiMaAI 제로 복사 메모리를 요청합니다. 내부 브리지는 `GST_QUERY_ALLOCATION`을 통해 표준 풀을 제안합니다. 풀은 검증된 평면을 하나의 패키지된 SiMaAI 할당에서 할당하고 DMA-BUF로 내보냅니다. `libcamerasrc`는 해당 DMA-BUF를 ISP로 가져오고, 브리지는 다운스트림 CVU/MLA 단계에 대해 동일한 패키지된 할당을 해제합니다.

카메라 스택이 OS/libcamera 버퍼만 제공하고 `allow_cpu_fallback`가 활성화된 경우, Neat은 내부 카메라 메모리 브리지를 삽입합니다. 브리지는 각 프레임을 풀링된 SiMaAI 버퍼에 복사하고, 예상되는 메타데이터를 추가한 다음 해당 버퍼를 모델에서 관리하는 CVU 전처리 단계로 전달합니다. 이 복사는 호환성 브리지입니다. 크기 조정, 색상 변환, 정규화, 양자화 및 테셀레이션은 여전히 CVU/EV74에서 실행되어야 합니다.

MIPI 카메라 입력을 작동시키기 위해 공개 `OsToSima`, `videoconvert` 또는 `videoscale` 단계를 추가하지 마십시오. [`CameraInput`](/reference/nodes/camera-input)을 사용하고 소스 경로가 메모리 적응을 담당하도록 합니다.

## 관련 유형

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer) — 버퍼 트리플 컨테이너.
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment) — 세그먼트 핸들.
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping) — 직접 CPU 액세스를 위한 RAII 맵 핸들.
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h) — 노드가 메모리를 할당하는 방법을 정의합니다.
- [`ConversionKind::Transfer`](/reference/cppapi/files/include-pipeline-tensorconversion-h) — 세그먼트 간에 복사하는 유일한 변환 유형.

## 추가 정보

- "텐서 및 버퍼" — 디자인 심층 분석의 §0.10, §18, §19, §20.
- "TensorBuffer ABI" — 디자인 심층 분석의 §20.
