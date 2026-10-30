---
title: "메모리 모델"
description: 할당 식별, FrameLayout 배치, 제로 카피 소유권 및 캐시 가시성.
sidebar_position: 3
slug: /develop-apps/advanced-concepts/memory_model
---

# 메모리 모델

Core가 저장 공간을 계획하고 Scheduler가 Runner를 통해 풀을 준비합니다. Runner는 할당 식별, DMA-BUF 매핑 및 CPU/장치 접근을 관리합니다.

## 할당과 뷰

각 단계에는 불변 `FrameLayout` 하나가 있습니다. 여기에는 `FrameAllocation` 목록, 할당 내부 범위를 나타내는 `FrameSpan`, 입력 경로 및 출력 소유권이 포함됩니다. 범위는 할당, 오프셋 및 예약 용량을 지정합니다. 논리 텐서의 형상, 스트라이드, 게시 바이트 수는 별도입니다. MLA 하드웨어 패딩은 예약 용량 안에 들어가야 합니다.

뷰는 루트 캐리어를 공유합니다. 할당 인덱스는 버퍼 그룹에 속합니다. 서로 다른 그룹의 오프셋이 모두 0이어도 버퍼가 겹치지는 않습니다. 가져온 저장 공간은 익스포터, 검증된 오프셋, 평면 형상 및 생산자 대여를 유지합니다.

저장소가 연결된 조밀한 수치형 `Tensor`에서 모양 `{}`는 0차원 스칼라를 나타냅니다. 조밀한 크기는 `dtype` 원소 하나의 크기입니다. 이는 `{0,256}`처럼 크기가 0인 차원을 포함하여 조밀한 크기가 0이 되는 모양과 다릅니다. 빈틈없는 바이트 복사와 `clone()`은 뷰 오프셋과 저장소 경계를 확인하면서 스칼라 값을 보존합니다. 저장소가 연결되지 않은 기본 Tensor의 `dense_bytes_tight()`는 0을 반환하며, 빈틈없는 복사와 복제는 거부됩니다. 명시적으로 인코딩, 바이트 스트림, 테셀레이션 또는 텍스트로 표시된 전달 버퍼는 기존의 전달 바이트 의미를 유지합니다. 공개 `map()`이 반환하는 바이트 범위는 바뀌지 않습니다.

## 배치

Core는 경로와 실행 대상을 결정하고 검증된 레시피, 캐리어 및 소유권을 해석한 뒤 접근과 수명 근거에 따라 패킹합니다. 단계 레이아웃과 게시 정보는 한 번만 확정합니다.

- EV74가 접근하면 CMA가 필요합니다.
- 소유한 MLA/A65 전용 캐리어는 기본적으로 DMS를 사용합니다.
- 조건을 충족하는 새 A65 공개 종단은 HostCpu 저장 공간을 사용할 수 있습니다.
- 가져온 저장 공간의 배치는 고정됩니다. 팬아웃은 모든 소비자를 포함하며 뷰는 캐리어 정책을 유지합니다.

`AllocationPolicy::PreferDmsThenCma`는 해당 소유 DMS 할당의 용량이 부족할 때 CMA로 폴백하도록 명시적으로 허용합니다. 일반 Linux 메모리로의 폴백은 허용하지 않습니다. Queue 경계, 독립 모델 호출 및 유지된 출력은 소유권과 수명 제약을 유지합니다.

EV74 Preproc → MLA → A65 BoxDecode에서는 추론 입력은 CMA를 사용하고 MLA 출력은 별도 DMS 할당을 사용합니다. 실제 배치는 모든 소비자의 확정된 실행 대상을 따릅니다.

## 소유권과 CPU 매핑

제로 카피 출력은 애플리케이션이 보유하는 동안 저장 공간과 생산자 대여를 유지합니다. Scheduler와 Runner는 장치의 확정된 완료까지 입력을 유지합니다. 시간 초과나 취소만으로는 재사용할 수 없습니다.

[`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer)을 적절한 [`MapMode`](/reference/cppapi/namespaces/simaai-neat)로 사용합니다. `Mapping`은 닫힐 때까지 검증된 접근과 저장 공간을 유지합니다. 호환되는 읽기는 공유할 수 있지만 쓰기는 배타적입니다. 필요한 캐시 관리는 CPU 읽기 전과 쓰기 후에 수행하며 인접 캐시 라인의 바이트도 보존합니다. 동기화에 실패한 저장 공간은 격리됩니다.

소유한 캐시형 CMA와 DMS는 허용된 SimaHeap 범위 API를 사용합니다. 가져온 버퍼는 익스포터 계약을 유지합니다. 장치만 사용하는 후속 단계는 CPU 읽기 에포크를 열지 않습니다. CPU 접근을 노드 이름이나 그래프 위치에서 추측하지 않습니다. 내부 CPU 배치는 모든 자식 접근이 종료될 때까지 부모를 유지합니다.

메타데이터 엔벌로프는 페이로드를 복사하지 않고 메모리를 공유할 수 있습니다. 하지만 자료형 변환, 명시적 전송 및 Owned 출력은 바이트를 복사할 수 있습니다. 제로 카피는 모든 단계가 제자리에서 실행된다는 뜻이 아닙니다.

## 카메라 버퍼

[`CameraInput`](/reference/nodes/camera-input)은 캡처한 DMA-BUF 레이아웃을 검증하고 하위 소비자가 사용하는 동안 카메라 부모 버퍼를 유지합니다. 호환 버퍼는 저장 공간과 평면 형상을 유지합니다. 레이아웃 조정이 필요하고 `allow_cpu_fallback`이 활성화된 경우 브리지는 명시적인 CMA 복사본을 만듭니다. 소스 조정과 모델 전처리는 별개입니다.

## 관련 타입

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer): 저장 공간과 뷰.
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment): 유지된 할당 범위.
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping): 범위가 지정된 CPU 접근.
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h): 노드가 선언한 메모리 요구 사항.
