---
title: 동적 그리드로 특징 샘플링
description: Neat 그래프에서 A65 NEON GridSample2D 실행
sidebar_position: 9
---

# 동적 그리드로 특징 샘플링

`SimaGridSample2D`는 A65에서 쌍선형 샘플링을 수행합니다. 각 입력 샘플에는 특징 `[N,H,W,C]`(FP32 또는 BF16), 이어서 정규화된 `(x,y)` 좌표의 그리드 `[N,Ho,Wo,2]`(FP32)가 들어갑니다. 출력은 특징과 같은 자료형이며 채널이 마지막인 `sampled` `[N,Ho,Wo,C]`입니다. 모든 차원은 양수여야 합니다.

## 빌드 및 실행

`grid_sample_2d`가 포함된 서로 대응하는 Kernels, Runner, Scheduler 및 Core 패키지 세트를 사용하세요. 이 C++ 코드는 `SimaNeat::sima_neat`에 링크한 애플리케이션 내부에서 실행합니다. 그래프를 빌드하기 전에 두 벡터에 연속 NHWC 특징 값과 NHW2 그리드 값을 채우세요.

```cpp
#include <neat.h>
using namespace simaai::neat;

std::vector<float> features(1 * 16 * 16 * 16, 1.0f);
std::vector<float> grid(1 * 32 * 8 * 2, 0.0f);
auto feature = Tensor::from_vector(features, {1,16,16,16}, TensorMemory::CPU);
feature.layout = TensorLayout::HWC;
auto coordinates = Tensor::from_vector(grid, {1,32,8,2}, TensorMemory::CPU);

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::SystemMemory;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";
SimaGridSample2DOptions sampling;
sampling.padding = SimaGridSample2DPadding::Zeros;
sampling.align_corners = false;
Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaGridSample2D(sampling));
graph.add(nodes::Output());
auto run = graph.build(TensorList{feature, coordinates});
auto output = run.run(TensorList{feature, coordinates}, 30000);
run.close();
```

형상과 자료형을 고정하고 제출마다 그리드 값을 바꿀 수 있습니다. 여러 처리 중 샘플에는 일반 비동기 push/pull API를 사용하세요. 보유한 제로 카피 출력은 생산자 버퍼 대여를 유지하므로 풀을 진행하려면 해제해야 합니다. 소유 출력은 실행 객체와 독립적으로 유효합니다.

## 샘플링 옵션 및 정밀도

`SimaGridSample2DPadding::Zeros`는 유효하지 않은 특징 탭을 0으로 대체합니다. `SimaGridSample2DPadding::Border`는 좌표를 가장자리 픽셀 중심으로 제한합니다. `align_corners=false`는 반 픽셀 중심을 사용하고 true는 정규화 좌표 -1/+1을 모서리 픽셀 중심에 매핑합니다. NaN 그리드 성분은 정규화 좌표 -1을 뜻합니다. 무한대 성분은 0 패딩에서 샘플을 0으로 만들고 경계 패딩에서는 가장자리를 선택합니다.

좌표, 가중치 및 누산은 FP32를 사용합니다. 각 합은 첫 곱셈에 이어 세 번의 명시적 융합 곱셈-덧셈으로 계산합니다. BF16 특징은 정확히 확장하고 최종 출력에서 최근접 짝수로 한 번 반올림합니다. 이는 그리드 값이나 중간 좌표를 BF16으로 반올림하는 MLA 구현과 정밀도가 다릅니다.

NCHW, 타일 저장, 비밀집 뷰 및 비항등 양자화는 이 노드 전에 명시적으로 변환해야 합니다. 최근접, 3차 및 반사 모드는 지원하지 않습니다. 이 노드를 추가해도 컴파일된 MLA 모델 내부의 GridSample 연산은 교체되지 않습니다. 성능 비교에는 같은 저장 입력과 측정 범위를 사용하세요. 그리드 초기화를 알 수 없는 ELF 프로파일링의 그리드 출처는 미확인 상태입니다.