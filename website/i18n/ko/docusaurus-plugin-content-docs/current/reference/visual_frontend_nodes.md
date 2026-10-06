---
title: "EV74 시각적 프런트엔드 노드"
description: "FeatureHistogram, GriderFast, TrackDescriptor 및 TrackKLT에 대한 고객 맞춤형 Neat 그래프 사용법"
sidebar_position: 8
---

# EV74 시각적 프런트엔드 노드

Neat은 EV74 시각적 프런트엔드 그래프를 일반 `Graph` 노드로 노출합니다. 공개 노드 팩토리와 옵션 구조체를 사용하고, 애플리케이션 코드에서 `processcvu`, ConfigManager 또는 디스패처 API를 직접 호출하지 마십시오.

| 노드 팩토리 | 그래프 이름 | 그래프 ID | 목적 |
| --- | --- | ---: | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | `feature_histogram` | 235 | 흑백 이미지 히스토그램 |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | `grider_fast` | 236 | 그리드 방식으로 분산된 FAST 특징 |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | `track_descriptor` | 237 | FAST 특징과 디스크립터 |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | `track_klt` | 238 | 피라미드 KLT 추적, 선택적으로 감지된 대체 특징 포함 |

그래프 ID는 문제 진단 및 펌웨어/패키지 일치 확인에 유용합니다. 애플리케이션 코드에서는 필수가 아닙니다.

## 텐서 계약

모든 텐서는 **논리적 배치 형태**를 사용합니다. `batch_size == B`인 경우, 흑백 이미지는 `[B,H,W]`이고, `[B*H,W]`는 아닙니다. 런타임은 모든 EV74 전송 패킹을 내부적으로 처리합니다.

| 노드 | 입력 | 공개 출력 |
| --- | --- | --- |
| `FeatureHistogram` | `input_image`: UInt8 `[B,H,W]` | `output_hist`: Int32 `[B,256]` |
| `GriderFast` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]` |
| `TrackDescriptor` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]`; `output_descriptors`: Int32 `[B,max_features,8]` |
| `TrackKLT` | `prev_image`: UInt8 `[B,H,W]`; `cur_image`: UInt8 `[B,H,W]`; `input_points`: Int32 `[B,num_points,2]` | `output_points`: Float32 `[B,num_points,2]`; `output_status`: Int32 `[B,num_points,1]`; 그리고 `output_features`: Int32 `[B,1 + max_features*3]` (단, `detect_new_features != 0`인 경우에만) |

특성 목록 텐서는 이 배치별 레이아웃을 사용합니다.

```text
[count, x0, y0, score0, x1, y1, score1, ...]
```

현재 디스크립터 그래프는 `descriptor_words == 8`을 필요로 합니다. 이를 변경하는 것은 EV74 ABI 변경 사항이며, 배포 전에 거부됩니다.

## C++ 빠른 시작

```cpp
#include <neat.h>

#include <cstdint>
#include <vector>

using namespace simaai::neat;

Tensor make_gray_batch(int width, int height, int batch) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * batch);
  // Fill pixels in batch-major order: b*height*width + y*width + x.
  auto tensor = Tensor::from_vector(pixels, {batch, height, width}, TensorMemory::EV74);
  tensor.layout = TensorLayout::HW;
  tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H, TensorAxisSemantic::W};
  tensor.route.name = "input_image";
  tensor.route.segment_name = "input_image";
  return tensor;
}

int main() {
  constexpr int width = 320;
  constexpr int height = 240;
  constexpr int batch = 2;

  Graph graph;

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = width;
  input.height = height;
  input.depth = 1;
  input.max_width = width;
  input.max_height = height * batch; // transport capacity; public tensor remains [B,H,W]
  input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "input_image";

  graph.add(nodes::Input(input));

  GriderFastOptions fast;
  fast.width = width;
  fast.height = height;
  fast.batch_size = batch;
  fast.max_features = 64;
  fast.threshold = 30;
  graph.add(nodes::GriderFast(fast));

  graph.add(nodes::Output());

  RunOptions run_opt;
  run_opt.output_memory = OutputMemory::Owned;

  Tensor image = make_gray_batch(width, height, batch);
  Run run = graph.build({image}, run_opt);
  TensorList outputs = run.run({image}, /*timeout_ms=*/30000);
  run.close();
}
```

## 세 개의 입력값을 받는 KLT

`TrackKLT`는 텐서 세트(이전 이미지, 현재 이미지 및 입력 포인트)를 사용합니다. 옵션 필드와 일치하도록 경로 이름을 지정하십시오.

```cpp
TrackKLTOptions klt;
klt.width = 320;
klt.height = 240;
klt.batch_size = 2;
klt.num_points = 32;
klt.max_features = 64;
klt.detect_new_features = 1; // publish output_features as the third output

graph.add(nodes::TrackKLT(klt));
```

`detect_new_features == 1` 실행 시 예상되는 공개 출력:

```text
output_points   Float32 [2,32,2]
output_status   Int32   [2,32,1]
output_features Int32   [2,193]
```

`detect_new_features == 0`가 활성화되면, Neat은 `output_points`와 `output_status`만 게시하며, EV에서 보이는 기능 버퍼는 내부 런타임 할당 상태로 유지됩니다.

## Python 표면

Python API는 C++ 옵션/팩토리 스타일을 반영하며, 의도적으로 계층 구조를 따릅니다. 옵션 객체를 생성하고, 공개 구성을 설정한 다음, 노드를 `Graph`에 추가합니다.

```python
import numpy as np
import pyneat

width, height, batch = 320, 240, 2

opt = pyneat.GriderFastOptions()
opt.width = width
opt.height = height
opt.batch_size = batch
opt.max_features = 64
print(opt.summary())

graph = pyneat.Graph()
input_opt = pyneat.InputOptions()
input_opt.payload_type = pyneat.PayloadType.Tensor
input_opt.format = pyneat.Format.UINT8
input_opt.width = width
input_opt.height = height
input_opt.max_width = width
input_opt.max_height = height * batch
input_opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
input_opt.buffer_name = "input_image"

graph.add(pyneat.nodes.input(input_opt))
graph.add(pyneat.nodes.grider_fast(opt))
graph.add(pyneat.nodes.output())

image_np = np.zeros((batch, height, width), dtype=np.uint8)
image = pyneat.Tensor.from_numpy(image_np, memory="ev74")
image.layout = pyneat.TensorLayout.HW
# If setting route metadata from Python in a custom app, keep it aligned with
# the option names used above.
```

## 안전 점검

이 노드들은 EV 배차 전에 그래프의 유효성을 검증합니다. 다음의 경우 유효성 검증에 실패합니다.

- 0 이하의 차원 또는 개수;
- 지원되지 않는 배치 크기입니다.
- `[0,255]` 범위를 벗어나는 임계값
- 중복되거나 비어 있는 텐서 이름
- `TrackDescriptorOptions.descriptor_words != 8`;
- 잘못된 KLT 창, 레벨 및 감지 모드 값입니다.
- 사전 배포 협상 중에 런타임 텐서의 크기가 부족합니다.

이는 불법 버퍼가 EV74에 문제를 일으킬 수 있기 때문에 중요합니다. 검증 실패는 호스트 측 오류로 처리하고 Node 계약 경로를 우회하지 않도록 하십시오.

## 빠른 검증 명령어

빠른 고객 맞춤형 DevKit 게이트는 다음과 같습니다.

```bash
ctest --test-dir /workspace/core_graph_changes/build/tests \
  -R visual_frontend_ --output-on-failure
```

실행 중:

- `320x240`, `batch_size=2`, `detect_new_features=1`를 사용하여 생성된 4개의 시각적 그래프.
- 특정 KLT에 대한 탐지 방지 ABI 검사;
- 불법적인 일괄 입력을 확인하는 사전 배송 검사 실패 시 발생하는 오류입니다.
  EV74 이전에 거부되었습니다.
