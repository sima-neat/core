---
title: "EV74 시각적 프런트엔드 노드"
description: "FeatureHistogram, GriderFast, TrackDescriptor, TrackKLT 및 MetoakDepth의 Neat Graph 사용법"
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
| `nodes::MetoakDepth` | `simor_depth_map` | 20 | I420 및 시차에서 RGB, 미터법 깊이, XYZ 생성 |

그래프 ID는 문제 진단 및 펌웨어/패키지 일치 확인에 유용합니다. 애플리케이션 코드에서는 필수가 아닙니다.

## 텐서 계약

특징 및 추적 텐서는 **논리적 배치 형태**를 사용합니다. `batch_size == B`인 경우, 흑백 이미지는 `[B,H,W]`이고, `[B*H,W]`는 아닙니다. 런타임은 모든 EV74 전송 패킹을 내부적으로 처리합니다.

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

## 6개 입력을 사용하는 Metoak 깊이

`MetoakDepth`는 `simor_depth_map`(그래프 20)을 사용하는 C++ 전용 노드입니다. 원시 SIMOR 카메라 프레임이 아니라 디코딩된 I420 평면, 원시 시차, 프레임별 보정을 입력으로 받습니다. 이 노드 앞에서 애플리케이션 또는 ROS 어댑터가 SIMOR를 언패킹하고 보정을 선택해야 합니다. Neat은 해당 어댑터를 대체하지 않습니다.

짝수 `width`를 `[8,2048]`, 짝수 `height`를 `[8,1536]`으로 설정합니다. S315의 기본 깊이 해상도는 `640x360`입니다. 배치는 1로 고정되며 앞에 배치 차원이 없습니다. 아래의 표준 경로 이름과 입력 순서를 유지하십시오. 별칭은 계약 컴파일 중 거부됩니다.

| 입력 경로 | 형식 | 형태 | 의미 |
| --- | --- | --- | --- |
| `y_src` | UInt8 | `[H,W]` | I420 Y |
| `u_src` | UInt8 | `[H/2,W/2]` | I420 U |
| `v_src` | UInt8 | `[H/2,W/2]` | I420 V |
| `disp_src` | UInt16 | `[H,W]` | 원시 시차; 고정 서브픽셀 스케일 32 |
| `bf_mm_src` | Float32 | `[1]` | 보정된 기준선 × 초점 거리(mm) |
| `proj_src` | Float32 | `[3]` | 투영 `{fx_fy,cx,cy}` |

| 출력 경로 | 형식 | 형태 | 의미 |
| --- | --- | --- | --- |
| `rgb_dst` | UInt8 | `[H,W,3]` | 인터리브 RGB |
| `depth_dst` | UInt16 | `[H,W]` | 깊이(mm); 0은 유효하지 않음 |
| `points_dst` | Float32 | `[H,W,3]` | 인터리브 XYZ(미터); NaN은 유효하지 않음 |

세 출력은 항상 함께 게시됩니다. `depth_dst`는 기본 경계 설명이며 출력 선택기가 아닙니다. 보정 BF와 초점 거리는 양의 유한 값이어야 하고 주점 좌표는 유한 값이어야 합니다.

입력 표와 일치하는 이름 지정 텐서 6개를 EV74 메모리에 준비하여 Graph를 구성합니다. 이 예제는 노드를 설정합니다. 실제 디코딩된 프레임 및 보정 텐서는 어댑터에서 제공하십시오.

```cpp
#include <neat.h>

using namespace simaai::neat;

// inputs contains the six named, decoded EV74 tensors from the table above.
Run build_metoak_depth(const TensorList& inputs) {
  Graph graph;
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.caps_override =
      "application/vnd.simaai.tensor, representation=(string)tensor-set, storage=(string)tensorbuffer";
  graph.add(nodes::Input(input));

  MetoakDepthOptions depth;
  depth.width = 640;
  depth.height = 360;
  graph.add(nodes::MetoakDepth(depth));
  graph.add(nodes::Output());

  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  return graph.build(inputs, options);
}
```

그래프 20을 포함하는 일치하는 Internals 및 EV74 펌웨어에서만 이 Graph를 실행하십시오. 아래 특징/추적 검증 명령은 다른 네 그래프를 대상으로 하며 `MetoakDepth`는 포함하지 않습니다.

## Python 표면

네 특징/추적 노드에는 C++ 옵션/팩토리 형식을 따르는 Python 바인딩이 있습니다. `MetoakDepth`에는 현재 Python 바인딩이 없습니다. 옵션 객체를 생성하고 공개 설정을 지정한 뒤 노드를 `Graph`에 추가하십시오.

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

네 특징/추적 노드는 EV 디스패치 전에 그래프의 유효성을 검증합니다. 다음의 경우 유효성 검증에 실패합니다.

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
