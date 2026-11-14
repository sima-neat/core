---
title: "시맨틱 세그멘테이션 디코딩"
description: "ResizeArgMax 또는 Model 시맨틱 세그멘테이션 옵션으로 클래스별 점수를 EV74에서 크기 조정된 클래스 맵으로 변환합니다"
sidebar_position: 9
---

# 시맨틱 세그멘테이션 디코딩

`ResizeArgMax`는 세그멘테이션 모델의 클래스별 점수를 원하는 크기의 클래스 맵으로
변환합니다. 출력 픽셀마다 모든 클래스 점수를 쌍선형 보간하고 점수가 가장 큰 클래스를
반환합니다. 이는 SegFormer와 유사한 모델이 MLA 이후에 필요로 하는 디코딩으로,
`interpolate(mode="bilinear")`로 목표 크기까지 보간한 다음 클래스에 대해 `argmax`를
구합니다. Hugging Face `SegformerImageProcessor.post_process_semantic_segmentation`에
`target_sizes`를 지정한 결과와 같은 클래스 맵을 만들며, 크기 조정된 점수는 저장하지 않습니다.

디코딩은 EV74에서 실행됩니다. A65 구현이나 폴백은 없습니다.

## 모델 출력 디코딩

`Model::Options::semantic_segmentation`을 설정하면 모델의 dequantize 또는 cast 후처리가 이
디코딩으로 대체됩니다. 노드는 저장 채널이 패딩된 MLA 출력을 포함해 MLA 점수를 그
자리에서 읽으며, 모델은 점수 대신 클래스 맵 하나를 반환합니다.

```cpp
#include <neat.h>

using namespace simaai::neat;

Model::Options options;
options.semantic_segmentation.enable = true;
options.semantic_segmentation.output_width = 1280; // camera frame size
options.semantic_segmentation.output_height = 720;
Model model("segformer_b0_ade_512_mpk.tar.gz", options);
```

```python
import pyneat

options = pyneat.ModelOptions()
options.semantic_segmentation.enable = True
options.semantic_segmentation.output_width = 1280  # camera frame size
options.semantic_segmentation.output_height = 720
model = pyneat.Model("segformer_b0_ade_512_mpk.tar.gz", options)
```

이 옵션은 모델에 필요한 전처리 옵션과 함께 사용하세요. 모델 출력은 단일 클래스 맵
텐서 `[N,H,W]`이며, `model.info().selection.selected_post_kind`는 `"resize_argmax"`입니다.

- `output_width`와 `output_height`를 모두 0으로 두면 모델 입력 크기를 사용합니다.
  원본의 모든 픽셀에 클래스를 지정하려면 둘 다 원본 프레임 크기로 설정하세요.
- 기본값 `align_corners = false`는 PyTorch 및 Hugging Face와 같이 반 픽셀 중심을
  사용합니다. 모서리 정렬 보간으로 학습한 모델에서만 `true`로 설정하세요.
- 모델의 후처리는 dequantize 또는 cast 단계로만 구성되어야 하며 `decode_type`은
  설정하지 않아야 합니다. 그렇지 않으면 `Model` 생성자가 `std::invalid_argument`
  (Python에서는 `ValueError`)를 발생시킵니다.

## 그래프에서 노드 사용

C++의 `nodes::ResizeArgMax(...)`와 Python의 `pyneat.nodes.resize_argmax(options)`는
아래 계약에 맞는 모든 텐서에 같은 디코딩을 실행합니다. 다음 조각은 점수 텐서 하나,
`scores`를 512 x 512 클래스 맵으로 디코딩합니다.

```cpp
InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::ResizeArgMax({.output_width = 512, .output_height = 512}));
graph.add(nodes::Output());
auto run = graph.build(TensorList{scores});
TensorList class_map = run.run(TensorList{scores}, 1000);
```

## 입력 및 출력 계약

| 포트 | 텐서 |
| --- | --- |
| `scores` | `[h,w,C]` 또는 `[N,h,w,C]`, 클래스는 마지막 축. INT8, UINT8, BF16 또는 FP32. 밀집이거나 픽셀 피치가 `C`보다 큰 스트라이드 |
| `class_map` | `[H,W]` 또는 `[N,H,W]`, 밀집. `C`가 256 이하이면 UINT8, 그렇지 않으면 INT32 |

- 정수 점수는 역양자화 없이 비교합니다. 하나의 양수 스케일이 모든 클래스에 적용되면
  역양자화한 점수와 같은 클래스가 됩니다. 축별 양자화가 있거나 스케일이 양수가 아닌
  점수는 그래프를 빌드할 때 거부됩니다.
- 점수가 같으면 가장 작은 클래스 인덱스를 선택합니다. BF16 및 FP32 점수에서는 보간된
  NaN이 우선하며 첫 번째 NaN 클래스가 선택됩니다.
- 패딩된 MLA 출력의 추가 채널과 같은 저장 패딩 채널은 클래스로 읽지 않습니다.
- MLA 타일 점수는 거부됩니다. 먼저 디테셀레이션하세요. 논리적 CHW 또는 NCHW 축을
  가진 점수도 거부됩니다. 클래스는 마지막 축에 있어야 합니다.
- DevKit은 이 연산을 포함하는 EV74 펌웨어로 실행되어야 합니다. 이전 펌웨어에서는 첫
  프레임을 제출하기 전에 실패합니다.
