---
title: "EV74 비주얼 프런트엔드 노드"
description: "FeatureHistogram, GriderFast, TrackDescriptor, TrackKLT의 공개 Graph 계약"
sidebar_position: 8
---

# EV74 비주얼 프런트엔드 노드

EV74 연산은 공개 `Graph` 노드 팩터리로 사용하세요. 서로 맞는 Core, Scheduler, Runner, Kernels 및 EV74 펌웨어 패키지를 설치해야 합니다. 기존 그래프 ID는 네이티브 펌웨어 호환성 검사 수단이 아니며 애플리케이션에 디스패처 ID는 필요하지 않습니다.

품질 옵션은 기존 소스 사용 방식과 기본 출력 위치를 유지하지만 `TrackKLTOptions`의 C++ 레이아웃을 변경하므로 기존 C++ 바이너리와 ABI가 호환되지 않습니다. 일치하는 Core 버전에 맞춰 C++ 애플리케이션과 Python 확장을 다시 빌드하세요. v2 추적기 요청에는 서로 일치하는 Kernels, Runner, Scheduler, EV74 펌웨어 조합을 사용하세요. 기존 네이티브 v1 요청은 계속 지원되지만, 이것이 v2 요청과 이전 펌웨어의 호환성을 의미하지는 않습니다.

| 노드 팩터리 | 용도 |
| --- | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | 그레이스케일 히스토그램 |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | 격자에 분산된 FAST 특징점 |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | FAST 특징점과 BRIEF 기술자 |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | 피라미드 KLT 추적 및 선택적인 현재 프레임 특징 검출 |

## 텐서 계약

이미지 텐서의 논리적 형태는 `[B,H,W]`이며 `B`는 `batch_size`입니다. 배치를 이미지 높이에 합치지 마세요. 장치 입력은 별도 할당 또는 유효한 공유 저장소 뷰를 사용할 수 있습니다. CPU 복사 우회 방법은 필요하지 않습니다.

| 노드 | 입력 | 기본 공개 출력 |
| --- | --- | --- |
| `FeatureHistogram` | UInt8 `[B,H,W]` | Int32 `[B,256]` |
| `GriderFast` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]` |
| `TrackDescriptor` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]`; Int32 `[B,max_features,8]` |
| `TrackKLT` | UInt8 `[B,H,W]`; UInt8 `[B,H,W]`; Int32 `[B,num_points,2]` | Float32 `[B,num_points,2]`; Int32 `[B,num_points,1]` |

배치 항목별 특징 형식은 `[count,x0,y0,score0,...]`입니다. 처음 `count`개 레코드만 읽으세요. 사용하지 않는 용량은 특징 목록이 아닙니다. TrackDescriptor는 `descriptor_words == 8`을 요구하며 기술자 각 행은 선택한 특징의 같은 행에 대응합니다.

## 격자 선택과 경계

`grid_x`와 `grid_y`는 전체 이미지를 셀로 나눕니다. 용량이 `K`, 셀 수가 `M`이면 셀마다 `K / M`개의 자리를 받고 행 우선 순서의 처음 `K % M`개 셀은 하나 더 받습니다. 셀마다 높은 점수를 선택하고 전체 결과는 점수 내림차순, 동점은 `(y,x)` 오름차순으로 정렬합니다. 빈 셀의 자리를 다른 셀에 넘기지 않습니다. `min_px_dist`는 셀 경계를 넘어 가까운 점을 억제하며 정확히 같은 거리는 허용합니다. 억제 후 후보를 보충하지 않으므로 결과가 `max_features`보다 적을 수 있습니다.

일반 FAST는 `x=3`, `y=3`, `x=W-4`, `y=H-4`의 중심을 포함합니다. TrackDescriptor는 완전한 31×31 BRIEF 패치가 필요하므로 각 가장자리에서 15픽셀 이상 떨어져야 합니다. 따라서 3번 행에 기술자가 없는 것은 의도된 동작입니다.

## KLT 입력과 출력

`TensorList`를 **이전 이미지, 현재 이미지, 입력 점** 순서로 전달하세요. 이 직접 공개 다중 입력 경로에서 비어 있지 않은 논리적 `route.name`은 설정된 위치와 일치해야 하며 목록을 재정렬하지 않습니다. 올바른 순서의 이름 없는 텐서도 유효합니다. 입력 누락, 이름 충돌, 잘못된 형식·형태 및 접근 불가능한 범위는 후속 프레임에서도 제출 전에 거부됩니다.

호환되는 출력 텐서를 재사용할 때는 얕은 복사본을 만들고 새 공개 위치에 맞게 `route.name`만 바꾸거나 지우세요. 저장소, 오프셋 및 물리적 라우팅 메타데이터는 보존해야 합니다. 이 규칙은 내부 그래프 연결의 이름을 바꾸거나 명시적 모델 입력 경로 프로세서를 대체하지 않습니다.

TrackKLT는 기본적으로 점과 상태를 출력합니다. `detect_new_features != 0`이면 **현재** 이미지에서 검출한 특징을 추가합니다. 검출을 끄면 숨겨진 특징 출력을 할당하지 않습니다. `publish_error = true`는 기존 출력 뒤에 Float32 `[B,num_points,1]`을 추가합니다. 검출을 끄면 인덱스 2, 켜면 인덱스 3입니다. 기존 출력 위치는 바뀌지 않습니다.

### C++ 예제

두 개의 `640×480` UInt8 이미지 벡터와 `64×2` Int32 초기 좌표를 제공하세요. 아래 각 입력은 독립적인 EV74 저장소를 사용하며 반환 순서는 점, 상태, 오차입니다.

```cpp
#include <neat.h>
#include <cstdint>
#include <vector>

using namespace simaai::neat;

TensorList track_pair(const std::vector<std::uint8_t>& previous,
                     const std::vector<std::uint8_t>& current,
                     const std::vector<std::int32_t>& xy) {
  constexpr int width = 640, height = 480, count = 64;
  auto image = [&](const auto& pixels, const char* name) {
    auto tensor = Tensor::from_vector(pixels, {1, height, width}, TensorMemory::EV74);
    tensor.layout = TensorLayout::HW;
    tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H,
                            TensorAxisSemantic::W};
    tensor.route.name = name;
    return tensor;
  };
  auto points = Tensor::from_vector(xy, {1, count, 2}, TensorMemory::EV74);
  points.route.name = "input_points";
  TensorList inputs{image(previous, "prev_image"), image(current, "cur_image"), points};

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = input.max_width = width;
  input.height = input.max_height = height;
  input.depth = input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "prev_image";
  input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                        "storage=(string)tensorbuffer";

  TrackKLTOptions klt;
  klt.width = width;
  klt.height = height;
  klt.num_points = count;
  klt.win_half = 7;
  klt.max_level = 2;
  klt.max_iters = 20;
  klt.publish_error = true;

  Graph graph;
  graph.add(nodes::Input(input));
  graph.add(nodes::TrackKLT(klt));
  graph.add(nodes::Output());
  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  Run run = graph.build(inputs, options);
  TensorList outputs;
  try {
    outputs = run.run(inputs, 30000);
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  return outputs;
}
```

## 품질 제어

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `publish_error` | `false` | 최종 좌표의 평균 절대 밝기 잔차를 출력합니다. `output_error_name`의 기본값은 `output_error`입니다. |
| `error_threshold` | `-1.0` | `-1`이면 비활성화하고, 그 외에는 `[0,255]`의 유한 임곗값입니다. 잔차가 임곗값보다 크면 거부하고 같으면 허용합니다. 오차를 출력하지 않아도 필터링할 수 있습니다. |
| `min_eig_threshold` | `0.0` | `0`이면 비활성화하고, 그 외에는 유한한 양의 정규화 최소 고유값 임곗값으로 텍스처 조건을 검사합니다. |
| `require_convergence` | `false` | 최종 원본 해상도 갱신에 수렴 조건을 요구합니다. false이면 반복 상한 도달만으로 실패로 처리하지 않습니다. |

잔차는 밝기 차이이며 픽셀 변위나 신뢰도가 아닙니다. 반환 좌표에서 계산하며 `+inf`는 유효한 잔차를 계산하지 않았음을 나타냅니다. 텍스처 지표는 중앙 차분 기울기 행렬의 최소 고유값을 창 면적과 `255²`로 나눕니다. OpenCV 임곗값과 같은 척도가 아닙니다. 대표적인 보정 데이터로 임곗값을 정하고 올바른 추적의 유지율과 잘못된 추적의 거부율을 모두 검증하세요.

좌표를 사용하기 전에 상태를 확인하세요. 계산 전의 무효 초기 점은 원래 좌표와 상태 0을 유지합니다. 계산한 유한 좌표가 이미지 밖에 있거나 품질 임곗값으로 후보가 거부되면 계산 좌표와 상태 0이 유지될 수 있습니다. 기본 상태는 해법·기하 유효성이며 실제 대응의 정확성을 보장하지 않습니다.

## 움직임, 경계 및 고정 점 수

개선 위치가 현재 이미지 타일을 벗어나면 타일을 다시 중심에 맞추고 읽습니다. ±4픽셀 탐색 범위로 조용히 제한하지 않습니다. 재로딩과 반복 횟수에는 상한이 있습니다. 경계 타일은 이미지 내부 DMA만 사용하고 가장자리 픽셀을 복제합니다. 이전의 피라미드 수준별 제외 영역은 제거하지만 임의의 움직임, 가림 또는 모호한 텍스처에서 추적을 보장하지는 않습니다.

Core 초기 점은 여전히 Int32이며 결과는 Float32입니다. `num_points`는 그래프를 만들 때 고정됩니다. 사용하지 않는 자리는 `(-1,-1)` 같은 무효 좌표로 채우고 상태를 확인하세요. `(0,0)`은 유효한 가장자리 좌표이지 비활성 표시가 아닙니다. 두 이미지 피라미드는 호출마다 재구성하며 이전 프레임 피라미드 재사용을 의미하지 않습니다.

## Python 옵션

Python도 같은 옵션을 제공합니다. `Tensor.from_numpy(..., memory="ev74")`로 입력 텐서를 만들고 입력 순서 계약을 지키며 C++처럼 공개 노드를 구성하세요.

```python
import pyneat

options = pyneat.TrackKLTOptions()
options.width, options.height = 640, 480
options.num_points = 64
options.win_half, options.max_level, options.max_iters = 7, 2, 20
options.publish_error = True
options.error_threshold = -1.0
options.min_eig_threshold = 0.0
options.require_convergence = False
node = pyneat.nodes.track_klt(options)
```

## 검증

설정된 포트 이름은 비어 있지 않고 고유해야 합니다. 잘못된 크기, 점 수, 창, 피라미드 수준, 기술자 크기 및 유한하지 않거나 범위를 벗어난 품질 매개변수는 거부됩니다. 검사를 우회하거나 값을 변환하려고 텐서 형식만 바꾸지 마세요.

## 정확성 테스트 실행

DevKit에서 `CORE_TEST_DIR`를 설치된 Core 테스트 디렉터리 또는 네이티브 빌드의 테스트 디렉터리로 설정한 다음 실행하세요.

```bash
ctest --test-dir "$CORE_TEST_DIR" \
  -R '^(grider_fast|track_descriptor|track_klt)_ev_test$' --output-on-failure
```

각 등록 테스트는 해당 커널의 전체 사례를 실행합니다. GriderFast는 격자·거리와 FAST 경계, TrackDescriptor는 정확한 기술자 대응도, TrackKLT는 독립 입력, 프레임 재사용, 검증, 품질, 움직임, 경계 및 검출기를 확인합니다. 별도 빠른 프리셋은 히스토그램, 검출 없는 KLT 및 배치 거부 스모크 테스트로, 수치 검증을 대체하지 않습니다.

## Metoak 깊이 (C++)

`nodes::MetoakDepth`는 기존 `MetoakDepthOptions` API를 복원합니다.
이름이 지정된 텐서 6개를 입력받아 RGB, 깊이, XYZ를 함께 공개합니다.
이 노드는 카메라 캡처가 아닌 EV74 연산을 수행하므로 카메라 없이 합성 텐서를 사용할 수 있습니다.

위의 배치 지원 노드와 달리 Metoak은 항상 배치 1이며 선행 배치 차원이 없습니다.
`W`는 `[8,2048]`, `H`는 `[8,1536]` 범위의 짝수로 설정합니다.
표준 라우트 이름을 유지하고 아래 순서로 입력을 전달하세요.

| 텐서 | 자료형 | 형상 |
| --- | --- | --- |
| y_src | UInt8 | [H,W] |
| u_src, v_src | UInt8 | [H/2,W/2] |
| disp_src | UInt16 | [H,W] |
| bf_mm_src | Float32 | [1] |
| proj_src | Float32 | [3] |
| rgb_dst | UInt8 | [H,W,3] |
| depth_dst | UInt16 | [H,W] |
| points_dst | Float32 | [H,W,3] |

`y_src`, `u_src`, `v_src`는 개별 I420 평면입니다. 시차 배율은 32입니다.
각 프레임에 양의 유한 값인 `bf_mm_src`(기선 × 초점 거리, mm)와
`proj_src = {fx_fy, cx, cy}`를 전달합니다. `fx_fy`는 양수여야 하며 세 값 모두 유한해야 합니다.
보정값은 프레임마다 변경할 수 있습니다.

깊이는 mm 단위로 반올림하고 UInt16 상한에서 포화됩니다. XYZ는 반올림 전 깊이를 m 단위로 사용합니다.
시차가 0이면 깊이는 0이고 XYZ는 NaN입니다. 기본 출력 설명이 깊이여도 세 출력 모두 사용할 수 있습니다.

```cpp
MetoakDepthOptions depth;
depth.width = 640;
depth.height = 360;
graph.add(nodes::MetoakDepth(depth));
```

기존 `debug`, `num_buffers`, `element_name` 필드를 그대로 사용할 수 있습니다.
`num_buffers = 0`은 그래프의 기본 용량을 사용하며 양수는 이 단계의 실행 용량을 지정합니다.
복원된 공개 API는 C++이며 Python Metoak 팩토리는 추가하지 않습니다.
