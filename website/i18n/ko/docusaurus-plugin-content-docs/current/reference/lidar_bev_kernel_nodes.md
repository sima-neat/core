---
title: "LiDAR 및 BEV 커널 노드"
description: "SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan, SimaBEVPool 커널 라이브러리 연산을 그래프 노드로 실행합니다"
sidebar_position: 8
---

# LiDAR 및 BEV 커널 노드

`SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan`, `SimaBEVPool`은 PointPillars와 BEVFusion의
MLA가 아닌 단계를 실행합니다. 각 노드는 선택한 대상에서 SiMa 커널 라이브러리 연산
하나를 실행합니다. 노드는 다른 대상으로 대체 실행하지 않으며, 레이아웃을 바꾸기
위해 텐서를 복사하지도 않습니다. 커널 라이브러리가 요청을 거부하면 그래프 빌드는
커널이 제공한 이유와 함께 실패합니다.

| 노드 | 연산 | 대상 |
| --- | --- | --- |
| `SimaVoxelize` | LiDAR 점에서 장식 및 양자화된 필러 텐서로 | A65, EV74 |
| `SimaPillarScatter` | 필러 특징에서 조밀한 BEV 캔버스로 | A65, EV74 |
| `SimaBEVPoolPlan` | 카메라 보정값에서 BEV 풀링 계획으로 | A65 |
| `SimaBEVPool` | 카메라 깊이 및 컨텍스트 특징에서 BEV 격자로 | A65, EV74 |

다음은 애플리케이션 안에 넣는 그래프 구성 코드 조각입니다. `read_sweep()`는
애플리케이션에서 구현해야 하며, x, y, z, 시간 오프셋 순서로 배치한 `4*N`개의
float 값을 비어 있지 않은 벡터로 반환해야 합니다. `N`은 INT32로 표현할 수 있어야
합니다. 이 함수는 Neat API가 아닙니다. 이 코드는 그래프를 구성할 뿐, 프레임을
제출하거나 출력을 가져오지는 않습니다.

```cpp
#include <neat.h>

using namespace simaai::neat;

SimaVoxelizeOptions voxelize;
voxelize.voxel_size = {0.25f, 0.25f, 8.0f};
voxelize.point_cloud_range = {-50.0f, -50.0f, -5.0f, 50.0f, 50.0f, 3.0f};
voxelize.quantization.multiplier = 2.720007f; // 1 / the consuming MLA input's scale
voxelize.quantization.zero_point = 8;
voxelize.max_pillars = 40000;
voxelize.max_points_per_pillar = 64;
voxelize.target = SimaKernelTarget::EV74;

// One sample holds every input tensor of the node, in order.
InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

std::vector<float> xyzt = read_sweep();  // x, y, z, time offset per point
const auto n = static_cast<std::int64_t>(xyzt.size() / 4);
auto points = Tensor::from_vector(xyzt, {n, 4}, TensorMemory::EV74);
auto point_count = Tensor::from_vector(std::vector<std::int32_t>{static_cast<std::int32_t>(n)},
                                       {1}, TensorMemory::EV74);

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaVoxelize(voxelize));
graph.add(nodes::Output());
auto run = graph.build(TensorList{points, point_count});
```

Python에서도 같은 옵션 형식과 `pyneat.nodes.sima_voxelize`,
`pyneat.nodes.sima_pillar_scatter`, `pyneat.nodes.sima_bev_pool_plan`,
`pyneat.nodes.sima_bev_pool` 팩터리를 제공합니다.

## 입력 및 출력 계약

각 노드는 입력 샘플의 텐서를 아래 순서대로 읽고 출력을 아래 순서대로 게시합니다.
텐서 개수, 데이터 형식, 형태는 그래프를 빌드할 때 검사합니다.

| 노드 | 입력 | 출력 |
| --- | --- | --- |
| `SimaVoxelize` | FP32 `points` `[N,4]`(x, y, z, 시간 오프셋), INT32 `point_count` `[1]` | INT8 또는 BF16 `decorated` `[1,M,T,10]`, (row, column) 형식의 INT32 `coordinates` `[M,2]`, INT32 `count` `[1]` |
| `SimaPillarScatter` | INT8 또는 BF16 `features` `[M,C]`, INT32 `coordinates` `[M,2]`, INT32 `count` `[1]` | `features`와 같은 형식의 `canvas` `[1,H,W,C]` |
| `SimaBEVPoolPlan` | FP32 `camera2lidar`, `camera_intrinsics`, `img_aug_matrix` `[N,4,4]`, FP32 `lidar_aug_matrix` `[4,4]` | UINT8 `plan` `[P]` |
| `SimaBEVPool` | INT8 또는 BF16 `depth`, 같은 형식의 `context`, UINT8 `plan` `[P]` | INT8 또는 BF16 `bev` `[1,nZ*C,nX,nY]` |

- **SimaVoxelize**는 처음 `point_count`개의 점만 읽습니다. `max_pillars`(`M`)와
  `max_points_per_pillar`(`T`)는 고정 용량이며, 이를 넘는 점과 필러는 버려집니다.
  사용하지 않는 행에는 양자화 영점과 0 좌표가 들어갑니다.
- **SimaPillarScatter**는 처음 `count`개의 필러만 읽습니다. 필러 특징 모델이 출력하는
  `[1,M,1,C]`처럼 크기가 1인 추가 축이 있는 특징도 받습니다. `C`는 16의 배수여야
  합니다. 빈 셀에는 `zero_point`가 들어가며, 이 값은 캔버스를 읽는 모델의 입력
  양자화가 0.0에 할당하는 INT8 코드여야 합니다. INT8 캔버스 바이트는 조밀한 NHWC
  순서입니다.
- **SimaBEVPoolPlan**은 절두체(`xs`, `ys`, `ds`)와 격자(`bx`, `dx`, `nx`)를
  체크포인트의 뷰 변환에서 가져옵니다. 기본값인 `SimaBEVPoolPlanFormat::V1`은 점을
  BEV 셀별로 묶습니다. EV74 `SimaBEVPool`에는 `context_channels`와 함께
  `SimaBEVPoolPlanFormat::V2`를 사용합니다. 기본적으로 `P`는 최악의 경우 계획 크기이며,
  계획이 사용한 길이를 넘는 바이트는 쓰지 않습니다.
- **SimaBEVPool**은 각 셀의 계획 점에 대해 `depth x context`를 합하고, 그 합을
  `quantization`으로 양자화합니다. `cells`는 계획의 `nx`로, `plan_format`은 계획의
  `format`으로, INT8 입력이면 `quantization.multiplier`를
  `depth_scale * context_scale / output_scale`로 설정합니다. 기본값인
  `SimaBEVPoolStorage::ChannelsLast`에서는 입력이 카메라 모델이 출력하는 그대로
  `[N,fH,fW,D]`와 `[N,fH,fW,C]`로 들어옵니다. 출력은 채널 마지막(channels-last)
  스트라이드를 가진 논리 형태 `[1,nZ*C,nX,nY]`로
  게시되므로, 바이트는 NHWC `[1,nX,nY,nZ*C]`입니다. `SimaBEVPoolStorage::Contiguous`는
  `[N,D,fH,fW]`와 `[N,C,fH,fW]` 입력과 조밀한 출력을 사용합니다.

### 논리 축과 물리 저장 순서

`SimaBEVPool`은 논리 축 순서와 물리 바이트 순서를 구분합니다.
`TensorLayout::HWC` 태그가 있는 텐서는 논리 축 `[N,H,W,K]`를,
`TensorLayout::CHW` 태그가 있는 텐서는 논리 축 `[N,K,H,W]`를 가집니다. 노드는
이 태그를 우선하며 스트라이드가 `input_storage`와 일치하는지 검사합니다.
`ChannelsLast`에서 HWC 입력은 조밀해야 하며 CHW 입력은 채널이 마지막인
스트라이드를 가져야 합니다. `Contiguous`에서는 조밀한 CHW 입력만 허용하고
HWC 태그는 거부합니다.
같은 표준 축 순서를 `Tensor.axis_semantics`로 지정할 수도 있습니다. 두 주석을
모두 지정하면 서로 일치해야 합니다.

태그가 없는 `ChannelsLast` 입력은 호환성을 위해 조밀한 NHWC 또는 채널이 마지막인
스트라이드를 가진 비조밀 논리 NCHW로 해석합니다. 크기가 1인 차원 때문에 논리 NCHW
뷰도 조밀해지는 경우 `Tensor.layout = TensorLayout::CHW`를 명시하세요. 이 경우
스트라이드만으로 논리 축을 식별할 수 없습니다. Pool 출력은 CHW 논리 축과
`output_storage`에 맞는 스트라이드를 유지합니다. 노드는 레이아웃 변경을 위해
페이로드 바이트를 전치하거나 복사하지 않습니다.

### 양자화 메타데이터

INT8 출력은 노드가 양자화를 알 때 그 값을 `Tensor.semantic.quant`에 담습니다.
`SimaVoxelize`와 BF16 입력의 `SimaBEVPool`은 `scale = 1 / quantization.multiplier`,
두 INT8 입력이 모두 스케일을 가진 `SimaBEVPool`은
`depth_scale * context_scale / quantization.multiplier`, `SimaPillarScatter`는 특징의
양자화를 게시합니다. 그 밖의 경우에는 양자화를 게시하지 않습니다.

INT8 Scatter 입력의 채널별 양자화는 `[1,M,1,C]` 같은 모양을 포함하여 마지막 특징
축 `C`에서만 지원됩니다. 유한한 양의 스케일 `C`개와 INT8 영점 `C`개를 제공하세요.
모든 영점은 빈 셀을 채우는 Scatter의 `zero_point`와 같아야 합니다. 캔버스는 이 값을
유지하고 양자화 축을 `[1,H,W,C]`의 `3`으로 바꿉니다. 다른 양자화 축은 거부합니다.
좌표에 따라 기둥이 배치된 뒤에는 기둥 축 매개변수로 고정된 캔버스를 설명할 수
없기 때문입니다.

`SimaBEVPool`은 채널별 양자화 메타데이터를 지원하지 않으며, 조용히 제거하는 대신
거부합니다. 기존 텐서별 메타데이터와 명시적 영점 옵션의 동작은 유지됩니다. 입력
메타데이터가 없으면 알 수 없는 상태로 남습니다.

## BF16 텐서

- **SimaVoxelize**는 `output_dtype = TensorDType::BFloat16`이면 binary32 장식 값을 각각
  최근접 짝수 반올림으로 BF16으로 만들고, 모든 NaN을 `0x7FC0`으로 저장하며, 빈 슬롯을
  +0.0으로 채웁니다. `quantization`은 기본값으로 두세요. 다른 값은 거부됩니다.
- **SimaPillarScatter**는 특징의 형식을 유지합니다. BF16 특징에는 `zero_point = 0`이
  필요하므로 빈 셀에는 +0.0이 들어갑니다. BF16 캔버스의 각 셀은 `C`개 채널의 하위
  바이트 다음에 상위 바이트를 저장합니다. `semantic.tess`가 이 물리 배치를 명시합니다.
  `slice_shape = [1,1,1,C]`, `axes = [0,1,2,3]`,
  `format = "MLA_TILED_BF16_COMPACT"`이며 `storage_bytes`는 검증된 페이로드 크기입니다.
  D/H/W/C 타일은 C/D/H/W 순서로 배치되고 타일 내부는 D/H/W/C 순서입니다. BF16은
  셀별 하위/상위 바이트 평면을 사용하며 각 타일은 16바이트에 정렬됩니다.
  `copy_dense_bytes_tight()`, 형식 지정 `data_ptr()`, DLPack은 이 비조밀 배치를 거부합니다.
  `copy_payload_bytes()`는 범위가 지정된 저장 바이트를 읽습니다. `clone()`과 Owned 출력은
  타일 해제 없이 바이트와 메타데이터를 보존합니다. `SimaArgMax`처럼 배치를 이해하는
  같은 그래프의 후속 노드는 생산자의 검증된 배치를 직접 받습니다. 조밀한 입력만
  지원하는 소비자와 지원되지 않는 타일 그래프 입력은 이를 거부합니다. 전체 채널 INT8 셀 타일은 조밀한 NHWC와 바이트 단위로 같으므로
  기존 조밀 복사와 DLPack 동작을 유지합니다. 형식 이름만 있는 기존 `TessSpec` 주석의
  동작은 변하지 않습니다. 0이 아닌 `storage_bytes`는 명시적인 타일 저장 형식을 뜻합니다.
- **SimaBEVPoolPlan**은 `SimaBEVPool` 입력의 형식인 `input_dtype`을 받습니다. `V2`
  계획은 EV74 캐시를 바이트 단위로 할당하므로 BF16 계획이 더 클 수 있습니다. `V1`
  계획은 두 형식에서 같습니다.
- **SimaBEVPool**은 영점이 0인 BF16 `depth`와 `context`를 받습니다. 각 셀의 합은 계획의
  점 순서대로 binary32로 계산되므로, `V1`과 `V2` 계획은 서로 다른 BF16 바이트를
  낼 수 있습니다. BF16 `bev`는 각 합을 최근접 짝수 반올림으로 저장하고 NaN은
  `0x7FC0`으로 저장하며, 기본 `quantization`이 필요합니다. BF16 입력의 INT8 `bev`는 합을
  양자화합니다. `quantization.multiplier`를 `1 / output_scale`로 설정합니다. INT8 입력으로는
  BF16 `bev`를 만들 수 없습니다.

추가된 `TessSpec::axes`와 `TessSpec::storage_bytes` 필드에는 기본값이 있으므로 기존의 두 필드 집합체 초기화는 소스 호환성을 유지합니다. 그러나 공개 구조체의 바이너리 레이아웃은 변경됩니다. 서로 일치하는 Core 헤더와 라이브러리로 C++ 소비자와 Python 바인딩을 다시 빌드하고, 이전 레이아웃으로 빌드한 바이너리를 재사용하지 마십시오.

## 노드를 파이프라인에 연결

이 노드들은 일반적인 그래프 단계이지만, 입력은 둘 이상의 상위 단계에서 옵니다.

- PointPillars에서 `SimaPillarScatter`는 필러 특징 모델의 필러 특징과 함께 `SimaVoxelize`의
  `coordinates`와 `count`가 필요합니다.
- BEVFusion에서 `SimaBEVPool`은 카메라 모델의 `depth`와 `context`와 함께 `SimaBEVPoolPlan`의
  계획이 필요합니다.

이 텐서들을 위의 순서대로 하나의 입력 샘플로 묶은 뒤 노드에 전달합니다. 예를 들어
애플리케이션에서 두 결과를 모두 가져온 다음, 노드의 그래프에 함께 푸시합니다.

`SimaPillarScatter`는 특징 바이트를 재양자화하지 않고 복사하므로, 특징은 이미 캔버스를
읽는 모델의 입력 양자화를 사용해야 합니다. `SimaBEVPool`은 모든 카메라를 하나의
`[N,...]` 텐서로 받습니다.

## 계획과 상태

계획은 보정값과 `SimaBEVPoolPlanOptions`에만 의존합니다. `cache_plan = true`이면 보정값
바이트가 이전 프레임과 같은 프레임은 계획을 다시 계산하지 않고 이전 프레임의 계획을
복사합니다. 이 캐시는 처리 중인 프레임마다 런타임이 소유하는 커널 라이브러리
컨텍스트에 있으므로, 여러 프레임을 동시에 처리하면 적중할 수는 있지만 보장되지는
않습니다. 어느 경우든 계획 바이트는 같습니다.

`SimaBEVPool`은 프레임 사이에 상태를 유지하지 않습니다. 실행할 때마다 계획의 헤더와
내용을 검사하고, 자신의 형태와 맞지 않는 계획을 거부합니다.

## 대상별 제한

EV74는 A65 계약에 다음 제한을 추가합니다.

- `SimaBEVPool`은 채널 수가 16의 배수인 채널 마지막 `context`와 채널 마지막 `bev`가
  필요합니다. `V2` 계획에는 채널 마지막 `depth`도 필요합니다.
- `SimaBEVPoolPlan`은 A65에서만 실행됩니다.

한 대상이 지원하지 않는 요청은 그 대상에서 실패합니다. 노드가 대신 다른 대상을
선택하지 않습니다.
