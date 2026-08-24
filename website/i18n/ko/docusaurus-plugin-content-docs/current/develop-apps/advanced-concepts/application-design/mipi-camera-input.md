---
title: "MIPI 카메라를 사용하세요."
description: "카메라 입력을 지원하는 MIPI 카메라를 Neat 그래프에 추가하고, 모델 전처리, MLA 추론, 선택 사항인 EV74 BoxDecode, 그리고 출력 추출 기능을 활용합니다."
sidebar_position: 4
slug: /develop-apps/advanced-concepts/mipi-camera-input
---

# MIPI 카메라를 사용하세요.

Modalix DevKit의 MIPI 카메라에서 직접 프레임을 읽어와야 하는 경우 `CameraInput`을 사용합니다. `CameraInput`은 검증된 libcamera 스트림과 가속기 우선 모델 그래프 사이의 Neat 경계입니다.

```text
CameraInput -> model-managed CVU preproc -> MLA -> Output
CameraInput -> model-managed CVU preproc -> MLA -> EV74 BoxDecode -> Output
```

`appsrc`는 없습니다. 사용자 코드에 `ostosima`는 없습니다. 사용자가 직접 추가하지 않는 한 프로덕션 경로에 CPU 기반의 `videoconvert` 또는 `videoscale`는 없습니다.

## 두 개의 게이트: 먼저 부팅을 하고, 그다음에 Neat를 실행합니다.

MIPI CSI-2는 카메라 인터페이스입니다. 모든 센서를 즉시 사용할 수 있도록 하지는 않습니다. 작동하는 카메라 경로를 위해서는 적절한 보드 오버레이, 센서 드라이버, libcamera 파이프라인, ISP 동작 및 기능이 필요합니다.

MIPI 카메라 작업을 두 단계로 나누어 생각하십시오.

1. **보드 초기화:** Modalix DevKit을 사용하면 센서와 libcamera가 요청된 모드를 스트리밍할 수 있는지 확인할 수 있습니다.
2. **Neat 그래프 초기화:** `CameraInput`은 해당 프레임을 모델에서 관리하는 CVU/MLA 단계로 전달합니다.

Neat는 게이트 2에서 시작합니다. `.dtbo` 오버레이를 선택하거나, 센서 드라이버를 로드하거나, ISP를 조정하지 않습니다. 오버레이 설정, 지원되는 오버레이 이름 및 `cam` 검증에 대해서는 [Modalix DevKit MIPI 카메라 인터페이스 가이드](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces)를 참조하십시오.

## Neat에서 지원하는 기능은 무엇인가요?

`CameraInput`은 플랫폼 카메라 스택에서 이미 활성화된 카메라를 지원합니다.

- 카메라는 전원이 꺼진 상태에서 Modalix DevKit MIPI 포트에 연결됩니다.
- 올바른 보드 오버레이가 활성화되어 있습니다.
- 커널 드라이버와 libcamera 파이프라인은 카메라를 노출합니다.
- `libcamerasrc`는 요청된 `video/x-raw` 형식의 기능을 협상할 수 있으며, 일반적으로 `NV12` 형식을 사용합니다.
- 캡션은 Neat에서 구성한 모델 전처리 방식과 일치합니다.

만약 그러한 조건이 아직 충족되지 않았다면, 먼저 카메라 스택을 수정하세요. Neat 그래프는 정확할 수 있지만, 연결되지 않은 센서를 스트림으로 변환할 수는 없습니다.

## 카메라 스트림을 확인합니다.

그래프를 빌드하기 전에 libcamera/GStreamer 레이어에서 카메라를 확인합니다. 이 레이어가 제대로 작동하지 않으면 Neat 그래프로 문제를 해결할 수 없습니다.

DevKit에서 `libcamerasrc`가 존재하는지 확인합니다.

<ShellCommand prompt="devkit">
gst-inspect-1.0 libcamerasrc
</ShellCommand>

`cam`이 사용 가능한 경우, 카메라 목록과 검사 모드를 나열합니다.

<ShellCommand prompt="devkit">
cam -l
cam -c 1 -I
</ShellCommand>

그런 다음 Neat에 요청할 정확한 대문자를 사용해 보세요.

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

시각적인 간단한 테스트를 위해 몇 개의 프레임을 JPEG 형식으로 인코딩합니다.

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! videoconvert ! jpegenc ! \
  multifilesink location=/tmp/mipi-frame-%03d.jpg
</ShellCommand>

`videoconvert`와 `jpegenc`은 일회성 디버그 검사에 적합합니다. 처리량이 중요한 경우 모델 파이프라인에서 제외하십시오.

## 원시 MLA 연기 그래프를 생성합니다.

먼저 원본 MLA 경로를 사용하세요. 이를 통해 카메라 프레임이 모델별 후처리를 추가하기 전에 CVU 전처리 및 MLA 추론 단계에 도달하는지 확인할 수 있습니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::CameraInputOptions camera;
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.format = "NV12";
camera.buffer_name = "camera0";
camera.allow_cpu_fallback = true;

neat::Model::Options model_options;
model_options.preprocess.kind = neat::InputKind::Image;
model_options.preprocess.input_max_width = static_cast<int>(camera.width);
model_options.preprocess.input_max_height = static_cast<int>(camera.height);
model_options.preprocess.input_max_depth = 3;
model_options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
model_options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
model_options.preprocess.resize.enable = neat::AutoFlag::On;
model_options.preprocess.resize.width = 640;
model_options.preprocess.resize.height = 640;
model_options.preprocess.resize.mode = neat::ResizeMode::Letterbox;
model_options.preprocess.resize.pad_value = 114;
model_options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
model_options.advanced_execution.preprocess_target = "EV74";
model_options.inference_terminal.mla_only = true;

neat::Model model("/models/yolo.tar.gz", model_options);

neat::Model::RouteOptions route;
route.include_input = false;
route.include_output = true;
route.upstream_name = camera.buffer_name;
route.buffer_name = camera.buffer_name;
route.name_suffix = "_camera0";
route.advanced_execution.preprocess_target = "EV74";

neat::Graph graph("camera_mla_smoke");
graph.add(neat::nodes::CameraInput(camera));
graph.add(model.graph(route));

neat::Run run = graph.build();
std::optional<neat::Sample> output = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.CameraInputOptions()
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.format = "NV12"
camera.buffer_name = "camera0"
camera.allow_cpu_fallback = True

model_options = pyneat.ModelOptions()
model_options.preprocess.kind = pyneat.InputKind.Image
model_options.preprocess.input_max_width = int(camera.width)
model_options.preprocess.input_max_height = int(camera.height)
model_options.preprocess.input_max_depth = 3
model_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
model_options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
model_options.preprocess.resize.enable = pyneat.AutoFlag.On
model_options.preprocess.resize.width = 640
model_options.preprocess.resize.height = 640
model_options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
model_options.preprocess.resize.pad_value = 114
model_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
model_options.advanced_execution.preprocess_target = "EV74"
model_options.inference_terminal.mla_only = True

model = pyneat.Model("/models/yolo.tar.gz", model_options)

route = pyneat.ModelRouteOptions()
route.include_input = False
route.include_output = True
route.upstream_name = camera.buffer_name
route.buffer_name = camera.buffer_name
route.name_suffix = "_camera0"
route.advanced_execution.preprocess_target = "EV74"

graph = pyneat.Graph("camera_mla_smoke")
graph.add(pyneat.nodes.camera_input(camera))
graph.add(model.graph(route))

run = graph.build()
output = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

`inference_terminal.mla_only = true`는 의도적인 설정입니다. 연기 경로를 `CameraInput -> CVU preproc -> MLA -> Output`에 유지하므로, 객체 감지 디코딩을 디버깅하기 전에 카메라 및 추론 움직임을 디버깅할 수 있습니다.

## 모델에 필요한 경우 EV74 BoxDecode를 추가합니다.

YOLO 스타일의 객체 감지 모델의 경우, BoxDecode를 명시적으로 활성화하고 후처리 작업을 EV74에서 수행하도록 설정합니다. 디코딩 토큰은 MPK의 출력 계약과 일치해야 합니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
model_options.inference_terminal.mla_only = false;
model_options.decode_type = neat::BoxDecodeType::YoloV9Seg;
model_options.advanced_execution.postprocess_target = "EV74";
model_options.score_threshold = 0.25f;
model_options.nms_iou_threshold = 0.45f;
model_options.top_k = 100;

route.advanced_execution.postprocess_target = "EV74";
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
model_options.inference_terminal.mla_only = False
model_options.decode_type = pyneat.BoxDecodeType.YoloV9Seg
model_options.advanced_execution.postprocess_target = "EV74"
model_options.score_threshold = 0.25
model_options.nms_iou_threshold = 0.45
model_options.top_k = 100

route.advanced_execution.postprocess_target = "EV74"
```

</CodeTab>
</CodeTabs>

`decode_type`은 설정하지 않은 상태로 두고, 원시 MLA 텐서를 사용하려는 경우 `mla_only = true`를 유지합니다. `decode_type`은 모델 경로에서 디코딩된 감지 결과 또는 분할 데이터를 출력해야 할 때만 설정합니다.

## 적절한 메모리 모드를 선택하세요.

`CameraInput`에는 두 가지 모드가 있습니다.

| 모드 | 사용 시: | 행동 |
| --- | --- | --- |
| 엄격한 제로 복사 | 귀하의 `libcamerasrc`는 `external-buffer-mode`를 노출하며, 메모리 라이브러리는 DMA-BUF 내보내기를 지원합니다. | Neat은 데이터를 직접 캡처하여 하위 DMA-BUF 풀에 저장해야 하며, 소스에서 해당 데이터를 제공할 수 없는 경우 오류가 발생합니다. |
| 적응형 대체 방식(명시적 선택) | 장치 버퍼를 내보낼 수 없는 카메라 스택과 호환되어야 합니다. | Neat는 OS/libcamera 버퍼를 받아들여, 이를 CVU/MLA 핸드오프를 위해 풀링된 SiMaAI 메모리에 복사하고, 소스에서 이미 해당 버퍼를 제공하는 경우 SiMaAI 버퍼를 그대로 전달합니다. |

엄격한 제로 복사 방식이 프레임워크의 기본 설정입니다. 이는 일반적인 `external-buffer-mode` 속성을 노출하는 `libcamerasrc`와 DMA-BUF 내보내기를 지원하는 메모리 라이브러리를 모두 필요로 합니다. `camera.allow_cpu_fallback = true`는 명시적인 호환성 옵션으로만 설정하십시오. 폴백 복사는 가속 파이프라인으로 연결되는 다리 역할을 하며, CPU 색상 변환 또는 스케일링을 주요 경로에 추가할 수 있는 권한이 아닙니다.

두 가지 모드 모두에서 Neat는 카메라 캡처 후 즉시, 그리고 모든 큐 이전에 자체 개인 메모리 브리지를 배치합니다. 이 브리지는 `GST_QUERY_ALLOCATION`을 통해 표준 버퍼 풀을 제안합니다. 해당 풀은 검증된 평면 레이아웃을 하나의 패킹된 SiMaAI 할당에서 할당하고 각 평면에 대해 하나의 DMA-BUF를 내보냅니다. `libcamerasrc`는 이러한 DMA-BUF를 ISP 캡처 큐로 가져옵니다. 캡처 후 브리지는 동일한 패킹된 할당을 풀어서 다운스트림 처리를 위해 사용합니다. 이것은 일반적인 경로이며, 폴백 복사 경로는 아닙니다.

애플리케이션은 ISP 출력 보존 정책을 소유합니다. `nodes::CameraInputWithCaptureBuffers()` 또는 `pyneat.nodes.camera_input()`의 `capture_buffer_count` 인수를 `0`으로 설정하여 카메라 기본값을 사용하거나, 시간적 인코더 또는 비동기 ML 그래프가 프레임을 더 오래 유지할 때 더 큰 최소값을 요청합니다. 활성 카메라 파이프라인은 자체 제한을 검증합니다. Neat의 공급자는 최대 128까지 지원합니다. 이 수는 개인 CSI-to-ISP RAW 전송 링 및 Neat의 활성 GStreamer 큐만 제어하는 `queue_depth`와는 별개입니다. 현재 프레임이 완전성보다 더 중요할 때는 누수 큐를 사용하고, 모든 프레임을 보존해야 할 때는 다운스트림 역압을 사용합니다. 호환성 복사 모드에서는 브리지의 개인 풀이 필요에 따라 증가하므로 누수 큐가 오래된 프레임을 삭제하기 전에 차단되지 않습니다.

## CVU/EV74에 대한 전처리 작업을 계속 진행하세요.

모델 파이프라인의 경우, 모델에서 관리하는 전처리 방식을 사용하는 것이 좋습니다.

- 크기 조정, 색상 변환, 정규화, 양자화 및 테셀레이션을 위해 `Model::Options::preprocess` 또는 `pyneat.ModelOptions.preprocess`를 설정합니다.
- 모델 경로에서 해당 기능을 지원하는 경우, 모델에서 관리하는 CVU의 사전/사후 목표값을 `EV74`에 유지합니다.
- 디버깅 전용 그래프를 만들고 있지 않다면, 모델 앞에 `VideoConvert`, `VideoScale`, 또는 GStreamer `videoconvert`/`videoscale`를 삽입하지 않도록 하십시오.

카메라는 프레임을 제공합니다. CVU는 프레임 처리를 담당해야 합니다. CPU는 의도치 않은 이미지 처리 엔진이 되어서는 안 됩니다.

## 문제 해결을 위한 빠른 지도

| 증상 | 먼저 확인하세요. |
| --- | --- |
| 카메라가 감지되지 않습니다. | 선택한 `.dtbo`, 케이블 방향, 전원 재부팅, 그리고 커널/libcamera 로그를 확인하십시오. |
| `libcamerasrc`가 없습니다. | DevKit 빌드에 맞는 Neat/런타임 카메라 이미지 또는 카메라 패키지를 설치합니다. |
| `misconfig.media_caps` 또는 `not-negotiated` | 정확한 `format,width,height,framerate`를 `gst-launch-1.0`를 사용하여 확인합니다. `NV12 1920x1080@30`과 같이 지원되는 것으로 알려진 모드를 사용해 보세요. |
| 엄격한 제로 복사 실패 | `allow_cpu_fallback = true`를 설정하거나, SiMaAI의 제로 복사 속성을 활용하는 카메라 스택을 사용하세요. |
| 출력되는 색상이 올바르지 않음 | 프레임이 RGB/BGR이 아닌 `NV12` 형식으로 해석되는지 확인합니다. `libcamerasrc`에서 직접 생성된 JPEG 이미지에도 문제가 있다면, Neat를 디버깅하기 전에 카메라 ISP/튜닝을 먼저 디버깅하십시오. |
| 처리량이 낮습니다. | CPU를 사용한 비디오 변환/확대 기능을 제거하고, 지속적으로 데이터를 가져오며, 최신 데이터를 우선적으로 사용하는 실시간 소스 큐 정책을 적용합니다. |

다른 증상에 대해서는 다음을 참조하십시오. [문제 해결](/reference/troubleshooting).
