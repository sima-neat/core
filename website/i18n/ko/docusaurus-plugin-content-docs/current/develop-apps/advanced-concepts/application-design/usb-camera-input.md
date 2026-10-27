---
title: "USB 카메라 사용하기"
description: "UsbCameraInput으로 USB(UVC) 웹캠의 MJPEG 프레임을 읽고, 하드웨어 디코더에서 제자리 디코딩하거나 JPEG 그대로 전달합니다. 장치 번호나 파이프라인 문자열은 필요 없습니다."
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# USB 카메라 사용하기

`nodes::UsbCameraInput`은 MJPEG를 보내는 USB(UVC) 웹캠을 읽습니다. 카메라의 고정 ID를 전달하면 [MIPI 카메라](./mipi-camera-input.md)와 마찬가지로 그래프가 SiMaAI 메모리의 프레임을 받습니다. `UsbCameraInputOptions`는 `CameraInputOptions`의 필드 이름과 기본값을 사용하지만, `camera_name`은 필수이고 `format`은 없습니다. 프레임은 카메라 고유의 레이아웃을 유지하며, 4:2:0 JPEG는 NV12, 4:2:2 JPEG는 NV16입니다.

```text
USB camera -> UsbCameraInput -> model-managed CVU preproc -> MLA -> Output
```

내부적으로 카메라는 압축된 각 프레임을 하드웨어 MJPEG 디코더가 제자리에서 읽는 CMA 버퍼에 직접 기록하므로, 캡처와 디코딩 사이에 CPU 복사가 일어나지 않습니다. 대부분의 웹캠은 4:2:2 JPEG를 보내며, 이는 NV16으로 디코딩됩니다. 모델은 NV16을 바로 읽습니다. `usb_camera_info()`가 `NV16`을 반환하면 모델의 `preprocess.color_convert.input_format`을 `PreprocessColorFormat::NV16`으로 설정하세요. 그러면 프레임이 변환 단계 없이 모델에 전달됩니다.

## Neat가 지원하는 범위

- Platform 3.0에서 Modalix DevKit USB 포트에 연결한 UVC 카메라.
- 4:2:0 또는 4:2:2 크로마 샘플링의 MJPEG 캡처.
- 카메라가 MJPEG용으로 제공하는 정확한 너비, 높이, 프레임 속도. Neat는 크기를 조정하거나 가까운 모드를 고르지 않습니다.

H.264 전용 카메라, 원시 YUYV/NV12 캡처, 깊이 또는 IR 스트림(예: RealSense 깊이), 노출 같은 카메라 제어는 이 경로에 포함되지 않습니다.

## 카메라 ID 찾기

각 USB 카메라에는 USB 포트와 공급업체 ID, 제품 ID로 만든 고정 ID가 있으며, 형식은 libcamera와 같습니다.

```text
/base/pci@7000000000-3.1:1.0-046d:08e5
```

ID는 재부팅하거나 연결 순서가 바뀌어도 그대로입니다. 카메라를 다른 USB 포트로 옮기면 바뀌므로, 서로 다른 포트에 연결한 같은 카메라 두 대는 ID가 다릅니다. 연결된 카메라를 나열합니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
for (const std::string& id : simaai::neat::usb_camera_ids()) {
  std::cout << id << "\n";
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

print(pyneat.usb_camera_ids())
```

</CodeTab>
</CodeTabs>

카메라가 제공하는 MJPEG 모드를 보려면 DevKit에서 형식 목록을 확인합니다. 장치 번호는 이 확인에만 쓰이며, Neat는 ID로 장치를 찾습니다.

<ShellCommand prompt="devkit">
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --list-formats-ext
</ShellCommand>

## 그래프 만들기

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::UsbCameraInputOptions camera;
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5";
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.buffer_name = "camera0";

neat::Graph graph("usb_camera");
graph.add(neat::nodes::UsbCameraInput(camera));
graph.add(neat::nodes::Output());

neat::Run run = graph.build();
std::optional<neat::Sample> frame = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.UsbCameraInputOptions()
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5"
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.buffer_name = "camera0"

graph = pyneat.Graph("usb_camera")
graph.add(pyneat.nodes.usb_camera_input(camera))
graph.add(pyneat.nodes.output())

run = graph.build()
frame = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

모델은 MIPI 카메라와 같은 방식으로 추가합니다. [MIPI 카메라 사용하기](./mipi-camera-input.md)의 원시 MLA 그래프 절을 참고하세요.

`UsbCameraInput`은 노드를 만들 때 모드를 확인하고 첫 프레임의 크로마 샘플링을 읽기 위해 카메라를 한 번 엽니다. 먼저 카메라를 사용하는 다른 프로그램을 닫으세요. `usb_camera_info(camera)`는 노드를 만들지 않고 같은 확인을 수행하며, 캡처 장치와 `decoded_format`(`NV12` 또는 `NV16`)을 반환합니다.

카메라를 뽑았거나 다른 포트로 옮긴 뒤에는 `UsbCameraInput`을 다시 호출하세요. 이전에 만든 그래프는 빌드할 때 실패하므로, 예전 장치 번호를 넘겨받은 다른 카메라를 여는 일은 없습니다.

## 카메라의 JPEG 전달하기

카메라 영상을 뷰어로 보내면서 동시에 모델을 실행하려면 `output`을 `UsbCameraOutput::Jpeg`로 설정하세요. 그러면 노드는 카메라의 JPEG 프레임을 디코딩하지 않고 그대로 출력합니다. 이 노드를 JPEG를 다시 인코딩하지 않고 RTP/JPEG로 보내는 `VideoSender` 패스스루와, `out_format`이 카메라의 `decoded_format`인 `SimaDecode`에 연결합니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
camera.output = neat::UsbCameraOutput::Jpeg;

neat::SimaDecodeOptions decode;
decode.type = neat::SimaDecodeType::MJPEG;
decode.out_format = neat::usb_camera_info(camera).decoded_format;  // NV12 or NV16
decode.dec_width = 1920;
decode.dec_height = 1080;
decode.dec_fps = 30;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::GraphLinkOptions video_link;
video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

neat::Graph source;
source.add(neat::nodes::UsbCameraInput(camera));
neat::Graph decoder;
decoder.add(neat::nodes::SimaDecode(decode));

neat::Graph graph("usb_camera_forward");
graph.connect(source, decoder);
graph.connect(decoder, detector);  // your model Graph
graph.connect(source, neat::nodes::groups::VideoSender(sender), video_link);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera.output = pyneat.UsbCameraOutput.Jpeg

decode = pyneat.SimaDecodeOptions()
decode.type = pyneat.SimaDecodeType.MJPEG
decode.out_format = pyneat.usb_camera_info(camera).decoded_format  # NV12 or NV16
decode.dec_width = 1920
decode.dec_height = 1080
decode.dec_fps = 30

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

source = pyneat.Graph()
source.add(pyneat.nodes.usb_camera_input(camera))
decoder = pyneat.Graph()
decoder.add(pyneat.nodes.sima_decode(decode))

graph = pyneat.Graph("usb_camera_forward")
graph.connect(source, decoder)
graph.connect(decoder, detector)  # your model Graph
graph.connect(source, pyneat.groups.video_sender(sender), video_link)
```

</CodeTab>
</CodeTabs>

이 모드에서는 디코딩 분기가 각 프레임을 디코딩하기 전에 복사합니다. 카메라 드라이버는 버퍼를 DMA-BUF로 내보내지만, 디코더는 대개 이를 제자리에서 가져올 수 없기 때문입니다. 송신기에는 [비디오 보내기](./video_sender.md)의 RTP/JPEG 제한이 적용됩니다.

## 메모리와 버려지는 프레임

| 옵션 | 효과 |
| --- | --- |
| `allow_cpu_fallback` | USB 카메라에서는 효과가 없습니다. 캡처는 항상 디코더 자체 버퍼에 기록되므로 복사되는 프레임이 없습니다. 옵션을 `CameraInputOptions`와 맞추기 위해 남겨 두었습니다. |
| `queue_depth`, `leaky_queue` | 디코더 앞의 작은 leaky 큐가 그래프가 뒤처질 때 오래된 프레임을 버립니다. |
| `capture_buffer_count`(두 번째 인수) | 카메라와 디코더가 공유하는 CMA 버퍼입니다. 0이면 큐 깊이에 디코더 입력 슬롯 두 개를 더한 값을 씁니다. 프레임 출력에만 해당하며, `UsbCameraOutput::Jpeg`에서는 0이어야 합니다. |

여기서 제로 복사란 캡처와 디코딩 사이에 압축 프레임이 복사되지 않는다는 뜻입니다. 디코딩된 프레임은 별도 버퍼입니다.

웹캠은 JPEG 종료 마커 뒤에 바이트를 덧붙이는 경우가 많고, 일부 프레임은 불완전하게 도착하거나 USB 드라이버가 손상으로 표시합니다. 디코더는 각 프레임을 종료 마커까지만 디코딩하고, 불완전하거나 손상된 프레임은 멈추지 않고 버립니다. 디코더 요소는 이를 `dropped-count` 속성에 셉니다.

## 문제 해결

| 메시지 | 조치 |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | 목록에 있는 ID 중 하나를 사용하세요. 카메라를 다른 포트로 옮기면 ID가 바뀝니다. |
| `no USB cameras are connected` | 케이블과 `lsusb`를 확인하세요. |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | 목록에 있는 모드를 고르세요. 일부 카메라는 높은 해상도를 낮은 프레임 속도에서만 제공합니다. |
| `is in use by another process` | 카메라를 열어 둔 다른 프로그램을 종료하세요. |
| `sent no usable MJPEG frame` | 카메라가 Neat가 디코딩할 수 없는 JPEG(예: 4:4:4 샘플링)를 보내거나 프레임을 전혀 보내지 않습니다. |
| `is no longer /dev/videoN` | 그래프를 만든 뒤 카메라를 뽑았거나 옮겼거나 번호가 바뀌었습니다. `UsbCameraInput`을 다시 호출하세요. |
| `requires a neatdecoder with the input-pool-buffers property` | 설치된 디코더가 너무 오래되었습니다. `simaai-scheduler` 패키지를 업데이트하세요. |
