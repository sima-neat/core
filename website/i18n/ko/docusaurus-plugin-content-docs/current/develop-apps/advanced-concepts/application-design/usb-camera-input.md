---
title: "USB 카메라 사용하기"
description: "UsbCameraInput으로 USB(UVC) 웹캠의 MJPEG 또는 YUYV 프레임을 읽고, JPEG를 전달하거나 디코딩하거나, YUYV 프레임을 모델과 H.264 뷰어에 공급합니다. 장치 번호나 파이프라인 문자열은 필요 없습니다."
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# USB 카메라 사용하기

`nodes::UsbCameraInput`은 USB(UVC) 웹캠을 읽습니다. 기본적으로 MJPEG를 캡처하여 카메라의 JPEG 프레임을 출력하며, `format`을 `UsbCameraFormat::YUYV`로 설정하면 [YUYV 캡처](#capture-yuyv)에 설명한 대로 압축되지 않은 YUYV 프레임을 대신 캡처합니다. 카메라의 고정 ID를 전달하세요. `UsbCameraInputOptions`는 ([MIPI 카메라](./mipi-camera-input.md)의) `CameraInputOptions`와 같은 필드 이름과 기본값을 사용하지만, `camera_name`은 필수이고 `format`은 `UsbCameraFormat`이며 기본값은 `MJPEG`입니다.

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model

USB camera -> UsbCameraInput (YUYV) -+-> Preproc -> model
                                     `-> ColorConvert (NV12) -> VideoSender (H.264)
```

대부분의 웹캠은 4:2:2 JPEG를 보내며, 이는 NV16으로 디코딩됩니다. `usb_camera_info()`는 카메라의 JPEG가 디코딩되는 레이아웃을 알려 줍니다. 이 값을 `SimaDecode`의 `out_format`으로 사용하고, `NV16`을 반환하면 모델의 `preprocess.color_convert.input_format`을 `PreprocessColorFormat::NV16`으로 설정하세요. 그러면 프레임이 변환 단계 없이 모델에 전달됩니다.

## Neat가 지원하는 범위

- Platform 3.0에서 Modalix DevKit USB 포트에 연결한 UVC 카메라.
- 4:2:0 또는 4:2:2 크로마 샘플링의 MJPEG 캡처.
- Platform 3.0.1에서 짝수 너비, BT.601 또는 BT.709 색, 전체 또는 제한 범위의 YUYV 캡처.
- 카메라가 선택한 형식용으로 제공하는 정확한 너비, 높이, 프레임 속도. Neat는 크기를 조정하거나 가까운 모드를 고르지 않습니다.

H.264 전용 카메라, NV12나 UYVY 같은 다른 원시 형식, 깊이 또는 IR 스트림, 노출 같은 카메라 제어는 이 경로에 포함되지 않습니다.

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

카메라가 제공하는 MJPEG 및 YUYV 모드를 보려면 DevKit에서 형식 목록을 확인합니다. 장치 번호는 이 확인에만 쓰이며, Neat는 ID로 장치를 찾습니다.

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

노드는 카메라의 JPEG 프레임을 출력합니다. 모델을 실행하려면 다음 절처럼 `SimaDecode`로 디코딩하세요.

`UsbCameraInput`은 노드를 만들 때 모드를 확인하고 첫 프레임의 크로마 샘플링을 읽기 위해 카메라를 한 번 엽니다. 먼저 카메라를 사용하는 다른 프로그램을 닫으세요. `usb_camera_info(camera)`는 노드를 만들지 않고 같은 확인을 수행하며, 캡처 장치와 `decoded_format`(`NV12` 또는 `NV16`)을 반환합니다.

`decoded_format`도 필요하면 다음 절처럼 `usb_camera_info`를 한 번 호출하고 그 결과를 `UsbCameraInput(camera, info)`에 전달하세요. 그러면 노드는 카메라를 다시 열지 않고 그 확인 결과를 재사용하며, 장치가 여전히 해당 카메라에 속하는지 sysfs로만 확인합니다. 두 호출에는 같은 옵션을 전달하세요.

카메라를 뽑았거나 다른 포트로 옮긴 뒤에는 `UsbCameraInput`을 다시 호출하세요. 이전에 만든 그래프는 빌드할 때 실패하므로, 예전 장치 번호를 넘겨받은 다른 카메라를 여는 일은 없습니다.

## 카메라의 JPEG 전달하기

카메라 영상을 뷰어로 보내면서 동시에 모델을 실행하려면, 이 노드를 JPEG를 다시 인코딩하지 않고 RTP/JPEG로 보내는 `VideoSender` 패스스루와, `out_format`이 카메라의 `decoded_format`인 `SimaDecode`에 연결합니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once

neat::SimaDecodeOptions decode;
decode.type = neat::SimaDecodeType::MJPEG;
decode.out_format = info.decoded_format;  // NV12 or NV16
decode.dec_width = 1920;
decode.dec_height = 1080;
decode.dec_fps = 30;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::GraphLinkOptions video_link;
video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

neat::Graph source;
source.add(neat::nodes::UsbCameraInput(camera, info));  // reuses info, no second open
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
info = pyneat.usb_camera_info(camera)  # opens the camera once

decode = pyneat.SimaDecodeOptions()
decode.type = pyneat.SimaDecodeType.MJPEG
decode.out_format = info.decoded_format  # NV12 or NV16
decode.dec_width = 1920
decode.dec_height = 1080
decode.dec_fps = 30

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

source = pyneat.Graph()
source.add(pyneat.nodes.usb_camera_input(camera, info))  # reuses info, no second open
decoder = pyneat.Graph()
decoder.add(pyneat.nodes.sima_decode(decode))

graph = pyneat.Graph("usb_camera_forward")
graph.connect(source, decoder)
graph.connect(decoder, detector)  # your model Graph
graph.connect(source, pyneat.groups.video_sender(sender), video_link)
```

</CodeTab>
</CodeTabs>

카메라는 하드웨어 디코더가 그 자리에서 읽는 CMA 버퍼에 캡처하므로, 캡처와 디코딩 사이에서 프레임이 복사되지 않습니다. 커널 UVC 드라이버는 여전히 USB 패킷을 각 프레임 버퍼로 복사합니다. 송신기에는 [비디오 보내기](./video_sender.md)의 RTP/JPEG 제한이 적용됩니다.

## 메모리와 버려지는 프레임

| 옵션 | 효과 |
| --- | --- |
| `queue_depth`, `leaky_queue` | 카메라 뒤의 작은 leaky 큐가 그래프가 뒤처질 때 오래된 프레임을 버립니다. 카메라의 캡처 풀은 `queue_depth` + 6개의 버퍼를 가지며 최대 64개이므로, `queue_depth`는 최대 58입니다. |

웹캠은 JPEG 종료 마커 뒤에 바이트를 덧붙이는 경우가 많고, 일부 프레임은 불완전하게 도착하거나 USB 드라이버가 손상으로 표시합니다. `UsbCameraInput`은 드라이버가 손상으로 표시한 프레임을 출력 전에 버리므로, `VideoSender` 패스스루와 디코더 모두 이 프레임을 받지 않습니다. 디코더는 각 프레임을 종료 마커까지만 디코딩하고, 불완전하거나 손상된 프레임은 멈추지 않고 버립니다. 디코더 요소는 이를 `dropped-count` 속성에 셉니다.

## 문제 해결

| 메시지 | 조치 |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | 목록에 있는 ID 중 하나를 사용하세요. 카메라를 다른 포트로 옮기면 ID가 바뀝니다. |
| `no USB cameras are connected` | 케이블과 `lsusb`를 확인하세요. |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | 목록에 있는 모드를 고르세요. 일부 카메라는 높은 해상도를 낮은 프레임 속도에서만 제공합니다. |
| `is in use by another process` | 카메라를 열어 둔 다른 프로그램을 종료하세요. |
| `sent no usable MJPEG frame` | 카메라가 Neat가 디코딩할 수 없는 JPEG(예: 4:4:4 샘플링)를 보내거나 프레임을 전혀 보내지 않습니다. |
| `is no longer /dev/videoN` | 그래프를 만든 뒤 카메라를 뽑았거나 옮겼거나 번호가 바뀌었습니다. `UsbCameraInput`을 다시 호출하세요. |
| `is not at '/dev/videoN' as its UsbCameraInfo says` | `usb_camera_info`를 호출한 뒤 카메라를 뽑았거나 옮겼거나 번호가 바뀌었거나, 정보가 다른 카메라의 것입니다. `usb_camera_info`를 다시 호출하고 새 정보를 전달하세요. |
| `has no YUYV mode 640x480 at 30/1 fps; YUYV modes: ...` | 목록에 있는 YUYV 모드를 고르세요. |
| `has no YUYV capture node` | 카메라가 YUYV를 보내지 않습니다. MJPEG를 캡처하세요. |
| `YUYV requires an even width` | 너비가 짝수인 YUYV 모드를 고르세요. |
| `adjusted the requested progressive YUYV mode` 또는 `did not accept 30/1 fps` | 카메라가 이 모드를 목록에 표시했지만 받아들이지 않았습니다. 목록에 있는 다른 모드를 고르세요. |
| `has unsupported YUYV matrix/range/transfer metadata` 또는 `has unsupported YUYV color primaries` | 카메라가 BT.601 또는 BT.709가 아닌 색 공간을 보고합니다. MJPEG를 캡처하세요. |
| `capture info format does not match the selected mode` | `usb_camera_info`가 반환한 `UsbCameraInfo`를 그대로 전달하세요. |
| `capture info format does not match opt.format` | 정보가 다른 `format`으로 확인되었습니다. `UsbCameraInput`에 전달하는 옵션으로 `usb_camera_info`를 호출하세요. |
| `capture info describes another mode` | 옵션이 `usb_camera_info`에 전달한 옵션과 다릅니다. 두 호출에 같은 옵션을 전달하세요. |
| 카메라가 스트리밍을 시작할 때 YUYV Run이 실패함 | DevKit에서 `dmesg`를 실행하세요. `Failed to submit URB 0 (-28)`(`ENOSPC`)은 USB 컨트롤러가 압축되지 않은 모드에 필요한 대역폭을 확보할 수 없다는 뜻입니다. 더 낮은 해상도나 프레임 속도를 고르거나, 동시에 스트리밍하는 다른 USB 카메라나 오디오 장치를 중지하거나, MJPEG를 캡처하세요. |
| `YUYV requires explicit input color` | `PreprocOptions::input_color`를 `info.color`로 설정하세요. |
| `YUYV input does not support antialiased sampling`, `FP32 transforms require supported RGB/Gray tensor output`, `crop currently requires RGB, BGR or Gray input` 또는 `ROI lists require RGB/BGR/Gray inputs/outputs` | 모델 전처리에 YUYV 입력이 지원하지 않는 옵션이 필요합니다. [YUYV 프레임으로 모델 실행하기](#run-a-model-on-yuyv-frames)를 참조하세요. |
| `Strict encoder ingress requires EncoderNv12 output from ColorConvert` | 엄격한 `VideoSender` 앞에 `ImageSurfaceLayout::EncoderNv12`와 NV12 출력을 사용하는 `ColorConvert`를 두세요. |
| `Strict encoder ingress requires an exact integer frame rate` | 정수 프레임 속도의 모드를 캡처하고 인코더의 `fps`를 그 값으로 설정하세요. |
| `Graph JSON does not store PreprocOptions::input_color or ColorConvertOptions::input_colorimetry` | YUYV를 캡처하는 그래프는 코드로 구성하세요. `Graph::save`는 카메라의 색 정보를 저장할 수 없습니다. |

## YUYV 캡처 {#capture-yuyv}

많은 UVC 카메라는 압축되지 않은 YUYV도 보냅니다. 이는 패킹된 4:2:2 프레임으로, GStreamer에서는 `YUY2`라고 부릅니다. YUYV 프레임은 디코더가 필요 없으며, 모델 전처리와 `ColorConvert`가 캡처 버퍼에서 그 자리에서 읽습니다. 압축되지 않은 프레임은 MJPEG보다 훨씬 많은 USB 대역폭이 필요하므로, 카메라는 보통 YUYV를 더 낮은 해상도나 프레임 속도로 제공합니다.

`format`을 `UsbCameraFormat::YUYV`로 설정하고 `usb_camera_info`를 호출한 다음 그 결과를 `UsbCameraInput`에 전달하세요. 카메라가 제공하는 YUYV 모드 중 너비가 짝수인 모드를 고르세요. 반환된 `UsbCameraInfo`는 프레임도 설명합니다. `row_stride`는 바이트 단위 행 피치, `storage_bytes`는 각 캡처 버퍼의 크기, `color`는 카메라의 행렬(BT.601 또는 BT.709)과 범위, `colorimetry`는 같은 해석을 GStreamer colorimetry 문자열로 나타낸 것입니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
camera.format = neat::UsbCameraFormat::YUYV;
camera.width = 640;
camera.height = 480;
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once
auto source = neat::nodes::UsbCameraInput(camera, info);         // reuses info
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera.format = pyneat.UsbCameraFormat.YUYV
camera.width = 640
camera.height = 480
info = pyneat.usb_camera_info(camera)  # opens the camera once
source = pyneat.nodes.usb_camera_input(camera, info)  # reuses info
```

</CodeTab>
</CodeTabs>

노드는 픽셀당 2바이트인 HWC 레이아웃의 `UInt8` 프레임을 출력합니다. 프레임은 30000/1001 같은 분수 속도를 포함해 카메라의 프레임 속도를 유지하며, 홀수 높이도 허용됩니다. 기본값인 `UsbCameraFormat::MJPEG`에서는 노드가 MJPEG를 캡처합니다.

### YUYV 프레임으로 모델 실행하기 {#run-a-model-on-yuyv-frames}

모델의 `preprocess.color_convert.input_format`을 `PreprocessColorFormat::YUYV`로 설정하고 최대 입력 크기를 카메라 모드에 맞추세요. YUYV 프레임은 어떤 행렬과 범위를 쓰는지 알려 주지 않으므로, `input_color`가 `info.color`인 `Preproc`으로 모델 경로를 구성하고 그 뒤에 모델의 추론과 후처리를 이어 붙이세요.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Model::Options options;
options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::YUYV;
options.preprocess.input_max_width = 640;
options.preprocess.input_max_height = 480;
neat::Model model("model.tar.gz", options);

neat::PreprocOptions preproc(model);
preproc.input_color = info.color;

neat::Graph graph("usb_camera_yuyv");
graph.add(neat::nodes::UsbCameraInput(camera, info));
graph.add(neat::nodes::Preproc(preproc));
graph.add(model.inference());
graph.add(model.postprocess());
graph.add(neat::nodes::Output());
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
options = pyneat.ModelOptions()
options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.YUYV
options.preprocess.input_max_width = 640
options.preprocess.input_max_height = 480
model = pyneat.Model("model.tar.gz", options)

preproc = pyneat.PreprocOptions(model)
preproc.input_color = info.color

graph = pyneat.Graph("usb_camera_yuyv")
graph.add(pyneat.nodes.usb_camera_input(camera, info))
graph.add(pyneat.nodes.preproc(preproc))
graph.add(model.inference())
graph.add(model.postprocess())
graph.add(pyneat.nodes.output())
```

</CodeTab>
</CodeTabs>

YUYV 프레임에 대한 모델 전처리는 다음을 지원하지 않습니다.

- 안티에일리어싱 크기 조정 샘플링(`ResizeInterpolation::BilinearAntialiased`)
- 고정 소스 연산: `rescale_factor`와 이 값이 선택하는 고정 정규화, `quantization_step`, `PreprocFloatingConversion::NearestEven`, `PreprocColorConversionPlacement::BeforeResize`
- `ResizeMode::Crop`
- 런타임 ROI 목록

전처리에 이 중 하나가 필요한 모델은 그래프를 빌드할 때 실패합니다. 대신 MJPEG를 캡처하거나, 먼저 `ColorConvert`로 프레임을 `ImageFormat::RGB`로 변환하고(`color`를 `info.color`로, `input_colorimetry`를 `info.colorimetry`로 설정) 모델의 입력 형식을 `PreprocessColorFormat::RGB`로 설정하세요.

### YUYV 프레임을 H.264로 보기 {#view-yuyv-frames-as-h264}

카메라 영상을 뷰어에도 보내려면 `Preproc` 앞에서 분기하고 뷰어 분기를 `RealtimeLatestByStream`으로 연결하여 뷰어가 모델을 지연시키지 않게 하세요. `input_colorimetry`가 `info.colorimetry`인 `ColorConvert`로 프레임을 인코더 레이아웃의 NV12로 변환한 다음, `SimaEncodeOptions::input_policy`가 `EncoderInputPolicy::RequireDirect`인 엄격한 H.264 `VideoSender`로 인코딩하세요.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::ColorConvertOptions convert;
convert.output_format = neat::ImageFormat::NV12;
convert.surface = neat::ImageSurfaceLayout::EncoderNv12;
convert.color = info.color;
convert.color.output_siting = {neat::ImageChromaSiting::Cosited,
                               neat::ImageChromaSiting::Centered};
convert.input_colorimetry = info.colorimetry;

neat::SimaEncodeOptions encode;
encode.fps = 30;
encode.input_policy = neat::EncoderInputPolicy::RequireDirect;
auto sender = neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
sender.host = "127.0.0.1";

neat::Graph viewer("usb_camera_yuyv_viewer");
viewer.add(neat::nodes::ColorConvert(convert));
viewer.add(neat::nodes::groups::VideoSender(sender));
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
convert = pyneat.ColorConvertOptions()
convert.output_format = pyneat.ImageFormat.NV12
convert.surface = pyneat.ImageSurfaceLayout.EncoderNv12
convert.color = info.color
convert.color.output_siting = [pyneat.ImageChromaSiting.Cosited, pyneat.ImageChromaSiting.Centered]
convert.input_colorimetry = info.colorimetry

encode = pyneat.SimaEncodeOptions()
encode.fps = 30
encode.input_policy = pyneat.EncoderInputPolicy.RequireDirect
sender = pyneat.VideoSenderOptions.from_raw(encode)
sender.host = "127.0.0.1"

viewer = pyneat.Graph("usb_camera_yuyv_viewer")
viewer.add(pyneat.nodes.color_convert(convert))
viewer.add(pyneat.groups.video_sender(sender))
```

</CodeTab>
</CodeTabs>

엄격한 송신기는 CPU에서 프레임을 변환하거나 복사하지 않으며, 카메라의 원색, 전달 특성, 행렬, 범위를 H.264 스트림에 신호로 전달합니다. 짝수 너비와 높이, 그리고 인코더의 `fps`와 같은 정수 프레임 속도가 필요하며, 그 밖의 입력은 반올림하지 않고 거부합니다. [비디오 보내기](./video_sender.md#strict-encoder-input)를 참조하세요.

캡처에서 모델 또는 인코더까지 프레임은 복사되지 않습니다. 다만 커널 UVC 드라이버는 여전히 USB 패킷을 각 프레임 버퍼에 복사합니다. `UsbCameraInput`은 드라이버가 손상으로 표시한 프레임을 버립니다. 각 프레임에 대한 드라이버의 바이트 수는 그래프에 전달되지 않으므로, 드라이버가 표시하지 않은 불완전한 프레임은 버퍼의 나머지 부분에 이전 프레임의 데이터가 남은 채로 통과합니다.
