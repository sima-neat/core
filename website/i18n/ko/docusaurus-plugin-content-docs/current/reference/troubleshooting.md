---
title: "문제 해결"
description: "새로운 오류에 대한 증상 기반 수정 Neat 사용자가 가장 많이 클릭함"
sidebar_position: 5
---

# 문제 해결

각 항목은 **증상 → 원인 → 해결 방법** 형식으로 구성되어 있습니다. 증상 제목은 정확한 오류 메시지이며, 이 페이지에서(Ctrl-F) 해당 메시지를 검색하여 확인할 수 있습니다. 모든 항목은 현재 소스를 기준으로 검증되었거나 DevKit에서 재현되었습니다.

어디서부터 시작해야 할지 모르는 경우, 다음으로 이동하십시오.
[문제가 발생했을 때: 진단 기능](#when-youre-stuck-diagnostics).

## 설치 및 환경 설정

### 1. `pyneat is not importable. Either Neat is not installed, or the venv is not activated.`

:::info 원인
`pyneat` 가상 환경이 활성화되지 않았거나, 실행 중인 환경에 휠이 설치되지 않았습니다.
:::

:::tip 수정
파이썬 코드를 실행하기 전에 DevKit 환경을 활성화하세요.
```bash
source ~/pyneat/bin/activate
```
:::

### 2. GST 플러그인을 로드하는 데 실패했습니다: `undefined symbol: _ZN16simaaidispatcher14DispatcherBase14submitPrepared...`

:::info 원인
Neat 런타임 공유 라이브러리가 동적 로더 경로에 없기 때문에 GStreamer 플러그인이 로드 시점에 런타임 심볼을 확인할 수 없습니다.
:::

:::tip 수정
실행하기 전에 런타임 디렉터리를 `LD_LIBRARY_PATH`에 추가하세요.
```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/neat/runtime:$LD_LIBRARY_PATH
```
:::

### 3. 모델 아카이브 누락 — `sima-cli modelzoo`가 아직 실행되지 않았습니다.

:::info 원인
코드에서 참조하는 `.tar.gz` 모델 아카이브(또는 `SIMA_YOLO_TAR` / `SIMA_RESNET50_TAR` / `SIMA_MODEL_TAR`)가 디스크에 존재하지 않습니다.
:::

:::tip 수정
Model Zoo에서 다운로드하세요.
```bash
sima-cli modelzoo get yolo_v8s     # or resnet_50, etc.
```
:::

## 구축

### 4. `find_package(SimaNeat CONFIG)`에서 패키지를 찾을 수 없습니다.

:::info 원인
CMake가 `SimaNeatConfig.cmake`을 찾을 수 없습니다(`lib/cmake/SimaNeat/`에 설치됨). 기본 DevKit 설치에서는 기본 시스템 접두사에 설치되지만, SDK 크로스 빌드에서는 sysroot가 `CMAKE_PREFIX_PATH`에 없습니다.
:::

:::tip 수정
`SYSROOT`를 내보내고 `CMakeLists` 파일에서 이를 프리픽스 경로에 추가하도록 합니다([안녕하세요 Neat 템플릿](/develop-apps/hello-neat/minimal) 예제에서 이 작업을 수행합니다).
```cmake
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu")
endif()
find_package(SimaNeat REQUIRED CONFIG)
```
:::

## 모델을 로드하고 설정을 진행 중입니다.

### 5. `failed to read image: <path>`

:::info 원인
OpenCV(`cv2.imread` / `cv::imread`)에서 null을 반환했습니다. 이는 파일이 존재하지 않거나, 읽을 수 없거나, 디코딩 가능한 이미지가 아니기 때문입니다.
:::

:::tip 수정
입력 텐서를 생성하기 전에 경로를 확인하고 파일이 유효한 JPEG/PNG 파일인지 확인합니다.
:::

### 6. `reason=topk must be > 0` (`boxdecode`에서)

:::info 원인
감지 모델의 `ModelOptions.top_k` 값이 `0`으로 설정되어 있습니다. 박스 디코딩 단계에서는 양수 값을 설정해야 합니다.
:::

:::tip 수정
긍정적인 `top_k` 값을 설정합니다(튜토리얼에서는 `100`를 사용합니다).
```python
opt.top_k = 100
```
*(이 메시지는 EV74 박스 디코딩 플러그인에서 생성되었습니다.)*
:::

### 7. `preproc_upsample_not_supported`

:::info 원인
원본 이미지는 모델의 입력 해상도보다 작으므로, 전처리 과정에서 **업스케일링**을 해야 합니다. 하지만 구형 EV74 전처리 펌웨어는 업스케일링을 지원하지 않으므로 다운스케일링만 가능합니다.
:::

:::tip 수정
모델 입력 크기 이상(예: YOLOv8의 경우 ≥ 640x640)의 소스 이미지를 입력하거나, 업샘플링 커널이 포함된 빌드로 `neat-ev74-firmware`를 업데이트하십시오.
*(이 메시지는 EV74 전처리 플러그인/펌웨어에서 생성됩니다.)*
:::

### 8. 낮은 `score_threshold` → 후처리 지연 시간 급증

:::info 원인
감지 임계값이 낮을수록 임계값 처리를 통과하는 후보 박스의 수가 증가하며, NMS(Non-Maximum Suppression) 연산 비용은 생존하는 박스 수의 **제곱**에 비례하여 증가합니다.
:::

:::tip 수정
미약한 감지를 포착하기 위해 필요한 만큼만 임계값을 낮추고, 최악의 경우를 `top_k`로 제한합니다. [감지 상자 읽기](/tutorials/read-detection-boxes)를 참조하십시오.
:::

## 추론 실행 중

### 9. `misconfig.media_caps … Internal data stream error … reason not-negotiated (-4)`

:::info 원인
원시 이미지 입력을 사용하는 경우, 전처리 단계가 활성화되지 않았거나 입력 유형이 선언되지 않았으므로, 앱 소스와 첫 번째 단계 간에 기능 협상이 이루어질 수 없습니다.
:::

:::tip 수정
`ModelOptions`에서 이미지 입력과 전처리 프리셋을 선언합니다.
```python
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
```
:::

### 10. `No channel available (all candidate channel opens failed)`

:::info 원인
EV74 디스패처는 로드된 펌웨어에서 구현되지 않은 커널을 예약하려고 시도했습니다. 이는 일반적으로 `neat-runtime`과 `neat-ev74-firmware`가 **동일한 빌드가 아니기** 때문입니다(내부 해시 불일치). 예를 들어, 부분 업데이트가 발생한 경우입니다.
:::

:::tip 수정
일치하는 `neat-*` 세트를 함께 설치합니다(동일한 해시 값). 런타임과 펌웨어가 동일한 해시 값을 보고하는지 확인합니다. [호환성 → 버전이 일치하는 세트](/getting-started/compatibility#the-version-matched-set-firmware--runtime)를 참조하십시오.
*(이 메시지는 EV74 디스패처에서 보냅니다.)*
:::

### 11. `frame=N rtsp_timeout`

:::info 원인
RTSP 스트림 요청 시간이 초과되었습니다. URL이 잘못되었거나 스트림에서 프레임을 제대로 전송하지 못하고 있습니다.
:::

:::tip 수정
RTSP URL에 접속하여 스트리밍이 제대로 이루어지는지 확인하고, 전송 방식(TCP 또는 UDP)을 확인합니다. [RTSP 스트림을 사용합니다.](/tutorials/consume-rtsp-stream)을 참조하십시오.
:::

### 12. `CameraInput strict zero-copy requires external-buffer-mode`

:::info 원인
`CameraInputOptions::allow_cpu_fallback`의 기본값은 false이므로, Neat는 시작부터 끝까지 SiMaAI/장치 제로 복사 지원을 필요로 합니다. `libcamerasrc`가 일반적인 `external-buffer-mode` 속성을 광고하지 않거나, 설치된 메모리 라이브러리가 메모리 할당을 DMA-BUF로 내보낼 수 없는 경우입니다.
:::

:::tip 수정
일관성 있는 카메라 및 메모리 패키지가 설치된 경우 엄격한 제로 복사 방식을 유지합니다. DMA-BUF 내보내기가 없는 카메라 스택에서 실행해야 하는 경우 호환성 브리지에 명시적으로 참여합니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::CameraInputOptions camera;
camera.allow_cpu_fallback = true;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera = pyneat.CameraInputOptions()
camera.allow_cpu_fallback = True
```

</CodeTab>
</CodeTabs>

적응형 모드에서는 다운스트림 CVU/MLA 단계에 SiMaAI 메모리가 계속 전달됩니다. 업스트림 카메라 버퍼가 EV74에서 아직 사용 가능하지 않은 경우에만 카메라 브리지에서 복사가 이루어집니다.
:::

### 13. `misconfig.media_caps … libcamerasrc … not-negotiated (-4)`

:::info 원인
요청하신 카메라 캡처 설정이 카메라 스택에서 지원하는 모드와 일치하지 않거나, 보드 오버레이 또는 드라이버가 카메라를 제대로 인식하지 못했습니다.
:::

:::tip 수정
Neat 외부에서 동일한 형식, 해상도 및 프레임 속도를 확인하세요.

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

만약 문제가 해결되지 않으면, 먼저 오버레이, 케이블, 센서 드라이버 또는 카메라 모드를 수정하십시오. [Modalix DevKit MIPI 카메라 인터페이스 가이드](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces)를 사용하여 `.dtbo` 및 libcamera 검증 경로를 확인하십시오. 검증을 통과하면, 캡스를 사용자의 `CameraInputOptions`와 비교하십시오.
:::

### 14. 카메라 화면이 녹색, 보라색 또는 색조가 과도하게 나타남

:::info 원인
프레임이 잘못된 픽셀 형식 또는 색상 변환으로 해석되고 있습니다. 가장 흔한 오류는 `NV12` 카메라 프레임을 RGB/BGR 바이트로 처리하는 것입니다. Neat 전에 동일한 색조가 나타난다면, 문제는 카메라 ISP 튜닝 또는 libcamera 파이프라인일 가능성이 높습니다.
:::

:::tip 수정
카메라 캡과 모델 전처리 형식을 일관되게 유지하세요.

- 권장되는 모델 경로를 요청합니다(`camera.format = "NV12"`);
- `preprocess.color_convert.input_format = PreprocessColorFormat::NV12`를 설정합니다.
- 프로덕션 모델 경로에서는 CPU 기반의 `videoconvert`/`videoscale`를 사용하지 않도록 합니다.
- `gst-launch-1.0 libcamerasrc ... ! videoconvert ! jpegenc`를 사용하여 간단한 테스트를 실행하여 Neat을 실행하기 전에 색조 문제가 있는지 확인합니다.
:::

### 15. MIPI 카메라 튜토리얼의 `frame=N output_timeout`

:::info 원인
튜토리얼의 타임아웃 전에 앱에 결과물이 전달되지 않았습니다. 카메라-모델 그래프에서 이는 카메라가 프레임을 제대로 전송하지 않았거나, 캡스 협상이 실패했거나, 모델 경로가 아직 시작 중이거나, BoxDecode와 같은 후속 단계에서 결과물이 생성되지 않았음을 의미할 수 있습니다.
:::

:::tip 수정
먼저 카메라만 사용하는 경로를 검증합니다. 그런 다음 더 긴 타임아웃과 백엔드 출력 기능을 사용하여 튜토리얼을 다시 실행합니다.

<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/model.tar.gz --frames 2 --decode none \
  --pull-timeout-ms 15000 --print-backend
</ShellCommand>

프로덕션 경로는 폴백이 활성화된 경우 `libcamerasrc`, `neatcamerabridge`, `neatprocesscvu`, `neatprocessmla` 및 `appsink`를 포함해야 합니다. BoxDecode 경로의 경우, `--decode` 토큰과 임계값이 모델 아카이브와 일치하는지 확인하십시오.
:::

### 16. 그래프 처리량이 낮거나 실시간 프레임이 누락됩니다.

:::info 원인
그래프에 역압이 가해지고 있습니다. 일반적인 원인으로는 처리 속도를 따라가지 못하는 풀 루프, 출력 샘플을 너무 오래 유지하는 경우, 성능에 영향을 미치는 주요 경로의 프레임별 로깅, 소스와 일치하지 않는 큐 정책, 그리고 명시적인 드롭/최신성 정책이 없는 실시간 스트림 등이 있습니다.
:::

:::tip 수정
재사용 가능한 `Run`을 사용하고, 런타임 정책을 명확하게 정의합니다.

- 최신 정보가 중요한 실시간 입력의 경우 `RunPreset::Realtime` / `pyneat.RunPreset.Realtime`을 사용하세요.
- 모든 입력이 중요한 일괄 처리 또는 파일 처리에 `RunPreset::Reliable` / `pyneat.RunPreset.Reliable`을 사용하세요.
- 앱이 큐가 가득 찼을 때 차단되지 않도록 하려면 `try_push(...)`를 사용하세요.
- `on_input_drop` 설정을 통해 `stream_id`, `frame_id`, `port_name` 및 이유별로 드롭된 횟수를 기록합니다.
- 지속적으로 데이터를 가져옵니다. 출력 대기열이 가득 차면 전체 그래프의 처리 속도가 저하될 수 있습니다.
- 앱이 런타임 기반 버퍼를 보유하고 있을 경우, 푸시하기 전에 출력물을 릴리스하거나 복사하세요.

다중 스트림 그래프의 경우, `stream_id` 및 `frame_id`를 유지하고 스트림별 출력 수를 확인합니다. 전체 FPS는 특정 스트림의 성능 저하를 숨길 수 있습니다. 자세한 내용은 [그래프 실행 → 실제 성능을 반영하여 처리량을 조정](/develop-apps/development-workflow/pipeline#tune-throughput-without-lying-to-yourself)를 참조하십시오.
:::

### 17. `unknown input/output name`, `no unambiguous default input`, 또는 `no unambiguous default output`

:::info 원인
그래프에는 명명된 엔드포인트가 있으며, 앱이 잘못된 이름을 `push(...)`하거나 `pull(...)`했거나, 둘 이상의 가능한 엔드포인트를 가진 그래프에서 명명되지 않은 엔드포인트를 사용했습니다.
:::

:::tip 수정
푸시하거나 풀하기 전에 이름을 확인하세요.

```python
run = graph.build()
print("inputs:", run.input_names())
print("outputs:", run.output_names())
```

그러면 정확한 엔드포인트 이름을 사용하세요.

```python
run.push("image", [tensor])
sample = run.pull("detections", timeout_ms=2000)
```

`Graph("name")`은 진단 레이블입니다. 엔드포인트를 생성하지 않습니다. 엔드포인트는 `nodes.input("name")` 및 `nodes.output("name")`에서 생성됩니다.
:::

### 18. `pull(...)`은 시간 초과 전에 아무런 결과도 반환하지 않습니다.

:::info 원인
제한 시간 내에 요청된 출력을 생성하는 샘플이 없습니다. 그래프가 여전히 실행 중이거나, 출력 이름이 잘못되었거나, 입력에 백프레셔가 적용되었거나, 그래프가 닫혔거나, 런타임 오류가 발생했을 수 있습니다.
:::

:::tip 수정
타임아웃, 연결 종료, 오류를 분리합니다. C++에서는 구조화된 풀 오버로드를 사용하세요.

```cpp
simaai::neat::Sample sample;
simaai::neat::PullError error;

switch (run.pull("detections", /*timeout_ms=*/1000, sample, &error)) {
case simaai::neat::PullStatus::Ok:
  break;
case simaai::neat::PullStatus::Timeout:
  // Keep waiting, push more input, or report timeout.
  break;
case simaai::neat::PullStatus::Closed:
  // End of stream.
  break;
case simaai::neat::PullStatus::Error:
  std::cerr << error.code << ": " << error.message << "\n";
  break;
}
```

또한 `run.last_error()`를 확인하고, 엔드포인트 이름, 입력 데이터 유형/레이아웃/형식, 그리고 애플리케이션이 모든 출력 브랜치에서 지속적으로 데이터를 가져오고 있는지 확인하십시오.
:::

### 19. 이전 코드 조각이 `push_timeout_ms`, `pull_or_throw`, 최상위 수준의 `input_max_*` 또는 `boxdecode_original_*`로 인해 실패합니다.

:::info 원인
이 코드 조각은 이전 버전의 옵션 인터페이스 또는 비공개/내부 경로를 사용하여 작성되었습니다. 현재 앱 코드는 공개 API인 `ModelOptions`, `RunOptions` 및 `Run`을 사용해야 합니다.
:::

:::tip 수정
현재 공개된 이름을 사용하세요.

- 입력 압력 조절을 위해 `RunOptions.queue_depth`, `overflow_policy` 및 `try_push(...)`를 사용하십시오.
- `pull_or_throw` 대신 `pull(...)` 또는 구조화된 `PullStatus` 오버로드를 사용하세요.
- 이전 코드 조각에서 최상위 수준의 `input_max_*` 필드를 설정하는 경우, 동적 입력 제한을 `ModelOptions.preprocess.input_max_width`, `input_max_height` 및 `input_max_depth` 아래로 이동시키고, 필요한 경우에만 해당 값을 설정합니다.
- BoxDecode 좌표 매핑의 경우, 전처리 메타데이터를 사용하는 것이 좋습니다. 새로운 예제에서는 더 이상 사용되지 않는 원래 크기 필드를 설정하지 마십시오.

참고한 페이지에 아직 이전 철자가 표시되어 있다면, 해당 페이지를 오래된 문서로 간주하고 문서 관련 버그를 신고하여 다음 독자가 동일한 문제에 직면하지 않도록 하십시오.
:::

## 텐서와 Python의 상호 운용성

### 20. `… expects a TensorList; pass [tensor] instead of a single Tensor`

:::info 원인
단순한 `Tensor` (또는 `Sample`)가 `run` / `push` / `build`에 전달되었습니다. API는 명시적인 목록을 요구하며, 이는 의도적인 것이며 버그가 아닙니다.
:::

:::tip 수정
다음으로 묶습니다: `model.run([tensor])`, `run.push([tensor])`, `graph.build([tensor])`.
:::

### 21. `image-mode Tensor input requires explicit image format metadata`

:::info 원인
이미지 입력을 받는 모델이 픽셀 형식이 지정되지 않은 텐서를 받았으므로, Neat은 바이트 레이아웃을 해석할 수 없습니다.
:::

:::tip 수정
명시적인 형식을 사용하여 텐서를 생성합니다: `pyneat.Tensor.from_numpy(arr, image_format=pyneat.PixelFormat.RGB)`.
:::

### 22. `byte_format tensors cannot also specify image_format`

:::info 원인
텐서가 `byte_format=` (불투명 바이트)와 `image_format=` (픽셀)를 모두 사용하여 구성되었으며, 이 둘은 상호 배타적입니다.
:::

:::tip 수정
둘 중 하나만 통과하면 됩니다. 둘 다 통과할 필요는 없습니다.
:::

## 다른 스택에서 왔습니다.

- **"내 `.engine` / `.blob` / `.dlc` / `.hef`는 어디에 있나요?"** — Neat은 `.tar.gz` 모델 아카이브를 로드합니다. 이는 컴파일된 아티팩트와 동일합니다.
- **"CUDA 스트림/OpenCL 큐에 작업을 고정하려면 어떻게 해야 하나요?"** — 그렇게 할 필요가 없습니다. 비동기 `push`/`pull`을 사용하여 생산자와 소비자를 분리하고 대신 `RunOptions`를 조정하세요.
- **"왜 처리량은 명시된 TOPS 값보다 낮을까요?"** — 일반적으로 호스트 오버헤드, 큐의 자원 부족, 출력 역압력 또는 드롭 정책이 원인이며, 가속기의 문제는 아닙니다. [그래프 실행](/develop-apps/development-workflow/pipeline)를 참조하십시오.

## 문제가 발생했을 때: 문제 해결 방법

섣불리 추측하기 전에 다음 사항을 확인하세요.

**파이프라인/실행(Python 및 C++)을 검사하세요:**
- `graph.validate()`는 빌드하기 전에 내장된 계약에 따라 연결을 검증하는 `GraphReport`입니다. 해당 `error_code`를 확인하세요.
- `graph.describe()` → 해결된 파이프라인을 텍스트 형태로 표현 (노드 이름 + 캡스 체인).
- `run.input_names()` / `run.output_names()`는 런타임 푸시/풀 호출에서 허용되는 이름입니다.
- `run.start_measurement()` / `MeasureReport` → 카운터, 지연 시간, 입력 스트림 텔레메트리, 플러그인/에지 타이밍, 그리고 선택적인 전력 측정.
- `run.json(...)` / `run.save_json(...)` 또는 C++ `save_run_json(...)` → 샘플이 이동한 후 실행 증거를 확인합니다.
- `NeatError::report()` → 실행 중 오류가 발생했을 때 제공되는 구조화된 오류 상세 정보.


### 지원 자료를 모으세요.

다른 개발자나 SiMa.ai 지원팀의 도움이 필요하면, 다른 개발자가 문제를 재현할 수 있도록 증거 자료를 보내주세요. 다음 내용을 포함하세요:

- Neat 버전/빌드 정보: Python `pyneat.build_info()` 또는 C++ `sima_neat_version()`, `sima_neat_platform_version()` 및 `sima_neat_abi_version()`;
- 모델 아티팩트 이름, 모델 경로, 그리고 해당 모델이 어떻게 생성되었는지.
- 오류를 재현하는 데 필요한 가장 작은 실행 가능한 코드 조각입니다.
- 입력 셰이프, 데이터 유형, 레이아웃, 픽셀 형식, 페이로드 패밀리, 그리고 그래프가 앱에서 푸시되는 방식인지 아니면 소스에서 소유하는 방식인지;
- `run.input_names()` 및 `run.output_names()`에서 엔드포인트 이름을 가져옵니다.
- `GraphReport`는 `graph.validate()` 또는 `NeatError::report()`에서 생성된 JSON 형식입니다.
- 샘플이 실행 과정을 거친 후 `run.save_json(...)` 또는 C++ `save_run_json(...)`을 사용하여 실행 결과를 JSON 형식으로 내보냅니다.
- 지연 시간, 처리량, 패킷 손실 또는 전력 소비 문제가 발생했을 때 측정 결과로 나타나는 값입니다.

다중 스트림 문제의 경우, 각 스트림별 입력 수, 수락된 수, 출력 수, 삭제된 수를 포함합니다. 전체 FPS는 특정 스트림의 문제를 숨길 수 있습니다.

`GraphReport`를 수집할 때, 발생한 상황을 설명하는 필드를 유지합니다.

- `error_code` 및 `repro_note`;
- `pipeline_string`;
- `bus`;
- `repro_gst_launch` 및 `repro_env`;
- `dot_paths` 및 `caps_dump`;
- 경계 프로브가 있는 경우 `boundaries` / `BoundaryFlowStats`가 표시됩니다.
- 시드된 `build(input, ...)` 빌드 실패에 대한 `build_adaptation` 적용;
- 실행 후 카운터 및 지표에 대한 JSON 내보내기를 실행합니다.

**프레임워크 디버그 출력을 활성화**합니다. `SIMA_DEBUG_PROFILE`은 쉼표로 구분된 추적할 컴포넌트 목록입니다. 모든 컴포넌트를 추적하려면 `all`을 사용하거나, 추적 범위를 좁힐 수 있습니다.
```bash
export SIMA_DEBUG_PROFILE=all                 # everything
export SIMA_DEBUG_PROFILE=graph,gst,pipeline  # just these areas
```
알려진 구성 요소: `pipeline`, `graph`, `gst`, `appsink`, `inputstream`, `tensor`. 기본적으로 비활성화되어 있으며(디버그 출력이 없음).

캡스가 깨지는 지점을 시각적으로 검사하기 위해 GStreamer 그래프를 덤프합니다.
```bash
export SIMA_GST_DOT_DIR=/tmp     # writes .dot graphs on build/failure; default: off
```

## 오류 코드

`NeatError` (및 `GraphReport::error_code` / `PullError::code`)는 `domain.reason` 코드를 보고합니다. 프레임워크는 정확히 이 코드를 정의하므로, 해당 코드를 활성화하고 메시지를 읽어 자세한 내용을 확인하십시오.

| 코드 | 발생 조건 |
|---|---|
| `io.open` | 파일을 열거나 장치 경로에 접근할 수 없습니다. 파일이 없거나, 권한이 없거나, 커널 장치가 누락된 경우입니다(예: `/dev/rpmsg*`). |
| `io.parse` | JSON/구성 파일 파싱 오류입니다. 일반적으로 잘못된 MPK 계약 또는 단계별 구성 파일이 원인입니다. |
| `misconfig.pipeline_shape` | 파이프라인의 기하학적 구조 또는 최종 이름의 무결성에 문제가 있습니다. 예를 들어, 잘못된 싱크 개수, 순환 구조, 누락된 최종 `Output` 또는 중복된 요소 이름 등이 있습니다. |
| `misconfig.caps` | 스트리밍 전에 캡스 오버라이드 또는 인접 노드 계약이 프레임워크 검증에 실패했습니다. |
| `misconfig.media_caps` | 인접한 미디어 스테이지 간의 런타임 GStreamer 협상이 실패했습니다. |
| `misconfig.input_shape` | 입력 텐서가 모델의 요구 사항(랭크, 공간 차원, 채널 수)을 위반합니다. |
| `misconfig.runtime_abi_mismatch` | 프레임워크/런타임 플러그인 ABI 불일치 — 일반적으로 `pyneat` 및 런타임 아티팩트가 혼합되어 발생합니다. |
| `build.plugin_missing` | 필요한 GStreamer 요소 또는 코덱 플러그인을 사용할 수 없습니다. |
| `build.property_invalid` | A GStreamer 요소 속성 이름 또는 값이 유효하지 않습니다. |
| `build.pipeline_syntax` | 사용자 정의 GStreamer 구문이 올바르지 않습니다. |
| `build.parse_launch` | `gst_parse_launch` 오류를 더 구체적으로 분류할 수 없습니다. |
| `runtime.pull` | 더 구체적인 상위 수준 또는 근본 원인 코드가 없는 상태에서 풀(pull) 작업이 실패했습니다. |
| `infra.dispatcher_unavailable` | MLA/EV74/A65 디스패처를 가져올 수 없습니다. 펌웨어가 로드되지 않았거나, 라이선스가 없거나, 하드웨어 오류가 발생했습니다. CPU로 대체할 수 없습니다. |

이것은 간단한 문제 해결 안내서입니다. 모든 코드와 C++/Python 상수 이름에 대해서는 [완전한 오류 코드 목록](/reference/error-codes)를 참조하십시오.
