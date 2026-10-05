# 007 모델 출력에서 감지 상자 읽기

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | yolo_v8s |
| Labels | postprocessing, boxdecode, detection |

## Concept

`SimaBoxDecode`를 사용하여 원시 모델 출력을 사용 가능한 경계 상자로 디코딩합니다. 여기에는 임계값 처리, NMS(Non-Maximum Suppression), 좌표 매핑이 포함되며, 이 모든 것이 하나의 후처리 단계에 통합되어 있습니다. 그런 다음 결과를 파싱된 상자 또는 원시 압축된 바이트 버퍼로 읽습니다.

## Walkthrough

감지기는 직접적으로 바운딩 박스를 반환하지 않습니다. 감지기의 원시 출력은 여전히 임계값 처리, 비최대값 억제 및 좌표 매핑이 필요한 특징 맵의 스택입니다. `SimaBoxDecode`는 이 세 가지를 하나의 최적화된 단계로 수행하여 추론 텐서를 소스 이미지 픽셀의 최종 감지 결과로 변환하는 후처리 단계입니다.

이 장에서는 해당 디코드를 구성합니다. 즉, `decode_type`을 사용하여 모델 패밀리를 선택하고, 점수 임계값을 사용하여 신뢰도를 조절하고, NMS IoU 임계값을 사용하여 겹치는 부분을 억제하고, `top_k`을 사용하여 출력을 제한한 다음 모델을 실행하고 반환된 감지 결과의 수를 확인합니다. 이 과정을 마치면 구성된 감지기 파이프라인과 출력에서 읽은 감지 결과 수가 있으며, 또한 (아래의 "실제 적용" 참조) 전체 와이어 형식을 통해 모든 런타임에서 직접 바운딩 박스를 파싱할 수 있습니다.

### 디코드 구성 {#step-configure-decode}

이러한 옵션은 입력 계약과 후처리 동작을 모두 설정합니다. `decode_type` (`YoloV8` 사용)는 모델 패밀리 디코드 경로를 선택합니다. 신뢰도 임계값은 NMS 전에 약한 후보를 제거합니다. NMS IoU 임계값은 겹치는 바운딩 박스가 얼마나 적극적으로 병합되는지를 제어합니다. `top_k`은 결정론적 다운스트림 비용을 위해 최종 수를 제한하고, `boxdecode_original_width`/`boxdecode_original_height`은 디코드된 좌표를 소스 이미지 픽셀로 다시 매핑합니다. 이러한 각 값에 대한 조정 지침은 아래의 "실제 적용"에서 확인할 수 있습니다.

**C++:** `decode_type`은 `BoxDecodeType::YoloV8` 열거형을 사용합니다. 임계값/NMS/`top_k` 값은 `Model::Options`가 아닌 `stages::BoxDecodeOptions`를 통해 나중에 전달됩니다.

**Python:** `decode_type`은 `pyneat.BoxDecodeType.YoloV8` 열거형을 사용하며, `score_threshold`, `nms_iou_threshold`, `top_k` 및 `boxdecode_original_width`/`boxdecode_original_height`은 `ModelOptions`에 직접 설정됩니다. (`score_threshold`와 C++의 `detection_threshold`는 동일한 컨트롤을 나타냅니다. 자세한 내용은 "실제 적용"의 명명 참고 사항을 참조하십시오.)

### 모델 구축 {#step-load-model}

아카이브와 옵션에서 `Model`을 구성하면 디코드 구성이 모델에 바인딩되어 추론 및 후처리 단계에서 해당 설정을 사용합니다.

### 전처리, 추론 및 디코드 실행 {#step-run-decode}

이 단계에서 프레임은 전처리, MLA 추론 및 바운딩 박스 디코드를 거쳐 감지 결과를 생성합니다.

**C++:** 경로는 단계별로 명확하게 정의됩니다. `stages::Preproc`는 입력 텐서를 생성하고, `stages::Infer`는 모델을 실행하며, `stages::BoxDecodeOptions` (`detection_threshold = 0.55`, `nms_iou_threshold = 0.5`, `top_k = 100` 포함)는 다음에 실행될 디코드를 구성합니다.

**Python:** `model.run([tensor])`는 구성된 전체 경로를 한 번의 호출로 실행하고 `TensorList`를 반환합니다. BoxDecode가 모델 경로에 연결되면 첫 번째 텐서는 패킹된 `BBOX` 출력입니다.

### 박스 읽기 {#step-read-boxes}

마지막으로 디코드 출력을 실제로 사용할 수 있는 형태로 변환합니다.

**C++:** `stages::BoxDecodeResults(...)`는 `BoxDecodeResultList`를 반환합니다. 첫 번째 결과의 `boxes` 벡터는 이미 소스 픽셀에 맞춰 `{x1, y1, x2, y2, score, class_id}`로 파싱되었으므로 `decoded.boxes.size()`는 감지된 객체의 개수입니다.

**Python:** 결과는 `outputs[0]`에 있는 단일 `BBOX` `uint8` 텐서입니다. 처음 네 개의 리틀 엔디안 바이트는 감지된 객체의 개수(`struct.unpack_from("<I", buf, 0)`)입니다. 전체 레코드 레이아웃은 "실제 적용"에 설명되어 있습니다. 런타임에서 BoxDecode를 `model.run`에 연결하지 않으면 반환되는 `TensorList`에는 원시 특징 맵 헤드가 포함됩니다.

## In Practice

`SimaBoxDecode`는 `BBOX` 태그가 지정된 단일 출력 텐서를 출력합니다. 텐서에는 런타임 파서가 부동 소수점 감지로 해석하는 패킹된 바이트 버퍼가 포함됩니다. 두 단계의 계약(와이어 버퍼와 파싱된 `Box` 레코드)을 이해하는 것이 Python 또는 C++에서 출력을 읽는 핵심입니다.

### BBOX 텐서

디코드 단계는 각 입력 프레임에 대해 다음 속성을 가진 하나의 `BBOX` 텐서를 생성합니다.

| 필드 | 값 |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | 랭크-1: `[N_bytes]`, 여기서 `N_bytes`는 모델 아카이브에 패킹된 버퍼 용량입니다(예: 표준 YOLOv8 패키지의 경우 `[20160]`). |

텐서의 형태는 **바이트 수**이며, 감지된 객체의 개수가 아닙니다. 패킹된 바이트에는 작은 헤더와 고정 크기 박스 레코드의 연속된 배열이 포함됩니다. `N_bytes`는 모델 아카이브의 `buffers.input[0].size` 필드(박스 디코드 단계의 구성 JSON 내)에 의해 결정되며, 디코더가 단일 프레임에서 출력할 수 있는 최대 감지 객체 수를 제한합니다("런타임 차원과 패키지 값의 상호 작용" 아래 참조).

### 패킹된 와이어 형식

`uint8` 버퍼는 리틀 엔디안으로 레이아웃됩니다.

```
offset  size  content
------  ----  -------
  0      4    uint32  N = number of valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   (trailing bytes up to buffer capacity are padding, ignored)
```

각 `RawBox` 레코드는 24바이트입니다.

| 레코드 내 오프셋 | 크기 | 유형 | 필드 | 의미 |
|---|---|---|---|---|
| 0 | 4 | int32 | `x` | 원본 픽셀 단위의 좌측 상단 x 좌표 |
| 4 | 4 | int32 | `y` | 원본 픽셀 단위의 좌측 상단 y 좌표 |
| 8 | 4 | int32 | `w` | 원본 픽셀 단위의 너비 |
| 12 | 4 | int32 | `h` | 원본 픽셀 단위의 높이 |
| 16 | 4 | float32 | `score` | NMS 후 탐지 신뢰도 (값은 `[0.0, 1.0]`에 의해 결정되며, `detection_threshold` 값을 기준으로 필터링됨) |
| 20 | 4 | int32 | `class_id` | 예측된 클래스 ID (모델에서 정의; 0부터 시작; 클래스 이름 매핑은 모델 아카이브 메타데이터에 있음) |

단일 레코드와 일치하는 표준 Python `struct` 형식은 `"<iiiifi"`입니다. (리틀 엔디안, 4개의 부호 있는 정수, 하나의 부동 소수점, 하나의 부호 있는 정수).

런타임의 파싱 도우미(`parse_bbox_bytes` / `decode_bbox_tensor`는 `include/pipeline/DetectionTypes.h`에 있고, `tests/unit_testing/unit_detection_types_bbox_test.cpp`는 인터페이스 계약을 고정함)는 각 `RawBox`를 후속 코드에서 사용할 수 있는 `Box` 구조체로 확장합니다.

```cpp
struct Box {
  float x1, y1, x2, y2;  // x2 = x + w, y2 = y + h; clamped to [0, img_w|h]
  float score;
  int   class_id;
};
```

### 좌표 공간

`BBOX`에서 디코딩된 좌표는 **원본 이미지 픽셀** 단위이며, 이는 `original_width` / `original_height`로 전달된 것과 동일한 좌표계입니다 (또는 모델 아카이브가 패키징된 좌표계). 좌표는 `[0, 1]`로 정규화되지 않으며, 모델의 내부 레터박스 입력 공간으로 표현되지도 않습니다. 파서는 `(x1, y1, x2, y2)`를 `[0, original_width]` / `[0, original_height]`로 제한하므로 호출 코드는 이를 소스 프레임에 직접 그릴 수 있습니다.

### 예제

튜토리얼의 런타임 구성(`original_width = 640`, `original_height = 640`, `top_k = 100`) 및 표준 YOLOv8 패키지(`buffers.input[0].size = 20160`는 박스 디코딩 구성에 있음)를 사용하면 단일 디코딩된 프레임에서 다음이 생성됩니다.

- `out.kind == SampleKind.Tensor`
- `out.payload_tag == "BBOX"`
- `out.tensor.dtype == UInt8`, `out.tensor.shape == [20160]`
- 바이트 `[0:4]`는 리틀 엔디안으로 `N`를 제공합니다. `0 <= N <= 100`는 `top_k = 100` 때문입니다. `N`가 `0`이면 "이 프레임에서 임계값보다 높은 탐지가 없음"을 의미합니다. 따라서 0번 반복하고 아무것도 출력하지 않습니다.
- 바이트 `[4 : 4 + 24 * N]`에는 유효한 탐지가 포함됩니다. 그 이후의 모든 바이트는 0/패딩이며 무시해야 합니다.

Python에서 박스를 읽는 것은 `struct.unpack_from`입니다.

```python
import struct
payload = out.tensor.copy_payload_bytes()
count = struct.unpack_from("<I", payload, 0)[0]
for i in range(count):
    x, y, w, h, score, cls = struct.unpack_from("<iiiifi", payload, 4 + 24 * i)
    # (x, y, w, h) in source pixels; x2 = x + w, y2 = y + h
```

C++에서 `stages::BoxDecode` 헬퍼 함수는 이미 이 언패킹 작업을 수행한 `BoxDecodeResult`를 반환합니다. `result.boxes[i]`는 `(x, y, x+w, y+h)`에서 값을 가져와 `(x1, y1, x2, y2)`에 이미 채워진 `Box`입니다. 또한 이미지에 맞게 값이 조정됩니다.

### 오버라이드 계약: 런타임 차원 대 패키지된 모델 아카이브 기본값

`SimaBoxDecode`는 `decode_type`, `detection_threshold`, `nms_iou_threshold`, `top_k`, `original_width` 및 `original_height`에 대한 패키지된 기본값을 포함하는 훈련된 모델 아카이브에서 생성됩니다. 공개 생성자는 ```cpp
SimaBoxDecode(const Model& model,
              const std::string& decode_type = "",
              int original_width = 0, int original_height = 0,
              double detection_threshold = 0.0,
              double nms_iou_threshold = 0.0,
              int top_k = 0);
```입니다.

그리고 해당 Python 버전인 `pyneat.nodes.sima_box_decode(model, ...)`은 각 필드에 대해 간단한 "양수 값은 우선 적용, 0/빈 값은 보존" 규칙을 사용합니다.

> **이름 지정 참고 사항:** `detection_threshold`는 `SimaBoxDecode`의 생성자에서 사용하는 이름입니다. `ModelOptions.score_threshold` (Python 튜토리얼에서 사용)는 동일한 인수로 전달됩니다. 두 이름은 동일한 기본 제어를 나타냅니다.

| 런타임 인자 | 전달 값 | 동작 |
|---|---|---|
| `decode_type` | `""` (빈 값) | 모델 아카이브 / 모델 경로 추론 보존 |
| `decode_type` | 빈 값이 아닌 문자열 | 이번 실행에 대한 모델 아카이브 값 재정의 |
| `original_width` / `original_height` | `0` | 모델 아카이브에 패키징된 차원 보존 |
| `original_width` / `original_height` | 양의 정수 | 유효한 구성에서 `original_width` / `original_height` 재작성 |
| `detection_threshold` | `0.0` | 모델 아카이브에 패키징된 임계값 보존 |
| `detection_threshold` | `> 0.0` | 재정의 (YOLOv8 경고도 트리거됨) |
| `nms_iou_threshold` | `0.0` | 모델 아카이브에 패키징된 NMS IoU 보존 |
| `nms_iou_threshold` | `> 0.0` | 재정의 |
| `top_k` | `0` | 모델 아카이브에 패키징된 상위 K 값 보존 |
| `top_k` | `> 0` | 재정의 |

규칙은 각 필드에 엄격하게 적용됩니다.

- **Python 경로** — 튜토리얼은 모든 필드를 재정의합니다. 왜냐하면 `ModelOptions`가 양수 값을 설정하기 때문입니다.
- **C++ 경로** — `read_detection_boxes.cpp`는 `0.55f, 0.5f, 100`를 전달합니다 (따라서 `detection_threshold`, `nms_iou_threshold` 및 `top_k`가 재정의됨) 그리고 `bgr.cols, bgr.rows`를 양수로 전달합니다 (따라서 `original_width` / `original_height`도 재정의됨).

실질적인 결과:

- 모델 아카이브가 소스 프레임과 다른 해상도로 패키징된 경우, `original_width` 및 `original_height`를 명시적으로 전달하여 좌표가 소스 픽셀에 위치하도록 합니다.
- `detection_threshold` 및 `nms_iou_threshold`를 `0.0`으로 유지하는 것이 모델 아카이브의 검증된 기본값을 얻는 가장 안전한 방법입니다. 의도적으로 재조정할 때만 재정의하십시오.
- 낮은 `detection_threshold`를 신중하게 사용하십시오. 값이 낮을수록 임계값을 통과하는 후보 박스의 수가 많아지고, NMS 비용은 생존 박스 수의 제곱에 따라 증가하므로 매우 낮은 임계값은 후처리 계산 및 지연 시간을 크게 증가시킬 수 있습니다. 약한 감지를 감지해야 하는 만큼만 낮추고, `top_k`와 함께 사용하여 최악의 경우를 제한하십시오.

### 디코딩 유형 및 텐서 계약

`BoxDecodeType`은 타입이 지정된 API(`simaai::neat::BoxDecodeType` / `neat.BoxDecodeType`)이며, 디코딩 단계에서는 항상 명시적으로 설정해야 합니다. 아래 런타임 계약은 `internals/gst_plugins/genericboxdecode_v2/gstneatboxdecode.cpp`(`infer_num_classes`, `infer_yolo_decoupled_classes`, `infer_yolo_packed_classes`, `compute_required_output_size`)에서 제공됩니다.

핵심 텐서 계약 규칙:
- `yolov5` 감지를 제외한 YOLO 계열 디코딩 타입(`yolo`, `yolov5-seg`, `yolov7*`, `yolov8*`, `yolov9*`, `yolov10*`):
  - 분리된 헤드: 클래스 헤드의 깊이는 반복 가능해야 하며 `> 4`여야 합니다.
  - 패킹된 헤드: 각 헤드의 깊이는 `depth = 3 * (num_classes + 5)`를 만족해야 하며 헤드 간에 일관성을 유지해야 합니다.
- `yolov5` 감지: stride 8/16/32 형상과 `3 * (num_classes + 5)` 깊이를 가진 디코딩되지 않은 P3/P4/P5 패킹 헤드가 정확히 3개 필요합니다.
- `yolo26`: 4채널의 원시 l/t/r/b 바운딩 박스 텐서와 반복 가능한 클래스 헤드 깊이 `> 4`를 갖는 분리된 그룹 헤드입니다.
- `detr`: 클래스 채널은 헤드 간의 최대 깊이에서 추론되며, `> 4`여야 합니다.
- 기타 YOLO가 아닌 디코딩 타입(`effdet`, `rcnn-stage1`, `centernet`): 폴백 클래스 추론은 최대 깊이를 사용하며 `> 4`가 필요합니다.
- 세그멘테이션 디코딩 토큰(`*-seg`)은 v2에서 세그멘테이션과 유사한 출력 크기를 활성화합니다(각 감지 결과에 마스크 페이로드를 추가합니다).

| API 열거형 | 백엔드 토큰 | 예상되는 계약 |
|---|---|---|
| `BoxDecodeType::Yolo` | `yolo` | YOLO 분리형 또는 패킹된 깊이 계약 |
| `BoxDecodeType::YoloV5` | `yolov5` | 디코딩되지 않은 P3/P4/P5 패킹 헤드 3개 |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLO 깊이 계약 + 분할 경로 |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLO 분리형 또는 패킹된 깊이 계약 |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLO 깊이 계약 + 분할 경로 |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLO 분리형 또는 패킹된 깊이 계약 |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLO 깊이 계약 + 분할 경로 |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLO 분리형 또는 패킹된 깊이 계약 |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLO 분리형 또는 패킹된 깊이 계약 |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLO 깊이 계약 + 분할 경로 |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLO 분리형 또는 패킹된 깊이 계약 |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLO 깊이 계약 + 분할 경로 |
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 그룹화된 원시 l/t/r/b 경계 상자 헤드 + 클래스-점수 헤드 |
| `BoxDecodeType::Detr` | `detr` | `num_classes = max(depth)` (반드시 `> 4`여야 함) |
| `BoxDecodeType::EffDet` | `effdet` | 폴백 최대 깊이 추론 (`> 4`) |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | 폴백 최대 깊이 추론 (`> 4`) |
| `BoxDecodeType::Centernet` | `centernet` | 폴백 최대 깊이 추론 (`> 4`) |

빠른 실패 동작:
- `stages::BoxDecodeOptions`는 디코딩 유형으로 명시적으로 구성해야 합니다.
- `stages::BoxDecode(...)` 및 `nodes::SimaBoxDecode(...)`는 `BoxDecodeType::Unspecified`에서 빠르게 실패합니다.

디코딩 유형을 명시적으로 설정:

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.5;
opt.top_k = 100;
```

```python
opt = neat.ModelOptions()
opt.decode_type = neat.BoxDecodeType.YoloV8
```

## Run

**Python** 및 **C++ (사전 빌드)** 명령을 **Neat 설치 루트** ( `share/` 및 `lib/`가 포함된 디렉터리)에서 실행하고, **소스에서 빌드** 명령은 **리포지토리 루트**에서 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/007_read_detection_boxes/read_detection_boxes.py \
  --model /tmp/yolo_v8s.tar.gz --width 640 --height 640
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_007_read_detection_boxes
./build/tutorials-standalone/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

예상 출력 (경계 상자 수는 프레임에 따라 다르며, 합성 프레임은 0을 반환합니다):

```text
boxes=0
[OK] 007_read_detection_boxes
```

(Python 빌드는 `detections=...`를 출력하거나, 런타임에서 BoxDecode를 `model.run`에 연결하지 않으면 `raw_output_heads=...`를 출력합니다.) 이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt`를 사용하여 자신의 프로젝트에 통합하려면 (추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/007_read_detection_boxes/read_detection_boxes.cpp`
- Python: `tutorials/007_read_detection_boxes/read_detection_boxes.py`
