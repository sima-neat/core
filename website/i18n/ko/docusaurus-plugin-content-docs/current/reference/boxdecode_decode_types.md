---
title: "BoxDecode 디코딩 유형"
description: "객체 감지 후처리 작업에 적합한 BoxDecodeType을 선택하세요."
sidebar_position: 6
---

# BoxDecode 디코딩 유형

`nodes::SimaBoxDecode`는 원시 감지 헤드 텐서를 감지 결과로 변환합니다. 모델 추론 후 선택한 모델 제품군의 디코딩 연산을 적용하고 감지를 필터링합니다. NMS와 출력 인코딩은 선택한 제품군에 따라 다릅니다. 감지 모델은 페이로드를 박스로 파싱할 수 있으며, 자세 및 분할 모델은 박스 뒤의 키포인트나 마스크도 파싱할 수 있습니다.

일반적인 모델 패키지 사용의 경우, `Model`을 인식하는 생성자를 사용하는 것이 좋습니다. 모델 아카이브는 디코더에 필요한 텐서 순서, 레이아웃, 양자화, 클래스 수, 크기 조정 메타데이터 및 점수 범위 힌트를 제공합니다. 일반적으로 애플리케이션은 디코딩 제품군과 필터링 임계값만 선택합니다.

## 런타임 백엔드

BoxDecode와 SuperPoint는 검증된 Kernels 실행 파일과 함께 `neatkernelrunner`를
사용합니다. 기존 감지 및 특징 출력 형식은 바뀌지 않습니다. YOLO26 회전 박스 디코딩은 별도의 버전 지정 형식을 사용합니다. 텐서를 직접 입력할
때는 디코딩이 A65에서 실행되더라도 공유 CMA 텐서(`TensorMemory::EV74`)를 사용합니다.
이 메모리 선택은 실행 타깃이 아니라 DMA-BUF 전송을 제공합니다.

RCNN 단계 1은 Kernels가 소유한 수준별 top-K 계약을 사용합니다. 전역 NMS 후
모델 영역 수준별로 최대 `top_k`개의 생존 항목을 선택합니다. 출력은 일반 BBOX
형식이 아니라 제안 와이어 형식입니다. 네 그룹 각각에 INT32 개수와 `top_k`개의
6-FP16 레코드(`score, 0, x1, y1, x2, y2`)가 있습니다. 좌표 변환은 최종 직렬화까지
소수 지오메트리를 보존합니다. 단계 2의 제안 입력으로 사용하며
`decode_bbox_tensor()`에 전달하지 않습니다. 레거시 플러그인은 로드하지 않습니다.

패딩된 인터리브 INT8, BF16 및 FP32 SuperPoint 헤드는 명시적 픽셀 스트라이드를
사용하는 네이티브 밀집 리더를 사용합니다. Cast 단계나 구체화 없이 검출기 리덕션과
디스크립터 샘플링에서 패딩을 제외합니다. INT8 포트는 각자 작성된 양자화를
유지합니다. 네이티브 MLA 타일 헤드는 선언된 물리 레이아웃을 유지하며, 밀집 스칼라를
MLA 바이트 평면으로 다시 표시하지 않습니다.

## 빠른 시작

```cpp
using namespace simaai::neat;

Model model("/path/to/yolov8_model.tar.gz");

auto boxdecode = nodes::SimaBoxDecode(
    model,
    BoxDecodeType::YoloV8,
    /* detection_threshold */ 0.25,
    /* nms_iou_threshold */ 0.45,
    /* top_k */ 100);
```

독립적인 스테이지 사용 시:

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.45;
opt.top_k = 100;
```

## 인수

| 인자 | 의미 |
| --- | --- |
| `decode_type` | 모델 패밀리/헤드 형식(예: `BoxDecodeType::YoloV8` 또는 `BoxDecodeType::YoloX`). 필수 항목입니다. |
| `detection_threshold` | 감지 결과를 유지하기 위해 필요한 최소 점수입니다. `0.25`와 같이 모델에 적합한 값을 사용하십시오. |
| `nms_iou_threshold` | 비최대 억제(non-maximum suppression)에 사용되는 IoU 임계값입니다. `0`이면 모델에 패키징된 값을 유지합니다. 패키징된 값이 없으면 YOLO26은 NMS를 건너뛰고, 다른 디코더는 겹치는 모든 박스를 억제합니다. `0.45`와 같이 모델에 맞는 값을 설정하세요. |
| `top_k` | 최대 출력 개수. 원시 바운딩 박스 디코딩에는 명시적인 양의 상한이 필요합니다. 모델에 연결된 노드에서 `0`은 해석된 Model 상한을 상속합니다. SuperPoint는 `0`에 프로파일 기본값을 사용합니다. |
| `original_width`, `original_height` | 원본 이미지 크기(raw-geometry 생성자를 사용할 때 좌표 매핑에 사용). |
| `model_width`, `model_height` | 모델 입력 크기 재정의. 다음을 사용하여 `Model` 생성자는 패키징된 텐서 계약이 아닌 공간 디코딩 관련 설정을 변경합니다. |
| `resize_mode_override` | 업스트림 `Preproc` 단계에서 크기 조정 메타데이터를 기록하지 않고, 크기 조정/레터박스/자르기 동작을 명시적으로 지정해야 할 때만 사용하십시오. |
| `decode_type_option` | 고급 하위 레이아웃 선택기입니다. 내보낸 헤드 레이아웃을 알고 있지 않다면 모델 팩을 사용할 때 `Auto`로 설정하십시오. |

## 입력 및 출력

**입력:** 모델에서 생성된 원시 감지 텐서입니다. 예상되는 텐서의 형태는 모델 종류에 따라 다릅니다. MPK/모델 아카이브를 사용하면 Neat이 패키지된 계약에서 해당 세부 정보를 읽어옵니다.

**출력:** 디코딩된 감지를 포함하는 하나의 BoxDecode 텐서입니다. 감지 모델은 표준 `BBOX` 페이로드를 사용합니다. 자세 및 분할 모델은 동일한 초기 박스를 유지하고 작업별 페이로드를 추가합니다.

| 모델 작업 | C++ 헬퍼 | Python 헬퍼 | 디코딩된 텐서 |
| --- | --- | --- | --- |
| 감지 | `decode_bbox(...)` | `pyneat.decode_bbox(...)` | `[N, 6]` float32 박스: `x1, y1, x2, y2, score, class_id` |
| 회전 객체 감지 | `decode_obb(...)` | `pyneat.decode_obb(...)` | 소유권이 있는 기하 정보 `[N, 5]` float32 (`cx, cy, width, height, angle_radians`), 점수 `[N]` float32, 클래스 ID `[N]` int32 |
| 자세 | `decode_pose(...)` | `pyneat.decode_pose(...)` | 박스 `[N, 6]` 및 키포인트 `[N, 17, 3]` float32: `x, y, visibility` |
| 분할 | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | 박스 `[N, 6]` float32 및 마스크 `[N, 160, 160]` uint8 |
| 분할 + 포즈 | `decode_segmentation_pose(...)` | `pyneat.decode_segmentation_pose(...)` | 박스 `[N, 6]` float32, 마스크 `[N, 160, 160]` uint8, 키포인트 `[N, 17, 3]` float32 |
| Segment Anything | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | 박스 `[N, 6]` float32와 원본 이미지 크기의 마스크 `[N, H, W]` uint8(프롬프트 디코더에서는 N이 1) |
| SuperPoint | `decode_superpoint(...)` | `pyneat.decode_superpoint(...)` | 키포인트 `[N,2]`, 점수 `[N]`, 디스크립터 `[N,D]` |

축 정렬 감지 표시 그래프는 결과를 `SimaRender`에 전달할 수 있습니다. `decode_bbox(...)`는 기존 박스 선두 형식에만 사용하십시오. 이 함수와 `SimaRender`는 회전 박스를 지원하지 않습니다.

## 레터박스 배치 {#letterbox-placement}

`LetterboxPlacement::TopLeft` 같은 가운데가 아닌 레터박스([Preproc 크기 조정](/reference/nodes/preproc#resize-and-aspect-ratio) 참조)는 검출 프로필이 지원하며, 검출 프로필은 각 프레임의 모델-원본 아핀 변환으로 박스를 변환합니다. 대상은 YOLO 검출 계열, 디코딩된 YOLO, EfficientDet, DETR, DetrFocal, CenterNet, SSD입니다. 포즈, 세그멘테이션, 91개 클래스 다중 클래스 SSD는 설치된 Kernels 릴리스가 해당 변환을 받으면([자르기 좌표 매핑](#crop-coordinate-mapping) 참조) 같은 방식으로 처리하고, 그렇지 않은 경우와 RCNN 제안은 어긋난 결과를 반환하는 대신 이를 거부합니다. 가운데 배치 레터박스는 기존 변환을 유지합니다. Segment Anything 유형에는 별도의 요구 사항이 있습니다. [Segment Anything 규칙](#segment-anything-rules)을 참조하세요.

## 기존 DETR {#classic-detr}

동일한 쿼리 격자의 4채널 박스 로짓 헤드와 마지막에 배경 채널이 있는 클래스 로짓 헤드에는 `Detr`를 선택합니다. `decode_type_option`은 `Auto`로 유지하세요. 레시피가 텐서 이름과 무관하게 Softmax를 선택합니다. 배경은 분모에는 포함되지만 전경 클래스 수와 출력 선택에서는 제외됩니다. 박스 좌표에는 sigmoid가 적용됩니다. 이미 정규화된 focal-loss 박스를 위한 디코더는 아닙니다.

Stretch와 양수 `top_k`를 사용하세요. 디코더는 NMS를 적용하지 않으며 전처리 메타데이터로 원본 프레임 좌표를 복원합니다. 확률값 레이아웃이나 충돌하는 점수 활성화 설정은 거부됩니다.

```python
options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.Detr
options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
options.top_k = 100
options.score_threshold = 0.7
```

## Focal loss 쿼리 검출기

정규화된 `cx,cy,width,height` 좌표(깊이4), 독립 클래스 로짓(깊이N) 순서의 두 텐서에는 `BoxDecodeType::DetrFocal`(Python: `pyneat.BoxDecodeType.DetrFocal`, 토큰 `detr-focal`)을 사용합니다. 쿼리 축과 데이터 형식이 일치해야 하며 N은1–1024입니다. FP32, BF16 및 명시된 양자화 정보를 가진 INT8 저장 형식을 지원합니다.

클래스 로짓에만 sigmoid를 적용하고 전체 쿼리/클래스 쌍에서 상위K개를 선택합니다. 한 쿼리에서 여러 클래스를 출력할 수 있습니다. 배경 클래스, 박스 sigmoid, 쿼리별 argmax, NMS 또는 작은 면적 필터는 없습니다. 양수 `top_k`, 필터 없는 top-K를 위한 `detection_threshold=0`, 필수 값 `nms_iou_threshold=0`을 설정합니다. Stretch와 실제 이미지/모델 크기를 사용하세요.

기존 `BBOX` 출력은 박스를 이미지 경계로 잘라내고 XYWH 좌표를 정수로 반올림합니다. `decode_bbox`는 이 값을 실수로 제공하지만 서브픽셀 정밀도를 복원하지 않습니다. 잘라내지 않은 부동소수점 기준과 비교할 때 이 표현의 정확도 영향을 별도로 측정하세요. 기존 `BoxDecodeType::Detr`는 변경되지 않으며, 이 텐서가 아닌 박스 로짓과 배경 클래스를 포함한 softmax 점수를 요구합니다.

Python에서 `decode_bbox(..., strict=True)`를 사용하면 기본 퇴화 박스 필터 대신 반올림으로 면적이0이 된 행도 유지합니다.

Python raw-geometry 오버로드는 헤드 텐서를 직접 제공할 때 C++ 팩토리와 같은 경로를 사용합니다. 조밀한 FP32 헤드는 모델의 전체 출력 어댑터를 유지하고 실제 스토리지를 명시하세요. 원시 MLA 출력에서 중단하거나 패딩된 바이트를 조밀한 텐서로 다시 표시하지 마세요.

```python
node = pyneat.nodes.sima_box_decode(
    pyneat.BoxDecodeType.DetrFocal,
    original_width=1280, original_height=720,
    model_width=640, model_height=640,
    detection_threshold=0.0, nms_iou_threshold=0.0, top_k=300,
    decode_type_option=pyneat.BoxDecodeTypeOption.InterleavedByHeadLogit,
    source_storage=pyneat.BoxDecodeSourceStorage.DenseHwc,
    detess=False, dequant=False,
    resize_mode=pyneat.ResizeMode.Stretch,
)
```

유한한 뒤집힌 크기는 기존 끝점 클리핑으로 면적이 0인 BBOX 레코드가 되며 선택된 쌍과 점수를 유지합니다. 비유한 좌표나 로짓은 게시 전에 실패합니다.

## Segment Anything 마스크 {#segment-anything-masks}

Segment Anything 모델마다 전용 디코드 타입이 있습니다. 프롬프트 디코더에서 BoxDecode는 디코더 한 번의 실행을 하나의 검출로 바꿉니다. 결과는 원본 이미지 크기의 선택된 마스크, 그 마스크를 감싸는 박스, 예측된 IoU입니다. EfficientSAM3 개념 이미지 모델에서는 한 번의 실행을 각각 자체 마스크를 가진 최대 `top_k`개의 검출로 바꿉니다.

| 모델 | `decode_type`(토큰) | 이미지를 받는 Model의 전처리 | 디코더 옵션 |
| --- | --- | --- | --- |
| MobileSAM | `MobileSam`(`mobilesam`) | 인코더: `resize.mode=Letterbox`, `resize.placement=TopLeft`, 1024×1024, `pad_value=116`; `preset=ImageNet` | `multimask_output` |
| EdgeTAM | 마스크에는 `EdgeTamSeg`(`edgetam-seg`), 추적에는 `EdgeTamTrack`(`edgetam-track`) | 이미지 인코더: `resize.mode=Stretch`, 1024×1024, 패딩 없음; `preset=ImageNet` | `multimask_output` |
| EfficientSAM-T / -S | `EfficientSam`(`efficientsam`) | 인코더: `resize.mode=Stretch`, 1024×1024, 패딩 없음; 프리셋 없음, `normalize.enable=On`(값은 0–1, 정규화는 인코더가 직접 수행) | `multimask_output`은 true로 유지 |
| EfficientSAM3 대화형 | `EfficientSam3`(`efficientsam3`) | 이미지 인코더: `resize.mode=Stretch`, 1008×1008, 패딩 없음; `normalize.enable=On`과 `normalize.mean`, `normalize.stddev` (0.5, 0.5, 0.5) | `multimask_output` |
| EfficientSAM3 개념 | `EfficientSam3`(`efficientsam3`) | 개념 이미지 모델 자체(텐서 경로): 애플리케이션이 이미지를 1008×1008로 늘리고 [−1, 1]로 스케일링한 뒤 `resize_mode="stretch"`인 `semantic.preprocess`를 붙입니다 | `top_k`, `score_threshold` |

Python도 같은 이름을 사용합니다: `pyneat.BoxDecodeType.MobileSam`, `EdgeTamSeg`, `EdgeTamTrack`, `EfficientSam`, `EfficientSam3`. MobileSAM은 정규화한 뒤 0으로 패딩하며, 이는 ImageNet 평균 픽셀(124, 116, 104)에 해당합니다. `pad_value=116`은 이를 근사합니다. 마스크는 패딩 영역을 덮지 않습니다. EfficientSAM3 Model은 출력이 검출 `[Q, 6]`과 Q 채널 마스크 로짓이면 개념 이미지 모델로, 그렇지 않으면 프롬프트 디코더로 처리됩니다. 각 모델의 전체 워크플로는 [sima-neat/models](https://github.com/sima-neat/models)에 있는 모델별 README에 나와 있습니다. 예를 들어 [EdgeTAM README](https://github.com/sima-neat/models/tree/develop/models/edgetam)를 참조하세요.

`Model::Options::multimask_output`(Python: `ModelOptions.multimask_output`)은 업스트림 `predict(multimask_output=...)`에 대응합니다. 모호한 클릭 한 번에는 `true`로 두고, 박스나 여러 점에는 `false`로 설정하세요.

| `decode_type` | `multimask_output=true`(기본값) | `multimask_output=false` |
| --- | --- | --- |
| `MobileSam` | 마스크 1–3 중 예측 IoU가 가장 높은 것 | 마스크 0 |
| `EdgeTamSeg`, `EdgeTamTrack` | 마스크 1–3 중 예측 IoU가 가장 높은 것 | 마스크 0이 안정적이면 마스크 0, 아니면 마스크 1–3 중 최상 |
| `EfficientSam` | 마스크 0–2 중 예측 IoU가 가장 높은 것 | 거부: EfficientSAM은 단일 마스크 토큰을 내보내지 않습니다 |
| `EfficientSam3` 프롬프트 디코더 | 4개면 마스크 1–3, 3개면 마스크 0–2 중 예측 IoU가 가장 높은 것 | 4개: EdgeTAM과 같음. 3개: 거부 |
| `EfficientSam3` 개념 | `score_threshold`를 넘는 최대 `top_k`개의 쿼리 | 거부: 개념 검출에는 멀티마스크 선택이 없습니다 |

- SAM2와 같이, 마스크 0의 로짓 중 −0.05를 넘는 것의 98% 이상이 0.05도 넘으면 마스크 0을 안정적이라고 봅니다.
- 마스크 채널 하나로 내보낸 디코더는 어느 값이든 그 마스크를 사용합니다.
- 다른 디코드 타입은 `multimask_output=false`를 거부합니다. 이 선택은 `Model`마다 하며, 두 디코더 Model이 같은 MPK를 사용할 수 있습니다.

### 각 출력의 의미 {#segment-anything-outputs}

BoxDecode 텐서는 [버전이 지정된 마스크 페이로드](#versioned-masks-payload)를 사용합니다. `decode_segmentation`은 박스 `[N, 6]`과 마스크 `[N, H, W]`를 반환합니다.

- **박스.** 원본 이미지 픽셀 단위의 `x1, y1, x2, y2`입니다. 프롬프트 디코더의 박스는 마스크 픽셀을 감싸는 최소 박스입니다. `x2`와 `y2`는 마스크의 마지막 열과 행의 다음 위치이므로 `x2 - x1`은 마스크의 너비입니다. 개념 검출의 박스는 검출 헤드에서 가져와 원본 이미지에 맞게 스케일링하며 잘라내지 않습니다.
- **점수.** 선택된 마스크의 예측 IoU, 또는 개념 쿼리의 점수입니다.
- **클래스.** 선택된 마스크 채널(위 표의 인덱스), 또는 개념 쿼리에서는 0입니다.
- **마스크.** `uint8`이며, 원본 이미지 크기로 리샘플링한 로짓이 0보다 큰 위치는 255, 나머지는 0입니다.
- **개수.** 프롬프트 디코더에서는 항상 1입니다. 빈 마스크의 박스는 모두 0이지만 점수와 클래스는 유지됩니다. 업스트림과 같이 EdgeTAM의 객체 점수가 0 이하이면 마스크가 비워집니다. 개념 모델은 유지한 쿼리를 점수 내림차순(동점은 쿼리 인덱스 순)으로 반환합니다.

`EdgeTamSeg`와 다른 모든 Segment Anything 타입은 BoxDecode 텐서만 반환합니다. `EdgeTamTrack`을 사용하면 디코더 자체의 출력이 그 뒤에 이어집니다. [EdgeTAM 추적](#edgetam-tracking)을 참조하세요.

`SimaRender`는 이 마스크를 그리지 않습니다. 이 렌더러는 마스크가 프레임보다 작은 헤더 없는 분할 페이로드만 그립니다. Segment Anything 마스크는 `decode_segmentation` 결과로 그리세요.

### 규칙 {#segment-anything-rules}

- **디코더 출력.** 프롬프트 디코더는 지정한 모델의 마스크 수(1, 3 또는 4)만큼의 마스크 로짓, 마스크마다 하나의 예측 IoU, 그리고 EdgeTAM의 경우 객체 점수를 내보냅니다. INT8, BF16, FP32 출력을 허용합니다. 개념 이미지 모델은 검출 `[Q, 6]`과 쿼리별 마스크 로짓을 내보냅니다.
- **옵션.** 프롬프트 디코더: `top_k`는 0 또는 1로, `nms_iou_threshold`는 0으로 두세요. `score_threshold`는 1보다 작아야 하며 그 밖에는 무시됩니다. `num_classes`는 마스크 수 또는 0입니다. 개념 이미지 모델: `top_k`를 1..Q로 설정하세요.
- **이미지 기하 정보.** 캔버스, 원본 이미지 크기, 리사이즈는 디코더 입력의 `tensor.semantic.preprocess`에서 가져오며 표와 일치해야 합니다. 일치하지 않으면 빌드가 바꿔야 할 설정을 알려 주며 실패합니다. 인코더 출력을 `.cvu()`로 디코더에 전달하고, 프롬프트 디코더는 `build(inputs)` 또는 `execution_plan(inputs)`로 입력에서 빌드하세요. 개념 이미지 모델의 텐서 경로에서는 그 픽셀 텐서에 이 정보를 붙이세요.
- **한 번 리사이즈된 이미지 전체.** 리사이즈 전 크롭이나 ROI는 거부되며, 디코더가 재현하지 못하는 스케일 크기도 마찬가지입니다. MobileSAM은 단독 `Preproc` 노드가 아니라 이미지를 전처리하는 Model에서 `resize.placement=TopLeft`로 레터박스를 적용하세요. 이 방식은 Segment Anything과 같이 반올림합니다. 직접 붙이는 정보에는 아핀 변환을 포함하세요. `affine_m00` = 원본 너비 / 스케일된 너비, `affine_m11` = 원본 높이 / 스케일된 높이입니다.
- **러너당 이미지 크기 하나.** 마스크가 원본 이미지 크기를 가지므로, 빌드된 러너는 하나의 크기만 처리합니다. `Model::run`은 새 크기에 대해 다른 러너를 빌드합니다. 스트리밍 그래프는 크기가 다른 프레임을 거부합니다.

### 인코더와 디코더 {#segment-anything-example}

MobileSAM 예제입니다. 다른 모델에는 표에 있는 해당 모델의 인코더 설정과 디코드 타입을 사용하세요.

```python
encoder_options = pyneat.ModelOptions()
encoder_options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
encoder_options.preprocess.resize.placement = pyneat.LetterboxPlacement.TopLeft
encoder_options.preprocess.resize.pad_value = 116
encoder_options.preprocess.preset = pyneat.NormalizePreset.ImageNet
encoder = pyneat.Model("/path/to/mobile_sam_image_encoder_mpk.tar.gz", encoder_options)

options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.MobileSam
options.multimask_output = False  # a box prompt; keep True for one click
decoder = pyneat.Model("/path/to/mobile_sam_prompt_decoder_mpk.tar.gz", options)

embeddings = encoder.run([image])[0]
# .cvu() keeps the encoder's preprocess metadata on the embeddings.
outputs = decoder.run([embeddings.cvu(), point_encoding, point_types])
result = pyneat.decode_segmentation(outputs)[0]
box = result.boxes.to_numpy()[0]  # x1, y1, x2, y2, IoU, mask index
mask = result.masks.to_numpy()[0]  # uint8 [H, W], 0 or 255
```

### EdgeTAM 추적 {#edgetam-tracking}

EdgeTAM의 메모리 인코더와 메모리 뱅크에는 선택된 마스크의 저해상도 로짓과 그 객체 포인터가 필요하지만, BoxDecode는 이를 디코딩하지 않습니다. `EdgeTamTrack`은 BoxDecode 텐서를 반환한 뒤, 같은 실행의 디코더 출력을 같은 Model이 BoxDecode 없이 반환하는 것과 정확히 같게 반환합니다. 출력은 `mask_logits` `[1, 256, 256, 4]`, `iou_scores` `[1, 4]`, `object_score_logits` `[1, 1]`, `object_pointers` `[1, 4, 256]`이며 모두 float32입니다. 각 출력마다 프레임당 한 번의 변환이 듭니다. 첫 번째 텐서만 디코딩하세요. 원시 텐서에는 검출 태그가 붙지 않습니다. BoxDecode 없는 모델 경로가 텐서를 반환할 수 없으면 `EdgeTamTrack`은 거부됩니다.

- 검출의 클래스는 선택된 플레인입니다. 메모리 인코더에는 모델 README의 설명대로 1024×1024로 업샘플링한 `mask_logits[0, :, :, class]`를 사용하세요.
- 두 개 이상의 점으로 된 프롬프트에는 `multimask_output=False`를 설정하고, 클릭 한 번과 추적 중인 프레임에는 `True`로 두세요. 클래스는 어느 규칙이 선택한 플레인이든 알려 줍니다.
- 메모리 뱅크에는 `multimask_output=True`이면 `object_pointers[0, class]`를 사용하세요. `False`이면 `object_pointers[0, 0]`을 사용하세요. SAM 2는 안정성 폴백이 플레인 1–3의 마스크를 선택하더라도 단일 마스크 포인터를 유지합니다.
- 객체 점수가 0 이하이면 마스크와 그 박스가 비워지지만, 클래스는 여전히 선택된 플레인을 가리킵니다. 메모리 인코더에 전달하는 로짓에는 README의 객체 점수 규칙을 적용하세요.

```python
options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.EdgeTamTrack
decoder = pyneat.Model("/path/to/edgetam_mask_decoder_mpk.tar.gz", options)

outputs = decoder.run(decoder_inputs)
result = pyneat.decode_segmentation(outputs[:1])[0]
plane = int(result.boxes.to_numpy()[0, 5])
logits = outputs[1].to_numpy()[0, :, :, plane]  # [256, 256] low-resolution logits
pointer = outputs[4].to_numpy()[0, plane if options.multimask_output else 0]  # [256]
```

### EfficientSAM3 개념 {#efficientsam3-concept}

```python
options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.EfficientSam3
options.top_k = 20
options.score_threshold = 0.5
options.preprocess.kind = pyneat.InputKind.Tensor  # two inputs: pixels and packed text
model = pyneat.Model("/path/to/efficientsam3_image_model_mpk.tar.gz", options)

# Stretch to 1008x1008 and scale to [-1, 1], as SAM 3 does, and record that geometry.
stretched = cv2.resize(rgb, (1008, 1008), interpolation=cv2.INTER_LINEAR)
pixels = pyneat.Tensor.from_numpy(stretched.astype(np.float32) / 127.5 - 1, copy=True,
                                  layout=pyneat.TensorLayout.HWC)
facts = pyneat.PreprocessRuntimeMeta()
facts.resize_mode = "stretch"
facts.original_width, facts.original_height = rgb.shape[1], rgb.shape[0]
facts.resized_width = facts.resized_height = 1008
facts.affine_m00 = facts.original_width / 1008  # model-to-original scale
facts.affine_m11 = facts.original_height / 1008
semantic = pixels.semantic
semantic.preprocess = facts
pixels.semantic = semantic

outputs = model.run([pixels, packed_text])
result = pyneat.decode_segmentation(outputs)[0]
boxes = result.boxes.to_numpy()  # [N, 6]: x1, y1, x2, y2, score, 0
masks = result.masks.to_numpy()  # uint8 [N, H, W]
```

## YOLOv8 및 YOLO11 회전 바운딩 박스 {#yolov8-oriented-boxes}

`BoxDecodeType::YoloV8Obb`(Python: `pyneat.BoxDecodeType.YoloV8Obb`, 토큰 `yolov8-obb` 또는 `yolo11-obb`)를 선택합니다. INT8, BF16 또는 FP32 헤드 9개를 역할별로 묶어 전달합니다. 64채널 DFL 박스 헤드, 클래스 로짓 헤드, 각도 로짓 헤드를 각각 3개씩 스트라이드 8, 16, 32 순서로 배치합니다. 전체 클래스 수는 클래스 헤드에서 추론합니다.

BoxDecode는 DFL, 클래스 sigmoid, `(sigmoid(angle)-0.25)*pi`, `nms_iou_threshold`에 따른 클래스별 회전 Fast-NMS를 적용합니다. 입력은 원시 로짓이어야 합니다. 기하 정보와 `BBOX_OBB_V1` 출력은 아래 YOLO26 계약을 따르며 같은 `decode_obb` 헬퍼를 사용합니다.

## YOLO26 회전 박스 {#yolo26-oriented-boxes}

`BoxDecodeType::YoloV26Obb`(Python: `pyneat.BoxDecodeType.YoloV26Obb`, 토큰 `yolo26-obb`)를 명시적으로 선택하십시오. 이 프로파일은 스트라이드 8, 16, 32의 4채널 거리 헤드 3개, 클래스 로짓 헤드 3개, 1채널 원시 각도 헤드 3개를 역할별로 묶은 FP32, BF16 또는 INT8 헤드 9개를 입력으로 받습니다. 모든 헤드는 전체 클래스 수(1–256)가 일치해야 합니다. `Auto`, `GroupedByRole`, `GroupedByRoleLogit`는 이 그룹 레이아웃을 선택합니다. 확률값 클래스 헤드와 인터리브 레이아웃은 거부됩니다.

INT8 헤드에는 양의 유한한 텐서별 양자화 메타데이터와 INT8 영점이 필요합니다. BF16 및 FP32 헤드는 정수 양자화 메타데이터를 사용하지 않습니다.

양수 `top_k`와 `nms_iou_threshold = 0`을 사용하십시오. 클래스 로짓에 sigmoid를 적용하고 최대 활성화 점수로 앵커를 선택한 뒤 앵커/클래스 쌍을 선택합니다. 동점이면 첫 단계의 순위와 클래스 순서를 유지합니다. 점수 기준은 엄격한 부등식(`score > detection_threshold`)입니다. 원시 각도는 이미 라디안 단위이므로 디코더는 sigmoid를 적용하거나 범위를 정규화하지 않습니다.

전처리 메타데이터에는 모델 좌표에서 원본 좌표로의 명시적인 유한 변환이 필요합니다. X/Y 배율은 양수이며 같아야 하고 전단 변환이 없어야 합니다. 모델 크기의 타일에도 명시적인 항등 변환이 필요합니다. 변환이 없거나 비등방성이면 회전 사각형을 근사하지 않고 거부합니다. 부호가 있는 유한 너비/높이를 유지하며 이미지 경계에서 자르지 않습니다. 타일 기반 DOTA 평가에서는 각 타일에 항등 변환을 사용하고, 타일 오프셋 복원과 타일 간 다각형 억제는 애플리케이션에서 수행하십시오.

소스 구동 그래프에서 Core는 검증된 업스트림 Preproc 계약으로부터 초기 준비용 지오메트리를 도출합니다. 이는 준비용 초기값일 뿐이며, 모든 프레임에는 유한하고 양수인 균일 아핀 변환과 모델에 일치하는 캔버스 크기가 여전히 필요합니다. 프레임 변환이 없으면 오류가 발생하며 항등 변환으로 폴백하지 않습니다.

출력 태그는 `BBOX_OBB_V1`입니다. 32바이트 버전 지정 헤더 뒤에 32바이트 레코드가 `top_k`개 이어집니다. 사용하지 않는 레코드는 0입니다. `decode_obb_tensor(tensor)`는 단일 캐리어를 검증하고, `decode_obb(tensors)`는 캐리어마다 형식이 지정된 결과 하나를 반환합니다. 둘 다 소유권이 있는 CPU 텐서로 복사하며 클래스 ID는 INT32로 유지합니다. 예:

```python
result = pyneat.decode_obb(outputs)[0]
geometry = result.geometry.to_numpy()  # cx, cy, width, height, angle_radians
scores = result.scores.to_numpy()
class_ids = result.class_ids.to_numpy()
```

이 형식은 의도적으로 선두 BBOX의 확장이 아닙니다. `decode_bbox`, 분할/자세 파서 또는 축 정렬 렌더러 대신 형식이 지정된 헬퍼를 사용하십시오.

## 슈퍼포인트

SuperPoint는 BoxDecode 제품군의 일부로 남아 있지만, 박스인 척하는 대신 특징점을 출력합니다. 최소 A65 기본 구성은 다음과 같습니다.

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.descriptor_output_dtype = TensorDType::Float32;

auto decoder = nodes::SimaBoxDecode(model, options);
```

Python에서도 동일한 기본값을 사용합니다.

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.SuperPoint)
options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32

decoder = pyneat.nodes.sima_box_decode(model, options=options)
```

`A65V1`은 기본 프로필입니다. 모델이 다른 숫자 연산 방식을 필요로 할 때 다른 프로필을 명시적으로 선택하십시오. Neat은 텐서의 형태나 값으로부터 연산 방식을 추론하지 않습니다.

| 프로필 | 선택 시점 | 생산 상태 |
|---|---|---|
| `LightGlueV1` | LightGlue 호환 감지기, NMS, 좌표 및 디스크립터 동작 | 지원 |
| `MagicLeapDemoV1` | 고정된 Magic Leap 데모 동작 | 지원 |
| `A65V1` | 이전 A65 슈퍼포인트 디코더와의 호환성 | 지원; 기본값 |
| `PaperBicubicV1` | 향후 완전히 정의된 양방향 보간 정책을 위한 예약된 숫자 ID | 프로덕션에서 정의될 때까지 거부됨 |

수치 처리 방식과 출력 인코딩은 서로 독립적입니다. 예를 들어, 기본 V1 출력에서 A65 수치 처리 방식을 선택할 수 있습니다.

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::FeaturePointsV1;
```

기존 바이트 레이아웃은 선택적으로 사용할 수 있으며, 다음과 같은 추가적인 제약 조건이 있습니다.

```cpp
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::LegacyA65InterleavedV0;
options.superpoint.descriptor_output_dtype = TensorDType::Int8;
```

`SuperPointProfile::Auto`는 먼저 권한 있는 MPK `superpoint.profile` 메타데이터를 사용합니다. API(`Model::Options.superpoint.profile`) 또는 MPK에서 프로필을 제공하지 않으면 `A65V1`로 해결됩니다. Neat는 텐서 모양, 값, 파일 이름 또는 다운스트림 노드를 기반으로 프로필을 추측하지 않습니다.

공개 센티널 값이 변경되지 않은 상태로 유지되면 `detection_threshold=0.0`, `top_k=0`, `nms_radius=-1` 및 `border_margin=-1`은 선택한 프로필에서 값을 가져옵니다. `A65V1`은 임계값 `0.1`, Top-K `600`, NMS 반경 `4` 및 경계 마진 `0`으로 해결됩니다. LightGlueV1 및 MagicLeapDemoV1은 각각 임계값 `0.0005` 및 `0.015`를 사용하며, 둘 다 Top-K `600`, NMS 반경 `4` 및 경계 마진 `4`를 사용합니다.

`nms_iou_threshold`는 SuperPoint에 적용되지 않습니다. 대신 픽셀 반경 `superpoint.nms_radius`를 사용하십시오. 기본 출력은 버전이 지정된 `FEATURE_POINTS_V1` 구조화된 배열 페이로드입니다. `LegacyA65InterleavedV0`는 명시적인 마이그레이션 형식이며 256차원 INT8 디스크립터가 필요합니다. `decode_bbox` 또는 `BoxDecodeResults` 대신 `decode_superpoint`를 사용하십시오.

버전이 지정된 MPK `superpoint` 스키마 v1 레코드는 실패 시 안전하게 작동하도록 설계되었습니다. 프로필, 고유한 디텍터 및 디스크립터 텐서 ID, 64개의 16진수로 구성된 `sha256:` 핑거프린트 및 지원되는 입력 표현 `raw-logits-65` 및 `coarse-pre-l2`를 명시해야 합니다. 스키마 0은 마이그레이션/수동 레코드로서만 허용되며, 생략된 스키마 0 표현 필드는 해당 두 가지 원시 입력 표현으로 표준화되고 진단 정보에 기본값으로 기록됩니다. 알 수 없는 스키마 버전 또는 표현 토큰은 컴파일 계약을 위반합니다. API 프로필 재정의가 다른 MPK 프로필에 대해 스탬프된 핑거프린트와 충돌하는 경우, 선택한 프로필에 대해 MPK를 다시 스탬프하십시오. Neat는 해당 출처를 삭제하거나 재해석하지 않습니다.

## BBOX 와이어 페이로드

객체 감지 디코더는 입력 프레임당 하나의 텐서를 출력하며, 이 텐서에는 `BBOX` 태그가 지정됩니다. 해당 텐서는 1차원 `UInt8` 바이트 버퍼입니다.

| 필드 | 값 |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | `[N_bytes]`이며, 여기서 `N_bytes`는 모델 아카이브에서 가져온 압축된 버퍼의 용량입니다. |

텐서의 형태는 감지 횟수가 아닌 바이트 수를 나타냅니다. 페이로드는 리틀 엔디안 레이아웃을 사용합니다.

```text
offset  size  content
------  ----  -------
  0      4    uint32  N = valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   trailing bytes are padding and must be ignored
```

각 `RawBox` 레코드는 24바이트입니다.

| 오프셋 | 크기 | 유형 | 필드 | 의미 |
| --- | --- | --- | --- | --- |
| 0 | 4 | `int32` | `x` | 원본 이미지의 왼쪽 상단 x 좌표입니다. |
| 4 | 4 | `int32` | `y` | 원본 이미지의 왼쪽 상단 y 좌표. |
| 8 | 4 | `int32` | `w` | 원본 이미지의 픽셀 단위 너비입니다. |
| 12 | 4 | `int32` | `h` | 원본 이미지의 픽셀 단위 높이입니다. |
| 16 | 4 | `float32` | `score` | `[0.0, 1.0]`에서의 NMS 후 신뢰도. |
| 20 | 4 | `int32` | `class_id` | 모델에서 정의한 클래스 ID입니다. |

하나의 레코드에 대한 일치하는 Python `struct` 형식은 `"<iiiifi"`입니다.

업스트림 전처리 메타데이터가 있는 경우 좌표는 원본 이미지 픽셀 단위로 표시됩니다. ROI에서 전처리한 프레임의 원본 이미지는 ROI가 아니라 전체 소스 프레임입니다([자르기 좌표 매핑](#crop-coordinate-mapping) 참조). 좌표는 `[0, 1]`로 정규화되지 않으며 모델의 내부 레터박스 입력 공간으로 표현되지 않습니다.

### 분할 + 포즈 결합 페이로드

`yolox-seg-pose`는 세 영역을 담은 단일 버퍼를 출력합니다. 모든 영역은 동일한 슬롯 수 `top_k`를 기준으로 배치됩니다.

| 영역 | 오프셋 | 스트라이드 | 내용 |
| --- | --- | --- | --- |
| 헤더 | `0` | 4 | `int32` 검출 개수 |
| 박스 | `4` | 24 | 위에서 설명한 `BoundingBoxOut` 레코드 |
| 마스크 | `4 + 24*top_k` | `mask_w * mask_h` | `uint8`, 슬롯당 평면 하나 |
| 포즈 | `4 + (24 + mask_w*mask_h)*top_k` | 204 | 17 x `{uint32 x, uint32 y, float32 visibility}` |

`decode_segmentation_pose(...)`로 박스, 마스크, 키포인트를 가져옵니다. 각 텐서의 행 `i`는 동일한 검출을 나타냅니다. 박스만 필요하면 `decode_bbox(...)` 또는 `BoxDecodeResults(...)`를 사용하십시오.

결합 디코더는 레거시 `BBOX` 태그도 허용합니다. 마스크와 포즈의 동시 렌더링은
지원하지 않습니다. 결합 버퍼를 `poseyolo`에 전달하지 마세요. 해당 렌더러는 박스
바로 뒤에 키포인트가 있다고 가정하지만 결합 페이로드는 그 위치에 마스크를 저장합니다.

검출마다 키포인트 슬롯이 17개 있습니다. 백엔드는 사용하지 않는 슬롯과 `pose_classes`에서 제외된 클래스의 키포인트를 0으로 설정합니다. 헬퍼는 이 값을 그대로 복사합니다.

채널 0에 객체성이 있으므로 Core는 클래스 헤드 깊이에서 1을 빼 `num_classes`를 구합니다. 예를 들어 30채널이면 클래스가 29개입니다. 명시한 `num_classes`는 이 값과 일치해야 합니다.

### 키포인트 클래스

키포인트가 있는 클래스 ID를 설정하십시오.

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.YoloXSegPose)
options.yolox_seg_pose.pose_classes = [0, 5]
```

`ModelOptions.yolox_seg_pose`에도 같은 설정을 사용할 수 있습니다.

- 목록에 없는 클래스는 키포인트 좌표와 가시성이 0이 됩니다.
- `BoxDecodeOptions` 목록이 비어 있으면 모델 설정을 상속합니다. 목록을 설정하지 않으면 모든 클래스에 키포인트가 생성됩니다.
- ID는 중복 없이 `[0, num_classes)` 범위에 있어야 합니다. 이 옵션은 `YoloXSegPose`에서만 지원됩니다.

### 버전이 지정된 마스크 페이로드 {#versioned-masks-payload}

Segment Anything 출력처럼 160×160이 아닌 마스크는 앞부분의 개수 대신 버전이 지정된 헤더를 사용합니다. 레이아웃은 `pipeline/DetectionTypes.h`의 [`SegmentationMasksHeaderV1`](/reference/cppapi/structs/simaai-neat-segmentationmasksheaderv1)이며, 헤더 뒤에 `RawBox` 레코드가 오고, 그 뒤에 레코드마다 하나의 `uint8` 마스크(값 0 또는 255)가 옵니다. `decode_segmentation`으로 읽으세요. 이 함수는 박스와 마스크를 반환하고 잘못된 헤더나 범위는 거부합니다. 텐서에는 `BBOX` 태그가 그대로 유지되며, `decode_segmentation`과 `decode_bbox`는 매직 값으로 페이로드를 인식하므로 기존 분할 페이로드는 변경되지 않습니다. `SimaRender`는 이 페이로드를 받지 않습니다.

## `model.run`이 원본 헤더를 반환할 때

일부 모델 경로는 디코딩된 `BBOX` 텐서 대신 `model.run(...)`에서 원시 특징 맵 헤드를 반환합니다. 이는 실패한 실행이 아닙니다. 모델이 실행되었지만 해당 경로에 출력을 읽는 시점에 BoxDecode가 포함되지 않았다는 의미입니다.

다음 규칙을 사용하십시오.

- `detections=...` 또는 `BBOX` 텐서: 압축된 BBOX 페이로드를 파싱하거나 사용합니다.
  디코딩 도우미 함수.
- `raw_output_heads=...`: BoxDecode 단계를 추가하거나, 모델 경로를 검사하거나.
  모델별 후처리를 통해 원시 텐서를 처리합니다.

원시 헤드를 박스로 파싱하지 마십시오. 원시 텐서 레이아웃은 내보낸 모델 패밀리와 모델 아카이브 계약에 따라 달라집니다.

## 계약 변경

MPK는 텐서 계약과 디코딩 계열 정보를 제공합니다. 필터링 제어는 `ModelOptions`에
설정합니다. `SimaBoxDecode(Model, ...)` 노드에서는 양수 필터링 인수가 해석된 Model
제어를 재정의하고, 0 인수는 이를 상속합니다. `ModelOptions.score_threshold = 0.0`은
패키징된 임계값을 복원하라는 요청이 아니라 실제 0 점수 기준입니다.

| 런타임 인수 | 전달한 값 | 동작 |
| --- | --- | --- |
| `decode_type` | 비어 있음 / `Unspecified` | 지원되는 경우 모델 아카이브나 경로 계획기의 추론을 유지합니다. |
| `decode_type` | 구체적 유형 | 이 실행의 디코딩 계열을 재정의합니다. |
| `original_width` / `original_height` | `0` | 패키징된 크기나 상위 전처리 메타데이터를 유지합니다. |
| `original_width` / `original_height` | 양의 정수 | 좌표 매핑의 소스 크기를 재정의합니다. |
| 모델에 연결된 노드의 `detection_threshold` | `0.0` | 해석된 Model 점수 기준을 상속합니다. |
| 모델에 연결된 노드의 `detection_threshold` | `> 0.0` | 점수 기준을 재정의합니다. |
| 모델에 연결된 노드의 `nms_iou_threshold` | `0.0` | 해석된 Model NMS IoU를 상속합니다. |
| 모델에 연결된 노드의 `nms_iou_threshold` | `> 0.0` | NMS IoU를 재정의합니다. |
| 모델에 연결된 노드의 `top_k` | `0` | 해석된 Model 출력 상한을 상속합니다. |
| 모델에 연결된 노드의 `top_k` | `> 0` | 유지할 최대 검출 수를 재정의합니다. |

바운딩 박스 디코딩에는 양수로 해석된 `top_k`가 필요합니다. 0은 무제한 출력 저장소를
요청하지 않습니다. 연결된 `Model`이 없는 원시 헤드에는 그래프 빌드 전에 `top_k`를
설정합니다. 모델에 연결된 노드는 자체 `top_k`를 0으로 두어 양의 Model 상한을
상속할 수 있습니다. 이러한 제어는 별도 결과 리더 헬퍼를 바꾸지 않습니다.
리더의 `top_k=0`은 이미 상한이 있는 출력에 추가 상한을 적용하지 않는다는 뜻입니다.
0 점수 기준은 0인 점수를 허용하지만 지오메트리 검증, 배경 제외, NMS 또는 출력
상한을 우회하지 않습니다. NMS IoU가 0이면 YOLO26은 NMS를 건너뜁니다. NMS가
활성화된 다른 YOLO 및 SSD 프로파일은 여전히 양자화된 IoU 억제를 적용합니다.
양의 YOLO26 임계값은 NMS를 활성화합니다.

`num_classes`는 호출자가 설정한 값과 MPK 텐서 계약에서 구한 클래스 수를 비교합니다.

| 모델 계열 | 설정한 `num_classes` | MPK에서 구한 `num_classes` | 동작 |
| --- | --- | --- | --- |
| 지원되는 모든 모델 | `0` | 추론 가능한 양수 값 | MPK에서 구한 클래스 수를 사용합니다. |
| 지원되는 모든 모델 | 양의 정수 | 같은 값 | 설정한 클래스 수를 사용합니다. |
| 클래스 분할이 모호한 모델 | 양의 정수 | 알 수 없음 | 설정한 클래스 수를 사용합니다. 분할된 단일 클래스 헤드를 신뢰성 있게 추론할 수 없을 때 필요합니다. |
| YOLOv5 또는 YOLO26 | 양의 정수 | 다른 값 | 파이프라인 구성 전에 실패하고 두 값을 보고합니다. 이 원시 헤드 레이아웃은 텐서 깊이에서 클래스 수를 구합니다. |
| SSD 또는 다른 YOLO26 이전 비포즈 YOLO 계열 | 양의 정수 | 다른 값 | 기존 계열별 명시적 재정의 동작을 적용합니다. 포즈 디코더와 SuperPoint는 계열별 규칙을 유지합니다. |

`detection_threshold`는 BoxDecode 노드/단계 생성자에서 사용하는 이름입니다. `ModelOptions.score_threshold`는 동일한 제어에 사용되는 모델 경로 옵션입니다.

## 디코딩 유형 매핑

| API 열거형 | 백엔드 토큰 | 일반적인 모델 제품군 |
| --- | --- | --- |
| `BoxDecodeType::Yolo` | `yolo` | 모델 픽셀 단위로 디코딩된 `cx,cy,w,h` |
| `BoxDecodeType::YoloV5` | `yolov5` | YOLOv5 감지 |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLOv5 분할 |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLOv7 감지 |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLOv7 분할 |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLOv8 감지 |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLOv8 분할 |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLOv8 자세 추정 |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLOv9 감지 |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLOv9 분할 |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLOv10 감지 |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLOv10 분할 |
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 감지 |
| `BoxDecodeType::YoloV26Pose` | `yolo26-pose` | YOLO26 자세 |
| `BoxDecodeType::YoloV26Seg` | `yolo26-seg` | YOLO26 분할 |
| `BoxDecodeType::YoloV26Obb` | `yolo26-obb` | YOLO26 회전 객체 감지 (`BBOX_OBB_V1`) |
| `BoxDecodeType::YoloV6` | `yolov6` | YOLOv6 감지 |
| `BoxDecodeType::YoloX` | `yolox` | YOLOX 감지 |
| `BoxDecodeType::YoloXSegPose` | `yolox-seg-pose` | 박스, 마스크 및 키포인트 헤드를 함께 포함하는 YOLOX 압축 내보내기 |
| `BoxDecodeType::Ssd` | `ssd` | 주문된 헤드 지오메트리에서 선택한 정확한 준비된 SSD300, SSD-Mobile-300, SSD-Mobile-320 또는 SSDlite-Mobile-320 계약 |
| `BoxDecodeType::SuperPoint` | `superpoint` | SuperPoint 검출기 및 특징점 후처리 |
| `BoxDecodeType::Detr` | `detr` | DETR 스타일의 트랜스포머 기반 객체 검출 |
| `BoxDecodeType::MobileSam` | `mobilesam` | MobileSAM 프롬프트 디코더(왼쪽 위 레터박스) |
| `BoxDecodeType::EdgeTamSeg` | `edgetam-seg` | EdgeTAM 마스크 디코더와 객체 점수(늘이기) |
| `BoxDecodeType::EdgeTamTrack` | `edgetam-track` | `EdgeTamSeg` 결과 뒤에 추적용 디코더 원시 출력 |
| `BoxDecodeType::EfficientSam` | `efficientsam` | EfficientSAM 프롬프트 디코더, 멀티마스크 토큰 3개(늘이기) |
| `BoxDecodeType::EfficientSam3` | `efficientsam3` | EfficientSAM3 프롬프트 디코더, 또는 유지한 쿼리마다 마스크 하나인 개념 이미지 모델(늘이기) |
| `BoxDecodeType::EffDet` | `effdet` | 효율적인 객체 감지 |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | 순서가 지정된 RPN 헤드 다섯 개; 전역 NMS 후 영역 수준별 top-K |
| `BoxDecodeType::Centernet` | `centernet` | CenterNet 감지 |

`BoxDecodeType::Unspecified`는 설정되지 않은 값이며, 런타임 전에 오류가 발생합니다. SSD 레시피 식별자는 내부 Core 계약이며(`ssd300-v1`, `ssd-mobile-300-v1`, `ssd-mobile-320-v1` 또는 `ssdlite-mobile-320-v1`), 다른 공개 디코딩 유형이나 백엔드 토큰이 아닙니다. Core는 로우어링 전에 이를 해결하고, 설치된 객체 디코더는 계속해서 지원되는 `ssd` 계열 토큰을 수신하고 이미 검증된 헤드 지오메트리에서 해당 고정 구현을 선택합니다.

## 올바른 유형을 선택하세요.

- 내보낸 헤드 계약에 따라 선택합니다. `YoloV6`는 원시 l/t/r/b 거리와 클래스 로짓을 요구합니다. 그래프 안에서 박스를 디코딩한 YOLOv6 패키지에는 `Yolo`를 사용합니다. `[bbox_0..2, class_prob_0..2]` 순서로 묶인 출력에는 `decode_type_option=GroupedByRoleProbability`와 학습된 클래스 수에 맞는 `num_classes`를 설정합니다.
- 경계 상자가 잘못 이동하거나 크기가 조정된 경우 이미지 크기 조정 정책을 확인하십시오. `resize_mode_override`는 그래프에 크기 조정 메타데이터를 기록하는 상위 `Preproc` 단계가 없을 때만 사용하십시오.
- 사용자 지정 모델 패키지를 제작하는 경우, 아카이브에 감지 헤드에 대한 정확한 정보(텐서 순서, 논리적 형태, 물리적 저장 방식, dtype/양자화, 점수 범위, 클래스 수, 슬라이싱된 출력 등)가 명확하게 설명되어 있는지 확인하십시오. 애플리케이션 코드는 이러한 세부 사항을 보정할 필요가 없습니다.

## 모양 및 레이아웃 지침

다양한 감지 모델은 서로 다른 헤드 레이아웃을 사용합니다. 일부 모델은 각 특징 맵 레벨에 대해 하나의 텐서를 사용하는 반면, 다른 모델은 박스, 객체성, 클래스, 키포인트 또는 마스크를 별도의 텐서로 분할합니다. 일부 모델의 출력은 밀집된 HWC 텐서이고, 다른 모델은 컴파일러/런타임에 의해 패킹되거나 분할됩니다.

모델 패키지 흐름의 경우, 이는 패키지된 계약에 의해 처리됩니다. 수동으로 연결된 텐서의 경우, 핵심 규칙은 다음과 같습니다. 내보낸 헤드 형식을 정확히 일치시켜야 합니다. 순위 또는 채널 수만 기준으로 디코딩 유형을 선택하지 마십시오.

고급 텐서 계약 규칙:

- `YoloV5` 검출을 제외한 YOLO 계열 디코딩 유형(`Yolo`, `YoloV7`,
  `YoloV8`, `YoloV9`, `YoloV10`, 분할/포즈 변형)은 모델 계열에 맞는 분리 헤드 또는 패킹 헤드를 요구합니다.
- 패킹 YOLO 헤드는 모든 특징 수준에서 클래스 수와 헤드 깊이를 일치시켜야 합니다.
- `YoloV5` 검출은 P3/P4/P5 순서의 디코딩되지 않은 패킹 헤드를 정확히 3개 받습니다.
  그리드는 stride 8/16/32 형상이며 각 논리 깊이는 `3 * (num_classes + 5)`여야 합니다.
  BoxDecode는 시그모이드, 그리드 및 stride 변환, 표준 YOLOv5 앵커를 적용합니다.
  (`{10,13},{16,30},{33,23}`; `{30,61},{62,45},{59,119}`;
  `{116,90},{156,198},{373,326}`). 사용자 지정 AutoAnchor 테이블과 디코딩된
  6텐서 박스/클래스 내보내기는 다른 계약을 사용해야 합니다.
- `YoloV26`은 그룹화된 원시 l/t/r/b 바운딩 박스 헤드와 클래스-점수 헤드를 사용합니다.
- `Ssd`는 일반적인 SSD 디코더가 아닙니다. 미리 준비된 **네 가지 프로필**만 처리합니다.
  컴파일 시점에 완전하고 순서가 지정된 로컬/구성 H/W/C 서명을 기준으로 합니다. 다른 헤드 세트 또는 순서를 사용하면 오류가 발생하며, 오류 메시지에는 관찰된 서명과 지원되는 서명이 함께 출력됩니다.
  - **SSD300**`dboxes300_coco`): 300x300 입력, 특징 맵
    `{38,19,10,5,3,1}`, 셀당 사전 확률 `{4,6,6,6,4,4}`, 신뢰도 채널 순서
    `class*A + anchor`, 클래스 차원을 기준으로 **소프트맥스** 함수를 적용한 클래스 점수 (배경은 인덱스 0에 포함).
  - **SSD-Mobile-300-v1** (`ssd_anchor_generator`): 300x300 입력, 특징
    맵 `{19,10,5,3,2,1}`, 셀당 사전 확률 `{3,6,6,6,6,6}`, 신뢰 채널
    순서 `anchor*C + class`, 클래스별 점수는 클래스별 **시그모이드** 함수를 통해 계산되며(배경은 무시됨).
  - **SSD-Mobile-320-v1** (`ssd_anchor_generator`): 320x320 입력, 특징
    맵 `{20,10,5,3,2,1}`, 셀당 사전 확률 `{3,6,6,6,6,6}`, 신뢰 채널
    순서 `anchor*C + class`, 클래스별 점수는 클래스별 **시그모이드** 함수를 통해 계산되며(배경은 무시됨).
  - **SSDlite-Mobile-320-v1** (TorchVision `DefaultBoxGenerator`): 320x320
    입력, 특징 맵 `{20,10,5,3,2,1}`, 모든 레벨에서 셀당 6개의 사전 값,
    위치 순서 `anchor*4 + {dx,dy,dw,dh}`, 신뢰도 순서
    `anchor*C + class` 및 배경을 포함한 모든 91개 클래스에 대한 **소프트맥스**를 통한 클래스 점수.

  모든 레시피는 그룹화된 레벨별 위치 헤드(깊이 = `4 * priors-per-cell`)와 클래스-신뢰도 헤드(깊이 = `num_classes * priors-per-cell`)를 쌍으로 사용하며, FasterRcnnBoxCoder 분산 스케일링(`scale_xy 0.1`, `scale_wh 0.2`) 및 **스트레치**(비등방성) 전처리 리사이징을 사용합니다. 점수 활성화는 레시피에 의해 고정됩니다(온디바이스 디코더와 일치). 역할별 그룹화된 레이아웃은 자동으로 선택되므로 `decode_type_option`을 `Auto`로 둡니다. 그룹화되지 않은 레이아웃 토큰은 거부됩니다.

  **모델 프레임은 헤드 지오메트리뿐만 아니라 프로필의 일부입니다.** SSD300-v1 및 SSD-Mobile-300-v1은 300x300이 필요하고, 두 개의 320-v1 프로필은 320x320이 필요합니다. 사전 테이블과 스트레치 역투영은 해당 프레임에서만 유효하므로, 해결된 전처리 리사이즈 대상 또는 다른 크기의 모델 차원 재정의는 빌드 시간에 거부됩니다.

  원시/독립적인 `SimaBoxDecode` 구성은 리사이즈 모드를 생성하지 않습니다. 상위 `Preproc` 메타데이터 요구 사항을 유지하거나, 명시적인 원시 오버로드를 사용하여 외부에서 수행된 `ResizeMode::Stretch`를 주장합니다. Letterbox 및 Crop은 거부됩니다.

  **`num_classes` 계약.** 인코딩된 클래스 수는 항상 신뢰도 헤드 깊이(`conf_depth / priors-per-cell`, 인덱스 0에 있는 배경 포함)에서 파생됩니다. SSD300-v1은 준비된 81-to-8 경로와 같은 연속적인 접두사 선택을 허용합니다. 다른 세 가지 프로필은 정확한 인코딩된 수를 요구합니다. 유효하지 않은 선택은 빌드 시간에 거부됩니다. 프로필 기본값을 사용하려면 설정하지 둡니다.

  **91개 클래스 MobileNet 내보내기.** 인코딩된 클래스가 정확히 91개이면 SSD-Mobile-300-v1(MobileNet V1 및 V2)과 SSDlite-Mobile-320-v1(TorchVision MobileNet V3)은 원본 모델과 같은 방식으로 디코딩됩니다. 모든 전경 앵커/클래스 쌍을 유지하고 NMS를 클래스별로 부동소수점으로 실행합니다(SSDlite는 NMS 전에 클래스당 최대 300개의 후보를 유지합니다). 출력은 동일한 BBOX 페이로드입니다. 다른 SSD 프로필과 클래스 수는 기존 동작을 유지합니다.
- `Detr`은 최대 헤드 깊이를 기반으로 클래스 채널을 추론하며, 유효한 값을 필요로 합니다.
  차원 클래스.
- `EffDet` 및 `Centernet`은 해당 모델 계열의 계약을 사용합니다.
  YOLO 디코딩 방식을 거치지 않고 경로를 설정합니다.
- `*-seg` 디코딩 유형은 박스 경계선을 표시하는 출력과 함께 특정 작업에 필요한 마스크 데이터를 생성합니다.

사용자 지정 모델 패키지가 전체 주문 서명 중 어느 것과도 일치하지 않으면 매처를 약화시키는 대신 새로운 명시적으로 지원되는 프로필을 준비하십시오.

## 파이썬 참고 사항

Python에서 모델 옵션을 구성할 때, 문자열 대신 사용할 수 있다면 형식화된 열거형을 사용하는 것이 좋습니다.

```python
opt = pyneat.ModelOptions()
opt.decode_type = pyneat.BoxDecodeType.YoloV8
```

모델 작업에 맞는 헬퍼를 사용하여 출력을 분석합니다.

```python
outputs = model.run([image])

boxes = pyneat.decode_bbox(outputs)[0].to_numpy()

pose = pyneat.decode_pose(outputs)[0]
pose_boxes = pose.boxes.to_numpy()
keypoints = pose.keypoints.to_numpy()

seg = pyneat.decode_segmentation(outputs)[0]
seg_boxes = seg.boxes.to_numpy()
masks = seg.masks.to_numpy()
```

## 업그레이드

이전 BoxDecode 옵션은 ABI 6을 도입했습니다. 현재 원본 연산 옵션은 ABI 7 (`libsima_neat.so.7`)이 필요합니다. 일치하는 헤더와 라이브러리로 C++ 앱, 플러그인, Python 바인딩을 다시 빌드하고 호환성 심볼릭 링크로 ABI 5 또는 6 바이너리를 로드하지 마세요.


### 결합 YOLOX 입력 허용 검사

`YoloXSegPose`는 BF16 및 INT8 헤드 또는 연속 저장소나 양의 겹치지 않는 스트라이드
저장소(평면 CHW 뷰 포함)의 네이티브 FP32 헤드를 허용합니다. FP32 주소와 스트라이드는
4바이트 정렬이어야 하며, 채널 블록 FP32와 UINT8은 거부합니다. 암묵적인 손실 FP32
변환은 추가하지 않습니다. 프로토타입에는 자체 dtype과 텐서별 INT8 양자화가 있습니다.
각 감지 헤드의 역할 간 dtype은 여전히 일치해야 합니다. Core는 첫 헤드의 dtype이나
양자화를 모든 포트에 적용하지 않고 포트별 정보를 보존하여 Kernels 허용 검사에 전달합니다.
세 감지 수준의 스트라이드는 두 축 모두 8, 16, 32입니다. 모델 너비와 높이는 32의
배수여야 하지만 같을 필요는 없습니다. 프로토타입은 160×160×32로 유지되며 모델
크기에 맞춰 조용히 리사이즈되지 않습니다.

중심에서 벗어난 패딩이나 자르기 좌표에는 전처리 메타데이터를 사용합니다. 아핀은
**모델에서 원본으로** 좌표를 매핑합니다. 이 결합 프로파일은 전단, 회전 및 슬롯이 둘 이상인 ROI 목록을
지원하지 않습니다. 자르기에는 준비 시점의 신뢰할 수 있는 아핀 메타데이터가 필요합니다.
이는 외부에서 전처리된 텐서의 디코더 계약이며, `Model` 전처리에 자르기 생성 지원을
추가하는 것은 아닙니다. 키포인트가 17개보다 적은 모델도 고정된 17슬롯 공개 와이어
형식을 게시하며 사용하지 않는 슬롯은 0으로 설정됩니다.


## 자르기 좌표 매핑 {#crop-coordinate-mapping}

모델 관리 Crop은 기존 DETR, DetrFocal 및 일반 YOLO 검출을 포함한 지원 프로파일에 실제 이미지 역변환을 전달합니다. 선택과 NMS는 모델 좌표에서 수행한 뒤 선택된 상자를 원본 이미지에 매핑하고 클리핑합니다. 변환 메타데이터가 없거나 충돌하면 명시적으로 실패합니다. RCNN 제안에는 자르기를 활성화하지 않습니다. 기존 회전 상자 프로파일은 여전히 동일한 축 배율을 요구하며 정수 크기 조정으로 배율이 달라질 수 있습니다.

관심 영역(ROI)을 잘라 모델 캔버스에 레터박스하거나 늘이는 Preproc는 프레임마다 모델 좌표를 전체 소스 프레임으로 되돌리는 아핀 변환을 게시합니다. 검출 프로필은 각 프레임을 해당 전처리가 게시한 아핀 변환으로 디코딩하므로, 단일 ROI, 가운데 또는 가운데가 아닌 레터박스, 늘이기, 전체 프레임 모두에서 박스가 소스 프레임 픽셀 단위로 나옵니다. 소스 프레임은 자르지 않은 이미지이며 ROI가 아닙니다. 전체 프레임 검출 결과도 소스 프레임 픽셀 단위로 보고됩니다. 이미지 크기만 게시하고 아핀 변환을 게시하지 않는 입력은 전체 프레임 매핑을 유지합니다.

BoxDecode 호출마다 ROI를 하나씩 전달하세요. ROI 목록 버퍼에는 용량 1을 사용하세요.

| 프로필 | 단일 ROI, 자르기 또는 늘이기 |
| --- | --- |
| YOLO26, YOLOv5, YOLOv6, YOLOv7, YOLOv8~YOLOv11, YOLOX, 디코딩된 YOLO, EfficientDet, CenterNet, DETR, DetrFocal, SSD(91개 클래스 다중 클래스 SSD 레시피 제외) | 소스 프레임 픽셀 단위의 박스 |
| YOLOX 세그멘테이션 및 포즈, 방향 박스 | 소스 프레임 픽셀 단위의 박스(방향 박스는 두 축의 배율이 같아야 함) |
| 포즈(YOLOv8, YOLO26), 세그멘테이션(YOLOv5, v7, v8, v9, v10, YOLO26), 91개 클래스 다중 클래스 SSD | 설치된 Kernels 릴리스가 해당 프로필의 변환을 받으면 소스 프레임 픽셀 단위의 박스와 키포인트, 그렇지 않으면 프로필 이름을 알리며 실패 |
| RCNN 제안 | 프로필 이름을 알리며 실패 |
| Segment Anything | 실패: 전체 이미지를 한 번만 크기 조정해야 함 |

세그멘테이션 마스크는 모델 공간에 남습니다. 각 160×160 마스크는 박스나 소스 프레임이 아니라 모델 캔버스 전체를 덮으며 모델 좌표의 박스로 잘립니다. 따라서 마스크는 ROI를 따라 이동하거나 확대되지 않지만 박스는 그렇게 됩니다. 마스크 픽셀을 소스 프레임에 매핑하려면 해당 프레임에 대해 Preproc가 게시한 것과 같은 아핀 변환을 사용하세요.

프로필의 커널이 변환을 받지 않으면(RCNN 제안은 항상, 포즈, 세그멘테이션, 다중 클래스 SSD는 변환보다 이전의 Kernels 릴리스에서) 프로필은 전체 프레임 좌표로 디코딩하는 대신 실패합니다. 전체 프레임은 계속 디코딩합니다. 이러한 프로필에 요청한 Crop 크기 조정은 그래프를 빌드할 때 실패합니다.
