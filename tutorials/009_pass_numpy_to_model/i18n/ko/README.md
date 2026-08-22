# 009 NumPy 배열을 모델에 전달

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | numpy, pytorch, tensor, io |

## Concept

Neat 텐서와 이미 가지고 있는 구조(NumPy 배열, PyTorch 텐서 또는 `cv::Mat`) 간에 데이터를 이동하여 레이아웃, 데이터 유형 및 복사 의미 체계를 제어합니다. 이렇게 하면 기존 추론 스택에 Neat을 쉽게 통합할 수 있습니다.

## Walkthrough

기존 추론 스택에 Neat을 통합하는 경우, 다음은 필요한 상호 운용 경계입니다. 호스트 데이터가 Neat `Tensor`로 변환되는 방식과 Neat `Tensor`가 다시 호스트 데이터로 변환되는 방식을 고려해야 합니다. 처음부터 제대로 처리하면 일반적인 통합 오류(잘못된 레이아웃, 암시적 dtype 변환, 두 영역 간의 예기치 않은 별칭)를 방지할 수 있습니다.

여기서 두 언어의 차이가 가장 두드러집니다. Python 사용자는 NumPy/PyTorch에서, C++ 사용자는 OpenCV에서 시작합니다. 변환 *개념*은 동일하지만 API 이름과 유형이 다르므로 아래의 언어별 설명이 중요합니다. 마지막에는 호스트 데이터를 Neat 텐서로 변환하고, 복사하지 않고도 페이로드를 검사하고, 소스 버퍼보다 오래 유지해도 안전한 소유된 복사본을 생성할 수 있습니다.

### 호스트 데이터를 텐서로 래핑 {#step-to-tensor}

첫 번째 단계는 이미 보유하고 있는 데이터를 Neat `Tensor`로 변환하는 것입니다. 이미지 레이아웃을 명시적으로 지정(`RGB`)하여 런타임이 바이트를 추측하는 대신 올바르게 해석하도록 합니다. `copy=True` (또는 C++의 CPU 메모리 선택)는 텐서가 자체 바이트를 소유할지 또는 소스를 참조할지 결정합니다. 소스 버퍼가 변경되거나 해제될 수 있는 경우 명시적 소유권이 안전한 기본값입니다.

**C++:** `simaai::neat::from_cv_mat(mat, ImageSpec::PixelFormat::RGB, TensorMemory::CPU)`는 `cv::Mat`을 CPU 기반 텐서로 래핑합니다.

**Python:** `pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)`는 HWC `uint8` NumPy 배열을 래핑합니다.

### 페이로드 검사 {#step-map-and-inspect}

데이터가 텐서가 되면 다시 읽을 수 있습니다. 이는 상호 운용의 절반에 해당합니다. 다운스트림으로 전달하기 전에 변환 중에 모양과 바이트가 손상되지 않았는지 확인합니다.

**C++:** `tensor.map_read()`는 원시 `data` 포인터와 `size_bytes`를 노출하는 `Mapping`을 반환합니다. 이는 텐서의 저장소에 대한 *뷰*이며 복사가 없으므로 예제에서 선행 바이트를 직접 체크섬할 수 있습니다.

**Python:** `tensor.to_numpy(copy=True)`는 텐서에서 NumPy 배열을 생성합니다. 예제에서는 HWC 레이아웃이 손상되지 않고 유지되었는지 확인하기 위해 `.shape`를 출력합니다.

### 복사본 소유 {#step-own-a-copy}

마지막으로 원본 소스 버퍼와 완전히 분리된 데이터를 생성합니다. 이렇게 하면 입력이 제거된 후에도 안전하게 유지할 수 있습니다. 이것이 장기간 사용할 소비자가 사용할 복사본입니다.

**C++:** `tensor.clone()`는 `cv::Mat`에서 가져온 것과 독립적으로 새로운 CPU 소유 저장소에 복사합니다.

**Python:** 동일한 아이디어가 PyTorch를 통해 보여집니다. `pyneat.Tensor.from_torch(t, copy=True, ...)` 및 `tensor.to_torch(copy=True)`를 사용하여 소유된 PyTorch 텐서를 통해 왕복 변환합니다. (`torch`가 설치되지 않은 경우 정상적으로 건너뜁니다.)

## Run

**Neat 설치 루트**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행합니다. **소스에서 빌드** 명령은 **리포지토리 루트**에서 실행합니다. 이 챕터에는 모델 아카이브가 필요하지 않습니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py \
  --width 128 --height 96
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_009_pass_numpy_to_model
./build/tutorials-standalone/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

예상 출력(C++):

```text
tensor_rank=3
tensor_bytes=36864
head_checksum=4342
clone_bytes=36864
[OK] 009_pass_numpy_to_model
```

예상 출력(Python, `torch`가 설치된 경우):

```text
numpy_roundtrip_shape=(96, 128, 3)
torch_roundtrip_shape=(96, 128, 3)
```

(`torch`가 없는 경우 Python 빌드는 torch 라인 대신 `torch_roundtrip_skipped=True`를 출력합니다.) 사용자 지정 `CMakeLists.txt`를 사용하여 이 챕터의 C++ 소스를 자체 프로젝트에 통합하려면(추가 폴더가 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

왕복 데모를 넘어서면 빠르게 참조할 수 있도록 요약된 상호 운용 인터페이스입니다.

### 변환 API

- NumPy: `pyneat.Tensor.from_numpy(array, copy=..., image_format=...)` (입력); `tensor.to_numpy(copy=...)` (출력).
- PyTorch: `pyneat.Tensor.from_torch(tensor, copy=..., image_format=...)` (입력); `tensor.to_torch(copy=...)` (출력).
- OpenCV (C++): `simaai::neat::from_cv_mat(mat, pixel_format, memory)` (입력); 제로 복사 뷰를 위한 `tensor.map_read()`; 소유된 복사본을 위한 `tensor.clone()`.

### 복사 vs 뷰

- `copy=True` (Python) / `clone()` (C++)는 소스에서 분리된 데이터를 제공합니다. 소스가 해제되거나 변경된 후에도 안전하게 유지할 수 있습니다.
- `copy=False` / `map_read()`는 소스를 참조하는 뷰를 제공합니다. 비용이 저렴하지만 소스가 유지되고 변경되지 않은 상태일 때만 유효합니다.

### 레이아웃 및 dtype

- 이미지 데이터에 대해 항상 명시적인 `image_format` / `PixelFormat`를 전달하여 레이아웃이 해석되도록 하고 추측하지 않도록 합니다.
- Neat는 dtype을 자동으로 변환하지 않습니다. 텐서의 dtype을 모델의 입력 계약과 일치시켜야 합니다.

## 소스 파일
- C++: `tutorials/009_pass_numpy_to_model/pass_numpy_to_model.cpp`
- Python: `tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py`
