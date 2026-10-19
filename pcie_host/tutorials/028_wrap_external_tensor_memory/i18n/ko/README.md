# 028 외부 텐서 메모리 래핑

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15분 |
| Model | yolo_v8s |
| Labels | PCIe, C++, tensor, external memory, zero-copy wrapping |

## Concept

`Tensor::from_external()`을 사용하면 C++ 애플리케이션이 기존 연속 할당을 새 호스트 측 텐서로 복사하지 않고 래핑할 수 있습니다. 텐서는 `std::shared_ptr<void>` 소유자를 유지하므로 PCIe와 GStreamer가 사용하는 동안 기반 할당이 살아 있습니다. 애플리케이션은 해당 추론 결과를 가져올 때까지 바이트를 변경하지 않아야 합니다.

이는 연속된 단일 입력에서 추가 호스트 스테이징 복사를 제거합니다. 하지만 종단 간 제로 카피는 아닙니다. PCIe 전송은 여전히 페이로드를 카드 소유 메모리로 복사합니다.

## Walkthrough

프로그램은 재사용 가능한 입력 슬롯 3개를 만들고 합성 FP32 프레임 8개를 제출합니다. 슬롯은 해당 슬롯의 순서가 보장된 결과를 가져온 후에만 사용 가능 큐로 돌아갑니다.

### 모델 계약 검사 {#step-inspect-contract}

메모리를 할당하기 전에 모델을 생성하고 `info().inputs`를 읽습니다. 이 튜토리얼은 단일 입력 YOLOv8s 아카이브를 사용하며 보고된 dtype이 FP32이고 형상이 정확히 `size_bytes` 바이트를 차지하는지 확인합니다.

외부 뷰는 해당 `TensorInfo`의 dtype, 형상, 바이트 크기, 이름과 일치해야 합니다. 다른 모델 빌드에서 이 값을 추정하지 마십시오.

### 애플리케이션 소유 메모리 래핑 {#step-wrap-memory}

각 링 슬롯은 `std::shared_ptr<std::vector<float>>`를 소유합니다. `Tensor::from_external()`에는 기본 포인터, 전체 기반 요소 수, 공유 소유자, 모델 형상, 라우트 이름을 전달합니다. 뷰가 연속적이므로 PCIe 호스트는 스테이징 할당을 만들지 않고 직접 래핑할 수 있습니다.

원시 포인터만 유지하는 것으로는 충분하지 않습니다. 전송 계층이 `push()` 반환 후에도 텐서를 보유할 수 있으므로 공유 소유자가 필수입니다.

### 모델 빌드 {#step-build-model}

모델 계약과 링 할당을 검증한 후 빌드합니다. 예제는 `max_inflight`를 링 크기로 설정하여 애플리케이션과 전송 계층에 동일한 명시적 한도를 적용합니다.

### 링 제출 및 안전한 재사용 {#step-submit-ring}

사용 가능한 슬롯을 채우고 `push()`를 호출한 뒤 해당 슬롯을 처리 중 큐로 이동합니다. `push()`가 반환했다는 이유만으로 저장 공간을 수정하거나 재사용하지 마십시오. 사용 가능한 슬롯이 없으면 예제는 `pull()`을 호출하고 일치하는 순서의 결과가 도착한 후에만 가장 오래된 슬롯을 사용 가능 큐로 되돌립니다.

마지막 드레인을 포함하여 수락된 모든 푸시마다 한 번씩 풀합니다. 시간 초과가 발생하면 아직 활성 상태일 수 있는 요청의 메모리를 재사용하지 않고 모델을 닫습니다.

## Run

PCIe 호스트 패키지를 설치하고 [튜토리얼 설정](/tutorials/before-you-run)에 설명된 대로 튜토리얼 번들을 다운로드합니다. 압축을 푼 PCIe extras 루트에 YOLOv8s를 다운로드합니다.

```bash
sima-cli modelzoo get yolo_v8s
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_028_wrap_external_tensor_memory
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_028_wrap_external_tensor_memory
./build/tutorials-standalone/tutorial_028_wrap_external_tensor_memory
```

기본값은 카드 0과 큐 0입니다. 다른 카드를 사용할 때만 `--card N`을 전달하십시오. 성공하면 다음이 출력됩니다.

```text
input=images
ring_slots=3
completed=8
[OK] 028_wrap_external_tensor_memory
```

## In Practice

직접 호스트 래핑 경로에는 연속 저장 공간이 필요합니다. 불연속 스트라이드가 있는 텐서도 설명자가 유효하면 허용되지만, 호스트가 스테이징 할당으로 압축합니다. 별도로 할당된 여러 입력도 스테이징 메모리로 패킹됩니다.

다중 입력 모델에서는 모든 텐서가 하나의 공유 패킹 할당에 대한 연속 뷰일 때만 이 스테이징 할당을 피할 수 있습니다. `info().inputs`가 보고한 순서로 제출하고 각 입력의 이름과 형상을 사용하십시오. 예를 들어 두 입력이 모두 FP32인 경우는 다음과 같습니다.

```cpp
const auto& first = info.inputs.at(0);
const auto& second = info.inputs.at(1);
const std::size_t first_count = first.size_bytes / sizeof(float);
const std::size_t second_count = second.size_bytes / sizeof(float);

auto packed =
    std::make_shared<std::vector<float>>(first_count + second_count);
pcie::Tensor input0 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, first.shape, first.name);
pcie::Tensor input1 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, second.shape, second.name,
    static_cast<std::int64_t>(first.size_bytes));

model.push({input0, input1});
```

이 최적화는 호스트 측 패킹 복사만 제거합니다. PCIe는 추론 전에 패킹된 페이로드를 카드 소유 전송 메모리로 계속 복사합니다.

## 소스 파일

- `run_external_tensor_memory.cpp`
