# 010 하나의 샘플에 여러 입력을 전달

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | None |
| Labels | multi-input, samples, sync |

## Concept

여러 개의 명명된 텐서를 하나의 `Sample`로 묶어 단일 추론 이벤트로 전달합니다. 이는 스테레오 프레임, 이미지 + 메타데이터 또는 센서 융합과 같이 둘 이상의 입력을 받는 모델에 적용되는 방식입니다.

## Walkthrough

실제 애플리케이션 중 상당수는 추론 이벤트당 하나 이상의 입력을 사용합니다. Neat 이를 **번들 샘플**로 나타냅니다. 즉, 단일 `Sample` 누구의 `fields` 목록은 여러 개의 명명된 텐서 페이로드를 포함하며, 각 페이로드는 특정 주소를 통해 접근할 수 있습니다. `port_name`런타임은 지정된 필드를 하나의 논리적 이벤트로 묶어 관리하므로 `left` 그리고 `right` (또는 이미지와 메타데이터)가 파이프라인을 통해 일관성을 유지합니다.

이 장에서는 텐서 입력/텐서 출력 그래프를 구축하고, 이름이 지정된 두 개의 부동 소수점 텐서를 묶은 다음, 묶음을 파이프라인에 전달하고, 이름이 지정된 필드를 다시 읽어옵니다. 마지막에는 여러 필드가 있는 샘플을 만들고 두 필드 모두 포트 이름이 변경되지 않은 상태로 왕복 이동을 완료했는지 확인합니다.

### 텐서 입력을 구성합니다. {#step-configure-tensor-input}

이 그래프는 디코딩된 이미지가 아닌 원본 텐서를 사용하므로 입력 계약은 텐서 페이로드로 선언됩니다.`FP32`, 함께 `width`/`height`/`depth`)보다는 픽셀 형식을 사용합니다. 이렇게 하면 입력 노드가 텐서 버퍼를 직접 받도록 지시합니다.

**C++:** 설정 `in.payload_type = PayloadType::Tensor`.

**Python:** 설정 `inp.payload_type = pyneat.PayloadType.Tensor` 그리고 `inp.format = pyneat.Format.FP32`.

### 그래프를 구축하고 초기 실행을 수행합니다. {#step-build-seed-run}

저희는 동일한 최소한의 `Input -> Output` 004장의 토폴로지에서 `build()` 그것을 ~로 `Run`. `build()` 협상된 형태를 고정하려면 대표적인 샘플이 필요하므로, 실제 필드가 사용할 것과 동일한 형태의 단일 시드 텐서(모두 0)를 전달합니다. 시드는 형태 협상에만 사용되며, 실제 데이터는 그 다음에 제공됩니다.

### 번들을 구축합니다. {#step-make-bundle}

이제 다중 입력 이벤트를 구성합니다. 각 입력은 를 통해 이름을 받습니다. `make_tensor_sample(port_name, tensor)`그리고 해당 이름이 지정된 필드는 모델이 포트를 통해 처리하는 대상입니다. `left` 으로 가득 차 있습니다 `1.0` 그리고 `right` 함께 `2.0` 그래서 나갈 때 서로 구별할 수 있습니다.

**C++:** `make_bundle_sample({...})` 이름이 지정된 필드를 하나로 묶습니다. `Sample` 누구의 `kind` 입니다 `Bundle`**Python:** 이름이 지정된 샘플 목록이 직접 전달됩니다. `push(...)`; pyneat이 번들 패키지를 생성합니다.

### 번들 패키지를 푸시하고 다시 읽어보세요. {#step-push-and-read}

마지막으로, 번들을 전송하고 결과를 확인합니다. 출력 결과 또한 번들입니다. `Sample`그래서 우리는 읽었습니다. `out.fields` 하나의 텐서로 취급하는 대신 — `out.fields.size()` 다음과 같아야 합니다. `2`각 필드는 다음 값을 포함합니다. `port_name` 그리고 텐서 페이로드.

**C++:** `run.run(Sample{bundle}, timeout_ms)`는 하나의 `Sample`을 반환합니다. 논리적 결과에 여러 필드가 있기 때문에 반환된 `Sample` 자체는 `Bundle`입니다. 따라서 `out.kind == SampleKind::Bundle`을 확인하고 `front()` (번들 내부의 "첫 번째 필드"를 의미)가 아닌 `out.fields`를 반복합니다.

**Python:** `run.push(fields)` 다음에 `run.pull(timeout_ms=...)`가 출력 샘플을 반환합니다. `out.fields`를 반복하고 각 `field.port_name` 및 `field.tensor`를 읽습니다.

## Run

**Neat 설치 루트** (`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++ (미리 빌드된 버전)** 명령을 실행하고, **소스에서 빌드** 명령은 **리포지토리 루트**에서 실행합니다. 이 챕터에는 모델 아카이브가 필요하지 않습니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/010_feed_multi_input_model/feed_multi_input_model.py \
  --width 64 --height 48
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_010_feed_multi_input_model
./build/tutorials-standalone/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

예상 출력 (C++):

```text
bundle_fields=2
  field=left has_tensor=yes
  field=right has_tensor=yes
[OK] 010_feed_multi_input_model
```

(Python 빌드는 `port=left has_tensor=True` 줄과 함께 동일한 필드 수를 출력합니다.) 이 챕터의 C++ 소스를 사용자 지정 `CMakeLists.txt`로 자신의 프로젝트에 통합하려면 (추가 폴더가 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

이 두 필드 데모를 넘어 번들 패턴을 적용하는 방법.

### 이름 지정 및 라우팅

- `port_name`은 배선 계약입니다. 다중 입력 모델이 각 필드를 참조하는 방법입니다. 이름을 모델에 선언된 입력 포트와 일치시킵니다.
- 출력 번들은 필드 구조를 유지하므로 위치가 아닌 이름으로 결과를 입력에 다시 매핑할 수 있습니다.

### 출력 번들 검사

- 항상 먼저 `kind`를 기준으로 분기합니다. 다중 필드 결과는 `SampleKind.Bundle`이며, 단일 텐서로 읽으면 작동하지 않습니다.
- 페이로드를 건드리기 전에 필드당 텐서 존재 여부(`field.tensor is not None` / `field.tensor.has_value()`)를 확인합니다. 필드에는 텐서 대신 메타데이터가 포함될 수 있습니다.

## 소스 파일
- C++: `tutorials/010_feed_multi_input_model/feed_multi_input_model.cpp`
- Python: `tutorials/010_feed_multi_input_model/feed_multi_input_model.py`
