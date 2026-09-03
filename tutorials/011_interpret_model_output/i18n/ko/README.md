# 010 모델 출력 읽고 해석하기

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | output, patterns, sink |

## Concept

`run.pull()` 또는 `model.run()`에서 안전하게 데이터를 읽어옵니다. 각 실행 결과는 `Sample`을 반환합니다. 이는 텐서, 명명된 필드들의 묶음, 또는 둘 다일 수 있는 작은 합 유형입니다. 따라서 페이로드를 처리하기 전에 해당 유형을 분류해야 합니다.

## Walkthrough

처리량을 최적화하거나 복잡한 그래프 로직을 추가하기 전에, 실행 결과로 반환되는 모든 것을 안정적이고 방어적인 방식으로 읽을 수 있어야 합니다. 출력은 항상 `Sample`이지만, 그 형태는 다양합니다. 단일 텐서일 수도 있고, 명명된 필드들의 묶음일 수도 있습니다(009장에서 설명). 묶음에서 `.tensor`에 접근하거나, 존재하지 않는 형태를 가정하는 것은 이 장에서 피해야 할 버그입니다.

이전과 동일한 최소 동기화 그래프를 구축하고, 한 프레임을 실행한 다음, 결과를 체계적으로 *검사*합니다. 즉, `kind`를 확인하고, 텐서가 존재하는지, 필드 수가 얼마나 되는지, 그리고 텐서의 랭크를 확인합니다. 이 과정을 마치면 런타임에서 제공하는 모든 모델에 대해 작동하는 재사용 가능한 출력 읽기 패턴을 갖게 됩니다.

### 입력 {#step-configure-input} 구성

입력 계약(픽셀 `format`, `width`, `height`, `depth`)을 선언하여 우리가 전달할 프레임과 일치시킵니다. 이는 이 장 전체에서 사용되는 동일한 경계 계약입니다.

### 그래프 구성 및 구축 {#step-compose-graph}

입력 노드를 출력 노드에 연결하고, 동기 `Run`으로 `build()`하여 프레임을 전달합니다. 이렇게 하면 `build()`가 구체적인 형태를 협상할 수 있습니다. 모델이 중간에 없으면 출력은 입력과 동일합니다. 이것이 바로 이 지점이 출력 구조를 연구하기에 적합한 이유입니다.

### 한 프레임 실행 {#step-run-frame}

한 프레임을 전달하고 동기적으로 하나의 결과를 가져옵니다. 단일 `run(...)` 호출은 한 프레임에 대한 바로 가기이며, 반환되는 값은 우리가 분석하려는 객체입니다.

**C++:** `run.run(...)`은 `TensorList`를 반환합니다. 단일 텐서 출력의 경우 이는 하나의 항목을 의미하며, 다음 단계에서 `out.size()` 및 `out.front()`를 통해 검사합니다.

**Python:** `run.run(...)`은 `Sample`을 반환하며, `.kind`, `.tensor`, `.tensors` 및 `.fields`를 직접 노출합니다.

### 샘플 검사 {#step-inspect-sample}

이것이 핵심입니다. 페이로드보다 먼저 구조를 읽습니다. 먼저 존재 여부와 종류를 확인한 다음, 텐서의 `shape`에서 랭크를 파생시킵니다. 각 단계를 보호함으로써(비어 있지 않은 출력, 비어 있지 않은 형태) 출력 판독기가 형태를 제어할 수 없는 모델에 대해 강력하게 작동하도록 할 수 있습니다.

**C++:** `out.size()` 및 텐서 존재 여부를 보고, 비어 있거나 `out.front().shape`가 비어 있으면 예외를 발생시킨 다음, `shape.size()`에서 `rank`를 출력합니다. (`fields=0` 줄은 자리 표시자입니다. `TensorList`는 Python `Sample`이 갖는 묶음 필드 구조를 포함하지 않습니다.)

**Python:** `sample.kind`, `sample.tensor is not None`, `len(sample.fields)` 및 첫 번째 텐서의 순위를 출력합니다. 이는 단일 위치에 있는 전체 합 유형 표면입니다. 텐서 종류 결과의 경우 `.tensor`가 존재합니다. 텐서 세트 결과의 경우 `.tensors`를 읽습니다. 번들 분기의 경우 `.kind`를 확인하고 `.fields`를 읽습니다.

## Run

**Neat 설치 루트**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행합니다. **소스에서 빌드** 명령은 **리포지토리 루트**에서 실행합니다. 이 장에서는 모델 아카이브가 필요하지 않습니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/011_interpret_model_output/interpret_model_output.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_011_interpret_model_output
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_011_interpret_model_output
./build/tutorials-standalone/tutorial_011_interpret_model_output
```

예상 출력(C++):

```text
outputs=1 has_tensor=yes fields=0
rank=3
[OK] 011_interpret_model_output
```

Python 빌드는 `Sample` 표면을 통해 동일한 정보를 출력합니다.

```text
sample_kind=SampleKind.TensorSet
has_tensor=False
num_fields=0
output_rank=3
```

사용자 지정 `CMakeLists.txt`를 사용하여 이 장의 C++ 소스를 자체 프로젝트에 통합하려면(추가 폴더가 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

모든 모델의 출력을 읽기 위한 방어적 체크리스트입니다.

### 읽기 전에 분류

- 먼저 `kind`를 확인합니다. 단일 텐서 결과는 `SampleKind.Tensor`이고, 다중 필드 결과는 `SampleKind.Bundle`입니다.
- 텐서 종류의 경우 `tensor`가 존재하고 `fields`는 비어 있습니다. 번들 종류의 경우 `fields`를 읽고 `tensor`가 있다고 가정하지 마십시오.

### 계약 유효성 검사

- 텐서를 참조하기 전에 텐서가 존재하는지 확인합니다.
- 순위 계산 또는 차원 인덱싱을 수행하기 전에 `shape`가 비어 있지 않은지 확인합니다.
- 소비자가 특정 요소 유형을 예상할 때 `tensor.dtype`을 검사합니다.

## 소스 파일
- C++: `tutorials/011_interpret_model_output/interpret_model_output.cpp`
- Python: `tutorials/011_interpret_model_output/interpret_model_output.py`
