# 014 그래프 내에 모델을 포함하기

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | yolo_v8s |
| Labels | graph, hybrid, model, mpk |

## Concept

컴파일된 모델을 `graph.add(model)`을 사용하여 공개 `Graph`에 추가하면, 내부 런타임 그래프에 직접 접근하지 않고도 모델 실행을 중심으로 그래프 수준의 오케스트레이션(라우팅, 스케줄링, 추가 입력/출력)을 수행할 수 있습니다.

## Walkthrough

제3장에서는 단순한 입력/출력 노드를 사용하여 그래프를 구축했습니다. 제1장에서는 모델을 독립적인 객체로 실행했습니다. 이 장에서는 이 두 가지를 결합합니다. `Model` 자체는 그래프와 호환되는 노드이므로 다른 모든 스테이지와 마찬가지로 공개 `Graph`에 포함할 수 있습니다. 이것이 바로 그래프 수준 제어(다중 입력, 명명된 출력, 사용자 지정 라우팅 등)가 필요할 때, 모델 실행을 재사용 가능한 단일 조각으로 취급하면서 사용할 수 있는 브리지 패턴입니다.

핵심 아이디어는 저수준 런타임 그래프, `StageModelExecutorOptions` 또는 내부 노드 ID를 절대 수정하지 않는다는 것입니다. 모델을 `graph.add(...)`에 전달하면 NEAT가 해당 조각(필요에 따라 전처리/추론/후처리)을 빌드 시간에 올바른 내부 실행 계획으로 변환합니다. 마지막에는 모델을 공개 그래프에 포함하고, 포함된 토폴로지를 출력하고, 모델의 출력 카디널리티를 다시 읽을 수 있습니다.

### 모델 로드 {#step-load-model}

구성은 컴파일된 아카이브를 로드하고 제1장과 마찬가지로 실행을 위해 준비합니다. 여기서는 옵션 객체 없이 경로만 사용합니다. 왜냐하면 이 장은 전처리보다는 구성에 관한 것이기 때문입니다. 결과 `Model`은 이제 그래프 레이어가 이해할 수 있는 객체입니다.

### 모델을 그래프에 포함 {#step-compose-graph}

이것이 이 장의 핵심입니다. 새 `Graph`에 세 개의 노드가 순서대로 추가됩니다. 명명된 입력 경계, 모델 자체, 명명된 출력 경계입니다. `Model`은 그래프와 호환되므로 `add(model)`은 전체 모델 경로를 단일 조각으로 추가합니다. 특별한 API가 없고 런타임에 직접 접근하지 않습니다. `graph.describe()`를 출력하면 포함된 토폴로지가 표시되므로 명명된 경계 사이에 모델이 올바르게 포함되었는지 확인할 수 있습니다.

**C++:** 경계는 `simaai::neat::nodes::Input("image")` 및 `nodes::Output("result")`에서 가져옵니다. 모델은 `graph.add(model)`에 직접 전달됩니다.

**Python:** 경계는 `pyneat.nodes.input("image")` 및 `pyneat.nodes.output("result")`에서 가져옵니다. 모델은 `graph.add(model)`에 직접 전달됩니다.

### 모델 검사 {#step-inspect-model}

마지막으로 모델 조각이 실제로 어떤 역할을 하는지 확인합니다. 이를 통해 모델이 올바르게 로드되었는지 확인하고 그래프가 다운스트림에서 생성할 출력 토폴로지를 확인할 수 있습니다.

**C++:** `model.info()`는 정보 구조체를 반환합니다. `model_name`와 `output_topology.physical_outputs` 및 `logical_outputs`를 출력하여 모델의 출력 연결이 명확하게 표시되도록 합니다.

**Python:** 이 장의 바인딩은 모델 조각이 공개 그래프에 추가되었다는 확인 메시지를 단순히 출력합니다.

## Run

이 장에서는 모델 아카이브(`yolo_v8s`)가 필요합니다. Neat 설치 루트( `share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행하고, **리포지토리 루트**에서 **소스에서 빌드** 명령을 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/014_embed_model_inside_graph/embed_model_inside_graph.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_014_embed_model_inside_graph \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_014_embed_model_inside_graph
./build/tutorials-standalone/tutorial_014_embed_model_inside_graph \
  --model /tmp/yolo_v8s.tar.gz
```

예상 출력(C++ 빌드도 먼저 구성된 그래프 설명을 출력합니다):

```text
model=yolo_v8s physical_outputs=1 logical_outputs=1
[OK] 014_embed_model_inside_graph
```

(Python 빌드는 그래프 설명을 출력한 다음 `model fragment added to public Graph`를 출력합니다.)

사용자 지정 `CMakeLists.txt`를 사용하여 이 장의 C++ 소스를 자체 프로젝트에 통합하려면(추가 폴더가 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/014_embed_model_inside_graph/embed_model_inside_graph.cpp`
- Python: `tutorials/014_embed_model_inside_graph/embed_model_inside_graph.py`
