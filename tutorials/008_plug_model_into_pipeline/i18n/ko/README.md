# 008 모델을 파이프라인에 연결하기

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | graph, composition, patterns |

## Concept

`Graph`에 `model.graph()` 및 `model.graph(options)`를 사용하여 모델을 연결합니다. 이 두 가지 구성 패턴은 연결 제어 수준이 다르므로, 간단한 실행과 다중 카메라 배포 중 어떤 것에 더 적합한지 알 수 있습니다.

## Walkthrough

3장에서는 그래프를 노드 단위로 구축했습니다. 이것이 가장 명확한 방법이지만, 일단 그래프를 구축하면 `Model`내부 배선을 직접 하려는 경우는 거의 없을 것입니다. `model.graph(...)` 모델의 파이프라인을 그룹으로 묶어 제공하여 사용자가 원하는 곳에 배치할 수 있도록 합니다. `Graph` 하나와 함께 `add(...)`흥미로운 질문은 그 집단이 가져오는 경계의 *얼마나 많은 부분*인가 하는 것이며, 그것이 바로 핵심입니다. `ModelRouteOptions` 컨트롤.

이 장에서는 동일한 모델을 기준으로 두 가지 경로 구성을 비교합니다. 하나는 자체적인 공개 입력/출력 경계를 포함하는 독립적인 실행 가능한 그래프이고, 다른 하나는 입력을 생략하여 상위 소스(예: 카메라)에 명시적인 이름으로 연결할 수 있는 연결된 그래프입니다. 마지막에는 두 가지를 모두 구성하고 각 그래프의 백엔드를 출력할 수 있습니다. GStreamer 문자열이므로 배선 방식이 어떻게 다른지 정확히 확인할 수 있습니다.

### 실행 가능한 모델 그래프를 구성합니다. {#step-model-graph}

첫 번째 패턴은 모델에 완전하게 실행 가능한 그래프를 요청합니다. 설정을 `include_input = true` 그리고 `include_output = true` 경로 옵션에서 알려줍니다. `model.graph(opts)` 모델 그룹을 중심으로 명확한 외부 입력 및 출력 경계를 설정하여, 결과적으로 `Graph` 다른 요소 없이 독립적으로 구축하고 실행할 수 있습니다. `graph.add(model.graph(opts))` 전체 구성은 바로 그 하나입니다. `add` 결국 모든 패턴은 이것으로 귀결됩니다. 인쇄 `describe_backend()` 생성된 내용을 보여줍니다. GStreamer 파이프라인 문자열.

**C++:** 경로 옵션은 다음과 같습니다. `Model::RouteOptions`; 그래프는 `simaai::neat::Graph`.

**Python:** 경로 옵션은 다음과 같습니다. `pyneat.ModelRouteOptions`; 그래프는 `pyneat.Graph`.

### 첨부 시점 라우팅 옵션을 구성합니다. {#step-route-options}

두 번째 패턴은 모델에 자체 입력을 제공하는 대신 상위 소스 아래에 모델을 연결합니다. `include_input = false` 공개 입력 범위를 제거합니다(프레임은 다른 곳에서 가져옵니다). `include_output = true` 출력 결과를 유지하고, `upstream_name`, `name_suffix`그리고 `buffer_name` 배선 및 요소 이름을 명확하게 지정합니다. 이와 같이 일관성 있는 명명 규칙을 사용하면 다중 카메라 또는 다중 모델 배포 환경에서 백엔드 그래프의 가독성과 문제 진단이 용이해집니다.

### 모델 그룹을 연결합니다. {#step-attached-graph}

해당 옵션을 설정하면, `graph.add(model.graph(opts))` 동일한 모델 그룹을 주입하며, 이제는 자체 소스를 전달하는 대신 지정된 상위 시스템에 연결되도록 구성됩니다. 이는 동일한 것입니다. `add` 첫 번째 패턴으로 호출합니다. 변경된 것은 경로 옵션뿐이며, 중요한 점은 다음과 같습니다. 구성은 하나의 작업이며, `ModelRouteOptions` 그룹이 가져오는 경계는 무엇인지 결정하는 다이얼입니다.

**C++:** 각 변형은 다음을 출력합니다. `describe_backend()` 따라서 두 개의 백엔드 문자열을 비교할 수 있습니다. 그런 다음 파일은 직접 연결된 회로를 구축하고 실행합니다. `Input -> Output` 그래프를 통해 전체 경로를 확인하고, 출력합니다. `direct_rank=`.

**Python:** 첨부된 변형은 다음을 출력합니다. `attached_graph_built=True` 구성 성공 여부를 확인합니다.

## Run

**Python** 및 **C++(미리 빌드된 버전)** 명령을 실행합니다.Neat root 디렉터리(해당 디렉터리에 포함된)를 설치합니다. `share/` 그리고 `lib/`); 소스 코드를 기반으로 빌드하는 명령어를 **저장소의 최상위 디렉터리**에서 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_008_plug_model_into_pipeline \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_008_plug_model_into_pipeline
./build/tutorials-standalone/tutorial_008_plug_model_into_pipeline \
  --model /tmp/yolo_v8s.tar.gz
```

예상 출력(C++ 빌드에서는 각 백엔드 그래프 문자열을 출력한 다음 직접 그래프의 순위를 출력합니다):

```text
model_graph_backend=
...
attached_graph_backend=
...
direct_rank=3
[OK] 008_plug_model_into_pipeline
```

(Python 빌드에서는 다음이 출력됩니다.) `direct_graph_backend=` 백엔드 문자열이 뒤따르고, 그 다음에는 `attached_graph_built=True`.) 이 장의 C++ 소스 코드를 사용자 지정 방식으로 자신의 프로젝트에 통합하려면 `CMakeLists.txt` (추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.cpp`
- 파이썬: `tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.py`
