# 013 사용자 지정 데이터 그래프 만들기

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | graph, traversal, metadata |

## Concept

가장 작고 유용한 공개 버전을 구축합니다. Neat `Graph` — 하나는 *이름이 있는* `Input` *이름이 지정된* 것에 연결됨 `Output` — 그런 다음 샘플을 이름으로 지정하여 해당 샘플을 통과시키고, 샘플의 메타데이터가 제대로 유지되는지 확인합니다.

## Walkthrough

3장에서는 익명의 입력 → 출력 그래프를 구축하고 위치 정보를 사용하여 실행했습니다. `run()` 호출. 실제 오케스트레이션 — 팬아웃, 팬인, 스트림별 라우팅 — 은 위치가 아닌 *이름*으로 엔드포인트를 처리해야 합니다. 이 장에서는 가능한 가장 작은 그래프에서 이름이 지정된 엔드포인트 표면을 소개하므로, 멀티스트림 및 임베디드 모델 장에서 이를 기반으로 구축하기 전에 이름 지정 및 연결 메커니즘을 독립적으로 확인할 수 있습니다.

공개 `Graph` 애플리케이션 구성 표면은 다음과 같습니다. `add(...)` 노드, `connect(...)` 이름이 지정된 엔드포인트, `build()` 한 번 재사용 가능한 형태로 `Run`그런 다음 `push("image", ...)` 그리고 `pull("out", ...)` 이름으로. 결국에는 하나의 텐서를 푸시하게 될 것입니다. `Sample` 이름이 지정된 그래프를 통해 확인한 결과 `stream_id`, `frame_id`그리고 `pts_ns` 변경되지 않은 채로 출력됨 — 런타임이 시작부터 끝까지 메타데이터를 보존한다는 증거입니다.

### 그래프 구성 {#step-compose-graph}

노드를 두 개 추가합니다. `Input("image")` 이름이 지정된 푸시 엔드포인트를 선언합니다. `image`; `Output("out")` 이름이 지정된 풀 엔드포인트를 선언합니다. `out`이름은 계약 조건입니다. 이는 실제로 전달할 문자열입니다. `push(...)` 그리고 `pull(...)` 나중에. 엔드포인트에 이름을 지정하는 방식(추가 순서에 의존하는 대신)은 여러 입력 또는 출력을 가진 더 큰 그래프를 명확하게 제어할 수 있도록 합니다.

**C++:** 노드는 다음에서 생성됩니다. `simaai::neat::nodes::Input("image")` 그리고 `nodes::Output("out")`.

**Python:** 노드는 다음에서 생성됩니다. `pyneat.nodes.input("image")` 그리고 `pyneat.nodes.output("out")`.

### 엔드포인트를 연결합니다. {#step-connect-endpoints}

`connect("image", "out")` 경계를 선언합니다: 프레임이 밀려난 곳은 `image` 흐름이 향하는 곳 `out`두 개의 노드만으로 전체 토폴로지를 구성할 수 있지만 `connect(...)` 더 큰 그래프에서 분기와 병합을 생성하는 데 사용하는 것과 동일한 함수입니다. 그런 다음 다음을 출력합니다. `graph.describe()` 구성된 토폴로지를 내보내고, 그래프가 의도한 대로 연결되었는지 빠르게 확인합니다. 그런 다음 빌드합니다.

### 샘플 빌드 및 푸시 {#step-build-and-push}

`build()` (여기서는 초기 샘플이 필요하지 않으므로) 설명을 실행 가능한 형태로 구현합니다. `Run`. 그런 다음 하나의 결정론적 텐서를 구성합니다. `Sample` — 알려진 값을 담고 있는 8×8×3 RGB 이미지 `stream_id`, `frame_id`그리고 `pts_ns` — 그리고 `push(...)` 해당 항목을 `image` 이름으로 엔드포인트를 지정합니다. 샘플의 메타데이터는 반대쪽에서 확인할 내용입니다.

**C++:** `push(...)` bool 값을 반환합니다. 실패 시 오류를 표시합니다. `run.last_error()`샘플은 다음과 같이 구성됩니다. `make_sample()`.

**Python:** `push("image", [sample])` 샘플 목록을 받습니다. 샘플은 다음과 같이 구성됩니다. `make_rgb_sample()`.

### 결과를 가져와서 메타데이터를 확인합니다. {#step-pull-and-verify}

`pull("out", ...)` 지정된 출력 엔드포인트에서 결과를 가져오고, 지정된 시간 제한이 지나면 `close()` 실행 결과입니다. 입력과 출력 사이에 변환 과정이 없으므로 올바른 파이프라인은 동일한 논리적 샘플을 반환합니다. 따라서 다시 읽어보면 `stream_id`, `frame_id`그리고 `pts_ns` 그리고 우리가 전송한 값을 확인하면 런타임이 반복 작업을 통해 샘플별 메타데이터를 보존한다는 것을 알 수 있습니다. 이러한 보장은 후속 단계에서 프레임 식별 정보와 타임스탬프를 신뢰할 수 있도록 합니다.

## Run

실행하면 그래프 설명과 그 뒤에 왕복 전송된 메타데이터가 표시됩니다. **Python** 및 **C++(미리 빌드된)** 명령을 실행합니다.Neat root 디렉터리(해당 디렉터리에 포함된)를 설치합니다. `share/` 그리고 `lib/`); 소스 코드를 기반으로 빌드하는 명령어를 **저장소의 루트 디렉터리**에서 실행합니다. 이 장에서는 모델 아카이브가 필요하지 않습니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_013_build_a_custom_data_graph
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_013_build_a_custom_data_graph
./build/tutorials-standalone/tutorial_013_build_a_custom_data_graph
```

예상 출력 (다음과 같음) `graph.describe()` 덤프:

```text
stream=graph frame=42 pts_ns=123456789
[OK] 013_build_a_custom_data_graph
```

(Python 빌드에서는 `stream_id=graph frame_id=42 pts_ns=123456789`가 출력됩니다.) 이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt`를 사용하여 자신의 프로젝트에 통합하려면(추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.cpp`
- 파이썬: `tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.py`
