# 015 하나의 그래프에서 여러 스트림 실행

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | None |
| Labels | graph, multistream, scheduler, join |

## Concept

하나의 공개 그래프(`Graph`)를 통해 여러 개의 논리적 스트림을 실행하고, 두 개의 명명된 입력을 결합하여 하나의 결정론적 번들 출력으로 만듭니다. 이는 다중 카메라 또는 다중 소스 시스템의 기본 패턴으로, 관련된 입력은 후속 처리 전에 정렬되어야 합니다.

## Walkthrough

이전 장에서는 하나의 입력을 밀어 넣고 하나의 출력을 가져왔습니다. 실제 멀티 카메라 및 병렬 분기 시스템은 더 복잡합니다. 여러 스트림이 독립적으로 진행되며, 다운스트림에서 이를 사용하기 전에 결과가 올바르게 *다시 결합*되어야 합니다. 이 장에서는 이를 결정적으로 수행하는 결합 기본 요소인 두 개의 명명된 입력과 하나의 명명된 출력을 가진 결합 그래프를 보여줍니다. 이 그래프는 양쪽 모두 일치하는 프레임을 생성했을 때만 번들을 출력합니다.

밀어 넣는 모든 샘플에는 `stream_id`와 `frame_id`가 포함됩니다. 결합 정책 `ByFrame`은 두 개의 명명된 입력(`left` 및 `right`)이 동일한 `frame_id`를 가진 샘플을 모두 전달할 때까지 기다린 다음 정확히 하나의 결합된 번들을 출력합니다. 마지막에는 결합 그래프를 구축하고, 두 개의 입력으로 결정적인 스트림/프레임별 워크로드를 분산시키고, 결합된 번들을 가져와 출력 개수와 각 번들에 두 개의 필드가 포함되어 있는지 확인합니다.

### 결합 그래프 구축 {#step-build-combine-graph}

`graphs::Combine` (C++) / `graphs.combine` (Python)은 일반적인 공개 `Graph` 조각을 반환합니다. 모양 외에는 특별한 점이 없습니다. 즉, 두 개의 명명된 입력, 하나의 명명된 출력 및 결합 정책이 있습니다. 입력 이름으로 `["left", "right"]`를, 출력 이름으로 `"combined"`를 전달하고, 프레임 ID 일치를 선택하기 위해 `CombinePolicy.ByFrame`을 사용합니다. `describe()`를 출력하면 결과 토폴로지가 표시되고, `build()`는 설명을 실행 가능한 핸들로 변환합니다. 그래프는 기본적으로 비동기적으로 실행되므로 각 스트림은 자체적으로 진행할 수 있습니다.

출력 큐는 제한됩니다. 전체 워크로드를 위한 충분한 큐 공간을 할당하는 대신 이 예제에서는 다음 쌍을 밀어 넣기 전에 각 결합된 번들을 가져옵니다. 프로듀서와 컨슈머가 함께 진행되므로 프레임 수가 증가함에 따라 메모리 사용량이 제한됩니다.

`CombinePolicy.ByFrame`은 `Sample.frame_id`를 기준으로 일치합니다. `CombinePolicy.ByPts`는 프레임이 깔끔한 프레임 인덱스를 공유하지 않을 때 프레젠테이션 타임스탬프(`Sample.pts_ns`)를 기준으로 일치시키는 대체 방법입니다.

### 스트림 밀어 넣기 {#step-push-streams}

이제 워크로드를 실행합니다. 각 프레임과 각 스트림에 대해 해당 `stream_id`와 고유한 `frame_id`로 태그가 지정된 작은 결정적 RGB 샘플을 합성한 다음 *두* 개의 명명된 입력 모두에 밀어 넣습니다. ID가 결정적으로 계산되기 때문에(`frame * streams + sid`), 결합은 찾을 수 있는 명확한 쌍을 갖습니다. 즉, `left` 프레임 N은 항상 일치하는 `right` 프레임 N을 갖습니다. 일치하는 `right`를 밀어 넣은 후에는 해당 쌍의 결합된 출력을 가져온 다음 다음 쌍으로 이동합니다.

**C++:** 각 샘플은 `frame_id` 및 `stream_id`가 설정된 `Tensor` (HWC, UInt8, RGB)를 래핑하는 `Sample`로 명시적으로 구성됩니다. `run.push("left", sample)`은 `run.last_error()`와 비교하여 확인해야 하는 bool 값을 반환합니다.

**Python:** `make_rgb_sample(...)`은 `Tensor.from_numpy(...)`를 통해 NumPy 배열에서 `Sample`을 생성합니다. `run.push("left", [sample])`은 샘플 목록을 받습니다.

### 각 결합된 번들을 가져오기 {#step-pull-bundles}

각 쌍이 푸시된 직후, 명명된 출력 `"combined"`에서 한 번 가져옵니다. 각 성공적인 가져오기는 런타임이 두 입력 모두 해당 프레임을 전달한 후에 출력한 번들을 반환합니다. 생성과 동시에 데이터를 소비함으로써 제한된 출력 큐가 가득 차서 입력 측에 역압력이 전파되는 것을 방지합니다. 두 예제 모두 각 번들에 두 개의 결합된 필드가 포함되어 있는지 확인한 다음 `close()`를 호출하여 실행을 깔끔하게 종료합니다. 예상되는 번들 수는 `streams * frames`와 같으며, 이는 쌍이 누락되지 않았음을 증명합니다.

**C++:** `run.pull("combined", timeout_ms)`는 선택적 번들을 반환합니다. `bundle.stream_id` 및 `bundle.fields.size()`를 읽고 각 번들에 두 개의 필드가 있는지 확인합니다.

**Python:** `run.pull("combined", 2000)`는 번들 또는 `None`을 반환합니다. 예제는 시간 초과 시 즉시 실패하고 각 번들의 필드 수를 확인합니다.

## Run

이 장에서는 모델 아카이브가 필요하지 않습니다. Neat 설치 루트( `share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(사전 빌드)** 명령을 실행합니다. **소스에서 빌드** 명령은 **리포지토리 루트**에서 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/015_run_multiple_streams/run_multiple_streams.py \
  --streams 8 --frames 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_015_run_multiple_streams
./build/tutorials-standalone/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

예상 출력(C++ 빌드는 그래프 설명도 출력합니다. 두 빌드 모두 처음 몇 개의 번들을 출력합니다):

```text
received=32 fields=2
[OK] 015_run_multiple_streams
```

사용자 지정 `CMakeLists.txt`로 이 장의 C++ 소스를 자체 프로젝트에 통합하는 방법(추가 폴더가 필요하지 않음)은 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/015_run_multiple_streams/run_multiple_streams.cpp`
- Python: `tutorials/015_run_multiple_streams/run_multiple_streams.py`
