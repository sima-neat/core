---
title: "그래프를 사용하여 애플리케이션 구축"
description: "공개 그래프 API를 사용하여 모델, 노드, 명명된 입력/출력, 분기, 결합, 실행을 어떻게 구성할 수 있을까요?"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/graphs
---

# 그래프를 사용하여 애플리케이션 구축

하나의 컴파일된 모델 아카이브만 로드하고 실행하려는 경우 `Model`을 사용합니다. 모델과 노드를 기반으로 애플리케이션을 구축하려는 경우 `Graph`를 사용합니다. 공개 입력 및 출력을 추가하고, 재사용 가능한 조각을 연결하고, 스트림을 분기하고, 스트림을 결합하고, 애플리케이션을 검증하고, 실제로 실행된 내용을 저장하거나 시각화합니다.

개념적 모델은 의도적으로 작습니다.

| 개념 | 의미 |
|---|---|
| `Model` | 디스크에서 로드된 컴파일된 모델 아카이브입니다. 예를 들어 `resnet50.tar.gz` 또는 `yolov8.tar.gz`입니다. |
| `Node` | 하나의 처리 단계입니다. 입력, 출력, 변환, 소스, 싱크, 모델 단계 또는 보조 단계가 될 수 있습니다. |
| `Graph` | 애플리케이션 배선 계획입니다. 어떤 노드/조각이 존재하고 데이터가 어떻게 흐르는지를 정의합니다. |
| `Run` | `Graph::build()`에서 반환된 실시간 실행 핸들입니다. 입력을 푸시하고, 출력을 가져오고, 메트릭을 수집하고, 중지합니다. |

요약하면:

```text
Graph = what to run
Run   = the running instance
```

더 짧은 경로가 필요할 때 작업 페이지부터 시작하세요.

- [그래프](/develop-apps/development-workflow/graph)는 그래프 작성 방법을 알려줍니다.
- [그래프 실행](/develop-apps/development-workflow/pipeline)는 런타임 라이프사이클, 큐, 측정, 처리량 등을 알려줍니다.
- [노드](/develop-apps/development-workflow/node)는 일반 노드와 그룹을 매핑합니다.
- [텐서 및 샘플](/develop-apps/development-workflow/core_types)은 페이로드와 메타데이터를 설명합니다.

대부분의 애플리케이션 코드는 공개 `simaai::neat::Graph` 및 `simaai::neat::Run`을 사용해야 합니다. 더 낮은 수준의 구현 네임스페이스를 사용하여 애플리케이션을 구축하지 마세요. 이는 고객 API가 아닙니다.

## 그래프가 필요한 경우는 언제인가요?

| 목표 | 권장 API |
|---|---|
| 하나의 입력에 대해 하나의 모델을 실행 | `Model::run(...)` 또는 `Model::build(...)` |
| 모델 주위에 애플리케이션 입력/출력 경계를 추가 | `Graph` |
| 사용자 지정 처리 노드를 사용하여 모델을 구성 | `Graph::add(...)` |
| 여러 앱에서 그래프 조각을 재사용 | `Graph` 조각을 반환/전달 |
| 여러 입력 또는 출력을 라우팅 | 명명된 `nodes::Input(...)` / `nodes::Output(...)`에 `connect(...)` 추가 |
| 하나의 스트림을 여러 소비자로 분기 | `graphs::Branch(...)` |
| 여러 스트림을 하나의 논리적 출력으로 결합 | `CombinePolicy`와 함께 `graphs::Combine(...)` |
| 실행된 토폴로지와 메트릭을 저장하거나 시각화 | `save_run_json(run, ...)` |

## 첫 번째 그래프: 하나의 입력, 하나의 모델, 하나의 출력

이것은 가장 작고 완전한 앱 스타일의 그래프입니다.

```cpp
#include <neat.h>

#include <iostream>

namespace neat = simaai::neat;

int main() {
  neat::Model model("resnet50.tar.gz");

  neat::Graph app;
  app.add(neat::nodes::Input("image"));
  app.add(model);
  app.add(neat::nodes::Output("classes"));

  neat::Run run = app.build();

  neat::Tensor image = /* create or load an image tensor */;
  run.push("image", neat::TensorList{image});

  std::optional<neat::Sample> result = run.pull("classes", /*timeout_ms=*/1000);
  if (result) {
    // Consume result->tensors, result->detections, or other Sample metadata.
  }

  run.stop();
}
```

줄 단위 설명:

- `nodes::Input("image")`는 `image`라는 이름의 공개 입력 노드를 선언합니다.
- `app.add(model)`은 모델의 선택된 경로를 그래프에 삽입합니다.
- `nodes::Output("classes")`는 `classes`라는 이름의 공개 출력 노드를 선언합니다.
- `app.build()`는 전체 그래프를 검증하고 컴파일한 후 `Run`을 반환합니다.
- `run.push("image", ...)`는 데이터를 지정된 입력으로 보냅니다.
- `run.pull("classes", ...)`는 지정된 출력에서 데이터를 수신합니다.

Python에서 동일한 구조:

```python
import pyneat

model = pyneat.Model("resnet50.tar.gz")

app = pyneat.Graph()
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

run = app.build()

image = ...  # Create or load a tensor-compatible object.
run.push("image", [image])

result = run.pull("classes", timeout_ms=1000)
run.stop()
```

Python의 `Run.push(...)`는 배치와 유사한 시퀀스를 예상합니다. 개별 텐서/샘플 객체가 아닌 `[tensor]` 또는 `[sample]`를 전달하십시오.

## 그래프 실행

Neat의 다른 곳에서 사용되는 것과 동일한 공개 페이로드 유형을 허용하는 `Run`이 있습니다.

| 페이로드 | 사용 시점 |
|---|---|
| `TensorList` | 텐서를 전달하고 추가 샘플 메타데이터가 필요하지 않은 경우. |
| `Sample` | 타임스탬프, `frame_id`, `stream_id`, 텍스트/오디오/비디오 메타데이터, 감지 결과 또는 EOS가 필요한 경우. |
| `std::vector<cv::Mat>` | OpenCV에서 편리하게 사용할 수 있는 이미지 입력을 원하는 경우. |

일반적인 C++ 호출:

```cpp
run.push(neat::TensorList{image});
run.push("image", neat::TensorList{image});

run.push(sample);
run.push("image", sample);

auto out = run.pull(/*timeout_ms=*/1000);
auto named = run.pull("classes", /*timeout_ms=*/1000);

neat::TensorList tensors = run.pull_tensors("classes", 1000);
neat::Sample sample_out = run.pull_samples("classes", 1000);
```

타임아웃/종료 시 빈 `std::optional`을 반환해야 하는 경우 `pull(...)`을 사용합니다. 타임아웃/오류 발생 시 예외를 발생시키는 형식화된 편리한 도우미 함수가 필요한 경우 `pull_tensors(...)` 또는 `pull_samples(...)`를 사용합니다.

유한한 앱 푸시 스트림의 경우, 최종 메트릭을 수집하기 전에 입력을 닫고 데이터를 비웁니다.

```cpp
run.close_input();
while (auto out = run.pull("classes", 1000)) {
  // Drain remaining output.
}
run.stop();
```

대기열 정책, 출력 소유권, 삭제 텔레메트리, 다중 스트림 측정 등을 포함한 작업 중심 런타임 플레이북에 대해서는 [그래프 실행](/develop-apps/development-workflow/pipeline)를 참조하십시오.

## `build()`와 `build(first_input)`

대부분의 그래프는 입력 샘플 없이도 구축할 수 있습니다.

```cpp
neat::Run run = app.build();
```

그래프가 이미 충분한 모양/캡 정보가 포함되어 있거나, 그래프가 자체 소스 노드(예: RTSP/파일/정지 이미지 입력)를 소유하고 있을 때 이 기능을 사용합니다.

시드 빌드를 사용하면 빌드 중에 Neat에 첫 번째 입력이 제공됩니다.

```cpp
neat::Run run = app.build(neat::TensorList{first_image});
```

스트리밍이 시작되기 전에 첫 번째 입력이 모양/형식 조정의 초기값을 설정해야 할 때 이 기능을 사용합니다. 초기값 설정 후 사전 검사는 기본적으로 활성화되어 있으므로, Neat은 빌드 중에 첫 번째 샘플 오류를 감지하기 위해 한 번 초기값을 적용하거나 가져올 수 있습니다. 즉시 실패하는 `Run`을 반환하는 대신에 말입니다.

처리량, 지연 시간 및 전력 소비량과 같은 지표는 실제 작업이 완료된 후에 저장하고, 빌드 직후에는 저장하지 마십시오.

## 그래프 이름은 엔드포인트 이름과 다릅니다.

:::warning
`Graph("name")`은 진단, 저장된 그래프 파일, 시각화에 사용되는 레이블입니다. `name`이라는 이름의 공개 입력 또는 출력을 선언하지 않습니다.
:::

잘못된 사고방식:

```cpp
neat::Graph camera("image");
// This does not make run.push("image", ...) valid by itself.
```

올바른 엔드포인트 선언:

```cpp
neat::Graph camera("camera_route");
camera.add(neat::nodes::Input("image"));
```

그리고 결과물은 다음과 같습니다.

```cpp
neat::Graph classifier("classifier");
classifier.add(neat::nodes::Output("classes"));
```

`Input("image")`와 `Output("classes")`를 그래프 조각의 공개된 진입점으로 생각하십시오. 그래프 이름은 건물에 붙은 간판과 같습니다.

## 추측 대신 엔드포인트 이름을 확인하십시오.

빌드하기 전에 그래프에서 선언된 논리적 공개 엔드포인트를 확인하십시오.

```cpp
for (const auto& name : app.inputs()) {
  std::cout << "graph input: " << name << "\n";
}
for (const auto& name : app.outputs()) {
  std::cout << "graph output: " << name << "\n";
}
```

빌드 후, `Run`이 실제로 어떤 입력을 허용하는지 확인합니다.

```cpp
for (const auto& name : run.input_names()) {
  std::cout << "run input: " << name << "\n";
}
for (const auto& name : run.output_names()) {
  std::cout << "run output: " << name << "\n";
}
```

이 기능을 모델 경로 및 다중 입력/다중 출력 앱에 사용하세요. 엔드포인트 일치는 정확하게 이루어집니다.

`Input("image_l")`은 `image_l`이라는 모델 입력에 바인딩될 수 있지만, `Input("my_random_name")`은 바인딩될 수 없습니다.

## 이름이 없는 편리한 API

단일 입력/단일 출력 그래프의 경우, 엔드포인트 이름을 생략할 수 있습니다.

```cpp
neat::Graph app;
app.add(neat::nodes::Input());
app.add(model);
app.add(neat::nodes::Output());

neat::Run run = app.build();
run.push(neat::TensorList{image});
auto result = run.pull(1000);
```

이는 간단한 스크립트와 테스트에 편리합니다. 복잡한 애플리케이션의 경우 이름을 사용하는 것이 좋습니다.

그래프에 여러 개의 가능한 입력 또는 출력이 있는 경우, 이름이 지정되지 않은 `push(...)` 또는 `pull()`은 오류를 발생시키고 사용 가능한 이름을 보고합니다. 이러한 오류는 의도적인 것입니다. Neat은 사용자가 어떤 카메라, 텐서 또는 출력 헤드를 의미하는지 추측해서는 안 됩니다.

## 모델은 그래프의 일부입니다.

`Model`은 그래프에 직접 추가할 수 있습니다.

```cpp
neat::Model yolo("yolov8.tar.gz");

neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

`Graph::add(model)`은 아카이브 및 모델 옵션에서 선택한 모델 경로를 삽입합니다. 해당 경로는 전처리, MLA 추론, 후처리, 텐서 변환 및 감지 디코딩 단계 등을 포함할 수 있습니다. 일반적인 선형 방식의 경우 `model.graph()`를 수동으로 호출할 필요가 없습니다.

고급 구성을 위해 해당 경로를 `Graph` 조각으로 검사하거나 재사용하십시오.

```cpp
neat::Graph route = yolo.graph();

auto model_inputs = route.inputs();
auto model_outputs = route.outputs();
```

### 다중 입력 모델

다중 입력 모델의 경우, 이름을 추측하지 마십시오. 경로를 물어보세요.

```cpp
neat::Graph route = model.graph();

for (const auto& name : route.inputs()) {
  std::cout << "model expects input: " << name << "\n";
}
```

다음으로 상위 조각의 이름을 모델 입력 이름과 일치하도록 지정합니다.

```cpp
neat::Graph left_camera;
left_camera.add(neat::nodes::Input("image_l"));

neat::Graph uv_camera;
uv_camera.add(neat::nodes::Input("image_uv"));

neat::Graph app;
app.connect(left_camera, route);  // Binds image_l -> model image_l.
app.connect(uv_camera, route);    // Binds image_uv -> model image_uv.
```

`left_camera`가 `Input("a_new_name_image_l")`을 선언하면 `image_l`에 바인딩되지 않습니다. 암시적 이름 변경에 의존하는 대신 올바른 엔드포인트 이름을 가진 작은 어댑터 그래프를 추가하십시오.

### 독립 실행형 모델 그래프

기본적으로 `model.graph()`는 열린 이름이 지정된 엔드포인트를 가진 재사용 가능한 모델 조각을 반환합니다. 반환된 그래프가 자체적으로 실행되도록 하려면 명시적인 공개 입력/출력 노드를 요청하십시오.

```cpp
neat::Model::RouteOptions route_opt;
route_opt.include_input = true;
route_opt.include_output = true;

neat::Graph standalone = model.graph(route_opt);
neat::Run run = standalone.build();
```

고급 기능 또는 디버깅 목적으로 모델 경로를 사용하여 개별 물리적 출력을 표시할 수 있습니다.

```cpp
route_opt.expose_all_outputs = true;
```

별도의 물리적 출력 버퍼가 특별히 필요한 경우가 아니라면 이 기능을 비활성화 상태로 두십시오. 기본 모델 동작은 라우트 계약에서 예상하는 논리적 모델 출력을 노출하는 것입니다. 모델에 물리적 출력이 하나만 있는 경우에도 `expose_all_outputs = true`는 여전히 하나의 출력만 노출합니다.

## `add()`와 `connect()`

두 가지 합성 도구가 있습니다.

| API | 의미 | 사용 시점 |
|---|---|---|
| `add(x)` | 현재 선형 체인에 추가하거나 연결합니다. | “동일한 파이프라인의 다음 단계”를 의미할 때 사용합니다. |
| `connect(a, b)` | 명명된 엔드포인트를 사용하여 두 개의 그래프 조각을 연결합니다. | 재사용 가능한 조각을 합성하거나 토폴로지를 구축할 때 사용합니다. |
| `connect("a", "b")` | 동일한 그래프 내에 이미 선언된 두 개의 엔드포인트를 연결합니다. | 작은 보조 조각을 구축할 때 사용합니다. |

선형 합성:

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
```

단편 구성:

```cpp
neat::Graph app;
app.connect(camera, model_route);
app.connect(model_route, output_sink);
```

헬퍼 조각 내부의 내부 엔드포인트 배선:

```cpp
neat::Graph pass_through("pass_through");
pass_through.add(neat::nodes::Input("in"));
pass_through.add(neat::nodes::Output("out"));
pass_through.connect("in", "out");
```

핵심 규칙: `add()`는 선형 체인을 의미합니다. `connect()`는 그래프 토폴로지를 의미합니다.

## 재사용 가능한 그래프 조각

함수는 재사용 가능한 그래프 조각을 반환할 수 있습니다.

```cpp
neat::Graph make_classifier(neat::Model& model) {
  neat::Graph g("classifier");
  g.add(neat::nodes::Input("image"));
  g.add(model);
  g.add(neat::nodes::Output("classes"));
  return g;
}
```

재사용 가능한 조각을 선형적으로 사용합니다.

```cpp
neat::Graph classifier = make_classifier(model);

neat::Graph app;
app.add(classifier);
```

또는 와이어 조각을 명확하게 표시합니다.

```cpp
neat::Graph app;
app.connect(camera, classifier);
app.connect(classifier, class_sink);
```

가지가 끝난 후에 `add()`를 추가하면 의미가 모호해지므로, Neat은 실패하고 대신 `connect(...)`를 사용하라고 알려줍니다. 이는 아무런 경고 없이 잘못된 가지에 추가하는 것보다 훨씬 낫습니다.

## 하나의 스트림을 분기하기

하나의 입력 스트림이 여러 개의 명명된 출력으로 가야 할 경우 `graphs::Branch`를 사용합니다.

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});
```

의미:

```text
image -> preview
      -> model_input
```

예시:

```cpp
neat::Graph camera;
camera.add(neat::nodes::Input("image"));

neat::Graph preview;
preview.add(neat::nodes::Output("preview"));

neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});

neat::Graph app;
app.connect(camera, branch);
app.connect(branch, preview);
```

브랜치를 모델에 연결할 때, 모델 입력 이름과 일치하도록 브랜치 출력 이름을 선택하세요.

```cpp
neat::Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "choose a branch output matching: " << name << "\n";
}
```

분기는 명시적으로 이루어지는데, 이는 큐와 역압에 영향을 미치기 때문입니다. 한 분기가 느리면, 출력 옵션과 후속 그래프에 따라 다른 분기에 비해 속도가 느려지거나 처리량이 줄어들 수 있습니다.

Python:

```python
branch = pyneat.graphs.branch("image", ["preview", "model_input"])
```

## 여러 스트림 결합하기

여러 입력 스트림을 하나의 논리적 출력으로 결합해야 할 때 `graphs::Combine`를 사용합니다.

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);
```

의미:

```text
left  --\
        +--> stereo
right --/
```

정책:

| 정책 | 의미 |
|---|---|
| `CombinePolicy::None` | 자동으로 결합하지 않습니다. 여러 프로듀서가 하나의 출력으로 연결될 때 오류가 발생하면 연결이 끊어집니다. |
| `CombinePolicy::ByFrame` | 정확히 동일한 `Sample::frame_id`를 가진 샘플을 매칭합니다. 프레임 ID가 누락되면 오류가 발생하며, PTS로 대체하는 기능은 없습니다. |
| `CombinePolicy::ByPts` | 정확히 동일한 `Sample::pts_ns` 프레젠테이션 타임스탬프를 가진 샘플을 매칭합니다. PTS가 누락되면 오류가 발생하며, 프레임 ID로 대체하는 기능은 없습니다. |

쉬운 설명:

- `ByFrame`은 “동일한 프레임 번호를 가진 왼쪽 및 오른쪽 샘플을 제공해 주세요”를 의미합니다.
- `ByPts`는 “동일한 미디어 타임스탬프를 가진 샘플을 제공해 주세요”를 의미합니다.

예시:

```cpp
neat::Graph left;
left.add(neat::nodes::Input("left"));

neat::Graph right;
right.add(neat::nodes::Input("right"));

neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);

neat::Graph app;
app.connect(left, pair);
app.connect(right, pair);

neat::Run run = app.build();
run.push("left", left_sample_with_frame_id_42);
run.push("right", right_sample_with_frame_id_42);
auto stereo = run.pull("stereo", 1000);
```

파이썬:

```python
pair = pyneat.graphs.combine(["left", "right"], "stereo", pyneat.CombinePolicy.ByFrame)
```

샘플에 필요한 키가 포함되어 있지 않으면, 추측 대신 진단 메시지와 함께 결합 단계가 실패합니다.

## 데이터 소스와 대상

데이터가 그래프에 입력되는 방식은 두 가지이고, 그래프에서 출력되는 방식도 두 가지입니다.

### 애플리케이션에서 직접 입력

애플리케이션 코드가 데이터를 직접 입력할 때는 `nodes::Input(...)`을 사용합니다.

```cpp
app.add(neat::nodes::Input("image"));
run.push("image", neat::TensorList{image});
```

### 그래프가 소유한 입력 소스

그래프가 데이터 소스를 소유하는 경우 소스 노드 또는 소스 조각을 사용합니다.

```cpp
app.add(neat::nodes::RTSPInput("rtsp://camera/stream"));
```

또는 재사용 가능한 디코딩된 RTSP 조각:

```cpp
neat::nodes::groups::RtspDecodedInputOptions opt;
opt.url = "rtsp://camera/stream";

app.add(neat::nodes::groups::RtspDecodedInput(opt));
```

그래프가 자체 소스를 소유할 때, 일반적으로 `build()`를 호출한 다음 결과를 가져옵니다. 애플리케이션 코드에서 해당 소스로 데이터를 푸시하지 않습니다.

### 애플리케이션에서 가져오는 출력

애플리케이션 코드가 결과를 가져와야 할 때 `nodes::Output(...)`를 사용합니다.

```cpp
app.add(neat::nodes::Output("detections"));
auto out = run.pull("detections", 1000);
```

### 그래프가 소유한 출력 대상

그래프가 직접 결과를 기록해야 할 때 출력 대상 노드 또는 그룹을 사용합니다.

```cpp
neat::UdpOutputOptions udp;
udp.host = "192.0.2.10";
udp.port = 5000;

app.add(neat::nodes::UdpOutput(udp));
```

해당 모드에 맞게 구성된 그래프의 경우 서버 스타일의 RTSP 출력을 사용할 수 있습니다.

```cpp
neat::RtspServerHandle server = app.run_rtsp(rtsp_options);
```

## 검증 및 진단

런타임 리소스를 시작하지 않고 구조화된 보고서를 얻고 싶을 때 빌드하기 전에 검증을 수행합니다.

```cpp
neat::GraphReport report = app.validate();
if (!report.error_code.empty()) {
  std::cerr << report.repro_note << "\n";
}
```

잡다 `NeatError` build/run/push/pull 관련 호출 주변:

```cpp
try {
  neat::Run run = app.build();
} catch (const neat::NeatError& e) {
  std::cerr << e.what() << "\n";

  const neat::GraphReport& report = e.report();
  std::cerr << "error_code: " << report.error_code << "\n";
  std::cerr << "hint: " << report.repro_note << "\n";
}
```

유용한 디버깅 도구:

```cpp
std::cout << app.describe() << "\n";
std::cout << app.describe_backend() << "\n";
```

- `describe()`는 공개 그래프 요약(엔드포인트, 조각, 토폴로지)을 출력합니다.
- `describe_backend()`는 생성된 파이프라인 문자열 또는 런타임 라우팅을 디버깅할 때 유용한 하위 수준의 백엔드 세부 정보를 출력합니다.

오류 코드 분류 및 문제 해결 워크플로에 대해서는 [오류 코드](/reference/error-codes/)를 참조하십시오.

## 그래프 구성을 저장하고 로드

`Graph::save(path)`는 공개 그래프 구성(노드, 엔드포인트 이름, 명시적 엔드포인트 에지, 출력 옵션, 결합 정책 및 모델-경로 출처)을 기록합니다.

```cpp
app.save("app.graph.json");

neat::Graph loaded = neat::Graph::load("app.graph.json");
neat::Run run = loaded.build();
```

이 기능은 실행 중인 파이프라인이나 런타임 지표가 아닌 그래프 계획을 저장합니다. 런타임 지표를 사용하려면 런 JSON 내보내기를 사용하세요.

모델 경로 추적 정보가 중요합니다. 모델 조각은 백엔드 코드 조각 목록 그 이상입니다. 모델 아카이브에서 파생된 입력/출력 이름, 경로 옵션, 다중 입력 모델에 대한 입력-경로 프로세서 메타데이터를 포함합니다. 저장된 그래프에 모델 조각이 포함된 경우, Neat은 해당 모델을 다시 로드하는 데 필요한 모델 아카이브 경로와 경로 옵션을 저장합니다. 아카이브가 로드 시 누락된 경우, Neat은 불완전한 경로를 조용히 생성하는 대신 조치 가능한 오류를 발생시킵니다.

## 실행된 내용을 내보내고 시각화

`Run`은 공개 그래프 모양과 최적화된 런타임 모양을 모두 알고 있습니다. CI, 디버깅, 지원 티켓 또는 오프라인 시각화를 위해 버전이 지정된 JSON 아티팩트로 내보낼 수 있습니다.

### 빌드 시점 토폴로지 스냅샷

그래프가 빌드되는 즉시 아티팩트를 얻고 싶을 때는 빌드 시점 내보내기를 사용하세요.

```cpp
neat::RunOptions opt;
opt.run_export.path = "/tmp/startup.graph_run.json";
opt.run_export.label = "startup";

neat::Run run = app.build(opt);
```

이것은 초기 토폴로지 스냅샷입니다. 아직 샘플이 실행되지 않았기 때문에 처리량/지연 시간 카운터가 없을 수 있습니다.

Python:

```python
opt = pyneat.RunOptions()
opt.run_export.path = "/tmp/startup.graph_run.json"
opt.run_export.label = "startup"

run = app.build(opt)
```

### 실행 후 스냅샷 및 지표

작업 부하가 실행되거나 완료된 후 실행 후 내보내기를 사용하세요.

```cpp
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

neat::RunExportOptions export_opt;
export_opt.label = "after_smoke_test";
export_opt.metadata = {{"test_name", "smoke"}};

std::string err;
if (!neat::save_run_json(run, "/tmp/final.graph_run.json", export_opt, &err)) {
  throw std::runtime_error(err);
}
```

파이썬:

```python
run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

export_opt = pyneat.RunExportOptions()
export_opt.label = "after_smoke_test"
export_opt.metadata = {"test_name": "smoke"}

run.save_json("/tmp/final.graph_run.json", export_opt)
```

내보내기 기능은 현재 실행 중인 작업의 스냅샷을 생성합니다. 이 기능은 작업을 중지하지 않습니다. 유한한 작업량에 대한 최종 결과를 얻으려면 저장하기 전에 `run.close_input()`를 호출하여 출력을 정리하거나 `run.stop()`를 호출하십시오.

보드 전력 텔레메트리 데이터를 포함하려면:

```cpp
neat::RunOptions opt;
opt.enable_board_power(/*sample_interval_ms=*/100);

neat::Run run = app.build(opt);
```

JSON 스키마는 `sima.neat.graph_run` 버전 `1`입니다. 스키마는 `schemas/graph_run_v1.schema.json`에 위치하며, CI 검증기는 `tests/perf/tools/graph_run_schema.py`에 위치합니다.

인터넷 연결 없이 그래프 아티팩트를 렌더링합니다.

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json -o /tmp/final.graph_run.html
```

렌더링할 보기를 선택하세요:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view public
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view lowered
```

- `public`은 사용자가 작성한 그래프를 보여줍니다. 여기에는 명명된 입력, 출력, 조각, 그리고 `connect(...)` 연결선이 포함됩니다.
- `lowered`는 Neat이 내부적으로 실행한 내용을 보여줍니다. 여기에는 파이프라인 세그먼트, 생성된 분기/결합 단계, 큐, 그리고 런타임 연결선이 포함됩니다.

## 일반적인 패턴

### 이미지 분류

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(resnet);
app.add(neat::nodes::Output("classes"));
```

### 객체 감지

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### RTSP 카메라에서 모델을 거쳐 앱으로 전송되는 출력

```cpp
neat::nodes::groups::RtspDecodedInputOptions source_opt;
source_opt.url = "rtsp://camera/stream";

neat::Graph app;
app.add(neat::nodes::groups::RtspDecodedInput(source_opt));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### 앱 입력을 그래프가 소유한 UDP 출력으로 전달

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::UdpOutput(udp_options));
```

### 브랜치 미리보기 및 모델 경로

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_image"});
```

모델 경로 입력과 일치하도록 이름 `model_image`를 지정하거나, 명시적인 어댑터 조각을 삽입합니다.

### 왼쪽/오른쪽 스트림 결합

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "pair",
                                         neat::CombinePolicy::ByPts);
```

사용하세요. `ByPts` 미디어 타임스탬프가 동기화 키일 때, 다음을 사용하십시오. `ByFrame` 프레임 ID가 동기화 키일 때.

### GenAI 및 기타 단계별 구성 요소

GenAI 및 기타 비선형/단계 기반 기능은 여전히 애플리케이션 코드에 공개적으로 포함되어야 합니다.
`Graph` 조각들을 모아 실행합니다. `Graph::build() -> Run`:

```cpp
neat::Graph app;
app.add(genai_fragment);

neat::Run run = app.build();
run.push("prompt", prompt_sample);
auto token = run.pull("tokens", 1000);
```

정확한 GenAI 조각 팩토리 및 샘플 헬퍼 이름은 설치된 GenAI 패키지에 따라 다릅니다. 그래프 규칙은 동일합니다. 공개 조각을 추가하거나 연결한 다음, 이름이 지정된 `Run::push(...)` 및 `Run::pull(...)`을 사용합니다.

## 좋지 않은 패턴 및 주의 사항

### 그래프 레이블을 엔드포인트로 사용하지 마십시오.

잘못된 예:

```cpp
neat::Graph image("image");
run.push("image", neat::TensorList{tensor}); // Graph label is not an endpoint.
```

정답:

```cpp
neat::Graph image;
image.add(neat::nodes::Input("image"));
```

### 모델 입력 이름은 추측하지 마세요.

잘못된 예:

```cpp
left.add(neat::nodes::Input("my_left"));
app.connect(left, model);
```

정답:

```cpp
for (const auto& name : model.graph().inputs()) {
  std::cout << name << "\n";
}
```

그런 다음 상위 엔드포인트의 이름을 일치하도록 지정합니다.

### 다중 엔드포인트 그래프에서 이름이 지정되지 않은 푸시/풀을 사용하지 마세요.

잘못된 예:

```cpp
run.push(neat::TensorList{left});
run.push(neat::TensorList{right});
```

정답:

```cpp
run.push("left", neat::TensorList{left});
run.push("right", neat::TensorList{right});
```

### CombinePolicy 없이 실수로 팬인(fan-in)하지 않도록 주의하세요.

잘못된 예:

```cpp
neat::Graph bundle;
bundle.add(neat::nodes::Output("bundle"));

app.connect(left, bundle);
app.connect(right, bundle); // Ambiguous: how should left/right be synchronized?
```

정답:

```cpp
neat::Graph bundle = neat::graphs::Combine({"left", "right"},
                                           "bundle",
                                           neat::CombinePolicy::ByFrame);
```

### 조각 경계를 의미하지 않는 한, 중간에 입력/출력을 삽입하지 마세요.

`Input`과 `Output`은 공개 경계 선언입니다. 재사용 가능한 조각에서 이는 정확히 원하는 동작입니다. 순전히 선형적인 애플리케이션에서 중간에 추가적인 `Output`을 추가하면, 해당 경계가 다른 `connect(...)` 엣지에 의해 소비되지 않는 한, 실제로 데이터를 가져올 수 있는 싱크와 역압력이 발생할 수 있습니다.

### 애플리케이션 코드에서 낮은 수준의 런타임 그래프 API를 사용하지 마세요.

낮은 수준의 런타임 그래프 API를 사용하여 애플리케이션 코드를 작성하거나 가르치는 것을 피하세요.

대신 공개 애플리케이션 인터페이스를 사용하세요.

```cpp
neat::Graph
neat::Run
app.build()
```

## 고급 참고 사항: 경계 구체화

이름이 지정된 `Input` 및 `Output` 노드는 프래그먼트의 공개 계약을 선언합니다. 이들은 버퍼를 이동하는 데 사용되는 런타임 객체보다 상위 수준입니다.

실행 가능한 파이프라인을 구성하기 전에 `Graph::build()`는 경계를 정규화합니다.

| 경계 선언 | 구체화되는 시점 | 생략되는 시점 |
|---|---|---|
| `nodes::Input("name")` | 상위 그래프가 연결되지 않은 경우, 즉 공개 `Run::push("name", ...)` 엔드포인트여야 함 | 상위 그래프가 데이터를 제공하는 경우, 즉 내부 프래그먼트 매개변수일 뿐임 |
| `nodes::Output("name")` | 하위 그래프가 데이터를 사용하지 않는 경우, 즉 공개 `Run::pull("name")` 엔드포인트여야 함 | 하위 그래프가 데이터를 사용하는 경우, 즉 내부 프래그먼트 반환 값일 뿐임 |

생략은 잊어버린다는 의미가 아닙니다. 컴파일러는 출처를 유지하므로 `describe()`를 통해 검증 오류, 지표 및 런타임 내보내기 JSON이 여전히 사용자에게 표시되는 엔드포인트 이름을 참조할 수 있습니다.

이를 통해 재사용 가능한 프래그먼트가 애플리케이션 중간에 숨겨진 appsrc/appsink 스타일의 런타임 I/O를 생성하는 것을 방지합니다. 예를 들어:

```cpp
neat::Graph app;
app.connect(camera, route);
app.connect(route, display);
```

실행 가능한 데이터 경로는 `camera -> route body -> display`이며, 중간에 추가적인 물리적 수신지/송신지가 있습니다. `camera -> route.Input -> route.Output -> display`는 아닙니다.

## API 빠른 참조

### C++

```cpp
// Composition
neat::Graph app("debug_label");
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
app.connect(fragment_a, fragment_b);
app.connect("from_endpoint", "to_endpoint");

// Endpoint inspection
auto graph_inputs = app.inputs();
auto graph_outputs = app.outputs();

// Build/run
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

// Runtime endpoint inspection
auto run_inputs = run.input_names();
auto run_outputs = run.output_names();

// Validation/debug/export
neat::GraphReport report = app.validate();
std::cout << app.describe() << "\n";
app.save("app.graph.json");
neat::save_run_json(run, "/tmp/app.graph_run.json");
```

### 파이썬

```python
app = pyneat.Graph("debug_label")
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

print(app.inputs())
print(app.outputs())

run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

print(run.input_names())
print(run.output_names())

app.save("app.graph.json")
run.save_json("/tmp/app.graph_run.json")
```

## 추가 자료

- [모델 프로그래밍 모델](/develop-apps/development-workflow/model)
- [노드 프로그래밍 모델: 그룹 및 경계](/develop-apps/development-workflow/node#boundary-nodes)
- [텐서 및 샘플 프로그래밍 모델](/develop-apps/development-workflow/core_types)
- [런타임 조정(튜토리얼 016)](/tutorials/tune-throughput-and-queues)
- [문제 진단 (튜토리얼 012)](/tutorials/diagnose-a-pipeline)
- [GStreamer 레이어](/develop-apps/advanced-concepts/gstreamer_layer)
