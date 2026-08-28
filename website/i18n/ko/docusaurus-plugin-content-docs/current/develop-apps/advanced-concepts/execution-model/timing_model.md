---
title: "비동기 방식과 동기 방식의 타이밍 모델"
description: "`Graph.run(...)`과 `Graph.build(...)`/`Run`이 어떻게 관련되는지 — 작업이 수행될 때, 결과가 반환될 때를 기준으로 설명합니다."
sidebar_position: 1
slug: /develop-apps/advanced-concepts/timing_model
---

# 비동기 방식과 동기 방식의 타이밍 모델

`Graph`와 `Run`은 작업을 실행하는 두 가지 방법을 제공합니다.

- **단일 실행**: `Graph.run(input, ...)`은 입력을 전달하고 한 번의 호출로 출력을 기다립니다. 입력이 없는 `Graph.run()`은 EOS(End of Stream)에 도달할 때까지 소스에서 소유한 그래프를 실행합니다.
- **재사용 가능한 실행**: `Graph.build(...)`는 활성 상태의 `Run`을 반환합니다. 애플리케이션은 입력을 전달하고, 출력을 가져오고, 입력을 닫고, 데이터를 비우고, 측정하고, 실행을 중지합니다.

두 모드 모두 동일한 `Graph` 계획과 동일한 하드웨어를 사용합니다. 차이점은 루프를 누가 소유하느냐입니다.

## 단일 실행 모드

하나의 입력이 있고 가장 짧은 올바른 경로를 원할 때 단일 실행 모드를 사용합니다.

```cpp
simaai::neat::Graph graph("classifier");
graph.add(simaai::neat::nodes::Input("image"));
graph.add(model);
graph.add(simaai::neat::nodes::Output("classes"));

simaai::neat::TensorList out = graph.run(
    simaai::neat::TensorList{image_tensor});
```

소스가 소유한 그래프의 경우, 입력 없이 `Graph.run()`을 호출합니다.

```cpp
simaai::neat::Graph graph("file_job");
graph.add(source_fragment);
graph.add(model);
graph.add(sink_fragment);

graph.run();  // Blocks until EOS or error.
```

다음과 같은 경우에 원샷 모드를 사용하세요.

- 간단한 테스트;
- 짧은 검증 실행;
- 완료될 때까지 실행되어야 하는 소스에서 관리하는 작업;
- 장기간 유지되는 런타임 핸들을 관리하지 않아야 하는 코드.

## 재사용 가능한 실행 모드

애플리케이션이 루프를 관리하는 경우 `Graph.build(...)`를 사용하세요.

```cpp
auto run = graph.build();

while (have_more_inputs()) {
  run.push("image", simaai::neat::TensorList{next_tensor()});

  if (auto out = run.pull("classes", /*timeout_ms=*/0)) {
    consume(*out);
  }
}

run.close_input();
while (auto out = run.pull("classes", /*timeout_ms=*/1000)) {
  consume(*out);
}
run.close();
```

다음과 같은 경우 재사용 모드를 사용하세요.

- 실시간 비디오, RTSP 또는 카메라 입력;
- 애플리케이션이 속도를 제어하는 스트림 처리;
- 다중 입력 또는 다중 출력 그래프;
- 처리량 테스트;
- 측정, 내보내기, 비우기 및 중지 제어.

## 푸시 타이밍

`Run::push(...)`는 입력이 그래프 경계에서 수락된 후 반환됩니다. 입력이 모든 노드를 통과할 때까지 기다리지 않습니다.

입력 큐가 가득 찼을 때:

- `OverflowPolicy::Block`는 생성자에 역압력을 적용합니다.
- `OverflowPolicy::DropIncoming`는 새 입력을 거부합니다.
- `OverflowPolicy::KeepLatest`는 오래된 큐에 있는 입력을 삭제하여 실시간 경로가 최신 상태를 유지하도록 합니다.
- `try_push(...)`는 차단하는 대신 `false`를 반환합니다.

소스에 맞는 정책을 선택하세요. 파일 및 배치 작업은 일반적으로 `Block`를 사용합니다. 실시간 스트림은 일반적으로 최신 상태 유지 정책을 사용합니다.

## 풀 타이밍

`Run::pull(...)`는 출력 경계에서 다음 사용 가능한 `Sample`을 반환합니다.

타임아웃과 EOS가 동일한 “샘플 없음” 경로를 공유할 수 있는 경우 편리한 오버로드를 사용하세요.

```cpp
if (auto sample = run.pull("classes", /*timeout_ms=*/1000)) {
  consume(*sample);
}
```

타임아웃, 종료, 오류가 발생했을 때 각각 다르게 처리해야 하는 경우, 구조화된 상태 오버로드를 사용하세요.

```cpp
simaai::neat::Sample sample;
simaai::neat::PullError error;

switch (run.pull("classes", /*timeout_ms=*/1000, sample, &error)) {
case simaai::neat::PullStatus::Ok:
  consume(sample);
  break;
case simaai::neat::PullStatus::Timeout:
  break;
case simaai::neat::PullStatus::Closed:
  break;
case simaai::neat::PullStatus::Error:
  throw std::runtime_error(error.message);
}
```

숨겨진 대체 기능 없음: 다중 출력 그래프는 명확하게 하나의 출력만 있는 경우가 아니라면 명명된 `pull("output", ...)`이 필요합니다.

## 텔레메트리: 사용자가 소유한 루프 측정

사용자의 앱이 소유한 워크로드 주변에 `Run::start_measurement(...)`를 사용합니다. 반환되는 `MeasureReport`는 공개 타이밍 표면이며 다음을 포함합니다.

- 전체 푸시-출력 지연 시간 및 처리량
- 푸시, 풀, 삭제된 샘플에 대한 런타임 카운터
- `MeasureOptions`에서 요청할 경우 플러그인/커널 및 에지 타이밍
- 실행 시 전력 모니터링이 활성화된 경우 선택적 전력 텔레메트리

측정되는 핵심 루프 외부에서 설정, 파일 다운로드, 프레임별 로깅 및 보고서 내보내기를 수행하되, 질문이 전체 애플리케이션 비용에 관한 것이라면 예외입니다.

## 추가 정보

- [그래프 실행](/develop-apps/development-workflow/pipeline) — 라이프사이클, 옵션, 역압, 측정 및 처리량.
- [`Graph::run()`](/reference/cppapi/classes/simaai-neat-graph) — 단일 실행 및 소스 소유 진입점.
- [`Graph::build()`](/reference/cppapi/classes/simaai-neat-graph) — 재사용 가능한 실행 진입점.
- [`Run`](/reference/cppapi/classes/simaai-neat-run) — 푸시, 풀, 닫기, 드레인, 중지 및 측정.
