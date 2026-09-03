---
title: "GStreamer 아래에"
description: "이 프레임워크가 GStreamer를 통해 어떤 부분을 추상화하고, 어떤 부분이 그대로 드러나는지, 그리고 언제 원시 GStreamer를 사용해야 하는지 설명합니다."
sidebar_position: 3
slug: /develop-apps/advanced-concepts/gstreamer_layer
---

# GStreamer 기반

Neat 프레임워크의 파이프라인은 GStreamer 위에서 실행됩니다. 애플리케이션에서 보이는 거의 모든 것, 즉 `Graph`, `Run`, `Node`는 GStreamer 개념을 래핑한 것입니다. 이 페이지에서는 레이어링을 설명합니다. 즉, 무엇이 숨겨지고, 무엇이 숨겨지지 않으며, 언제 원시 GStreamer가 올바른 해결책인지 설명합니다.

## 프레임워크가 추상화하는 것

| GStreamer 개념 | 프레임워크 추상화 |
|---|---|
| `gst-launch` 텍스트 조각 | `Node::backend_fragment(int node_index)` |
| 요소 이름(`name=…`) | `Node::element_names()`에서 결정적인 `n<idx>_<role>` |
| 파이프라인 문자열(조각 연결) | `Graph::add()`는 파이프라인을 구축하고 연결합니다. |
| Caps 협상 | `Graph::build()`는 `NodeCapsBehavior`를 통해 caps를 검증합니다. |
| `gst_pipeline_set_state()` | `Graph::run()` / `Run::start()` |
| 버스 메시지 | `GraphReport::bus_messages` |
| `appsrc` 푸시 API | `Run::push()` ( `InputRole::Push`가 있는 노드에서만) |
| `appsink` 풀 API | `Run::pull()` |
| 요소별 타이밍 | `Run::start_measurement()`에서 `MeasureReport` |

애플리케이션은 런처 문자열을 직접 작성하거나, 요소를 직접 지정하거나, GStreamer C API에 직접 액세스하지 않습니다. 모든 것은 노드를 통해 이루어집니다.

## 드러나는 것

프레임워크가 숨기지 않거나 숨겨서는 안 되는 몇 가지 GStreamer 개념:

- **Caps 의미** - 비디오/오디오 caps가 포함하는 필드. 애플리케이션 코드는 [`FormatTag`](/reference/cppapi/files/include-pipeline-formatspec-h)를 읽고 관련 caps 필드를 반영하는 `Sample` 메타데이터를 검사할 수 있습니다.
- **버퍼 플래그** - 불연속성, EOS, 간격. 프레임워크는 이러한 플래그를 `Sample`에 전달하여 애플리케이션 코드가 스트림 경계에 반응할 수 있도록 합니다.
- **이벤트 순서** - GStreamer는 이벤트(caps, 세그먼트, EOS)가 버퍼와 함께 순서대로 흐르도록 보장합니다. 프레임워크는 풀 측에서 이 순서를 유지합니다.

빌드된 그래프에 대한 정확한 GStreamer 런처 문자열이 필요한 경우 `Graph::describe()`를 호출하십시오. 그러면 파이프라인을 바이트 단위로 재현하는 결정적인 `gst-launch` 재현기가 생성됩니다.

## 원시 GStreamer를 사용해야 하는 경우

일반적인 애플리케이션 코드에서는 필요하지 않습니다. 적절한 경우는 다음과 같습니다.

- **사용자 지정 GStreamer 플러그인** - 프레임워크가 노드로 제공하지 않는 GStreamer 요소를 원할 경우, 플러그인을 래핑하고 올바른 `backend_fragment()`를 출력하는 노드 서브클래스를 작성하십시오. "사용자 지정 노드 빌드"를 디자인 심층 분석(§0.10)에서 참조하십시오.
- **진단 도구** - `Graph::describe()`에서 생성된 `repro_gst_launch` 재현기는 GStreamer가 소비할 런처 문자열과 정확히 동일합니다. 오프라인 디버깅을 위해 `gst-launch-1.0`에 붙여넣을 수 있습니다.
- **플러그인 작성** - SiMa의 자체 GStreamer 플러그인( `sima*` 제품군)은 플러그인 매니페스트 ABI( [`gst/SimaPluginStaticManifestAbi.h`](/reference/cppapi/files/include-gst-simapluginstaticmanifestabi-h) 참조)에 설명되어 있으며 프레임워크에서 자동으로 로드됩니다.

## 결정론적 보장

프레임워크의 요소 명명 방식은 결정적입니다. 즉, 동일한 옵션을 가진 동일한 노드 목록은 항상 동일한 `gst-launch` 문자열을 생성합니다. 이것이 다음을 가능하게 합니다.

- `repro_gst_launch` 필드를 실제로 재현 가능하게 만듭니다.
- 테스트 스냅샷을 여러 실행에 걸쳐 안정적으로 유지합니다.
- 요소 식별(예: 측정 속성 지정)을 기계가 쉽게 처리할 수 있도록 합니다.

규칙은 `n<node_index>_<role>`이며, 여기서 `role`은 노드 작성자가 선택하는 짧고 안정적인 식별자입니다. 이 명명 방식에 참여하는 공개 노드 래퍼는 [노드 API 그룹](/reference/cppapi/groups/nodes)에서 확인할 수 있습니다.

## 추가 정보

- "GStreamer 추상화" — 디자인 심층 분석의 §0.8.
- [노드 API](/reference/cppapi/groups/nodes) — 결정적인 백엔드 조각을 생성하는 구체적인 노드 래퍼.
- [`Graph::describe()`](/reference/cppapi/classes/simaai-neat-graph) — 실행 문자열을 출력합니다.
- "SiMa 플러그인 매니페스트" — 디자인 심층 분석의 §51 및 §95.
