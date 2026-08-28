---
title: "고급 개념"
description: "더욱 풍부한 기능을 갖춘 Neat 애플리케이션을 구축하기 위한 디자인 세부 사항"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/
---

# 고급 개념

애플리케이션에 기본적인 `Model`, `Graph`, `Run` 워크플로우보다 더 많은 기능이 필요할 때 이 페이지들을 활용하세요. 이 페이지들은 더욱 복잡한 Neat 애플리케이션의 작동 방식과 런타임 동작에 대한 계약 내용을 설명합니다.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-app">
    <h2>애플리케이션 디자인</h2>
    <p>애플리케이션이 어떻게 구성되고, 어떤 방식으로 결과를 출력하는지 설계합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/graphs/"><strong>그래프</strong><span>모델, 노드, 명명된 입력 및 출력, 분기, 그리고 실행을 구성합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mipi-camera-input/"><strong>MIPI 카메라를 사용하세요.</strong><span>libcamera 입력을 사용하고 CVU 전처리를 통해 소스에서 직접 카메라-모델 그래프를 구축합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/video_sender/"><strong>비디오 보내기</strong><span>Neat 애플리케이션에서 H.264 RTP/UDP를 통해 비디오 출력을 스트리밍합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/metadata_sender/"><strong>JSON 메타데이터 전송</strong><span>UDP JSON을 통해 구조화된 애플리케이션 메타데이터를 게시합니다.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>실행 모델</h2>
    <p>작업이 어떻게 예약되고, 스레드로 처리되며, 파이프라인 계층에 매핑되는지 이해합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/timing_model/"><strong>타이밍 모델</strong><span>동기 및 비동기 실행 방식, 푸시/풀 방식, 그리고 작업이 수행되는 시점을 이해합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/threading/"><strong>스레드 모델</strong><span>어떤 스레드가 존재하는지, 그리고 애플리케이션 코드가 어디에서 실행될 수 있는지 파악하십시오.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/gstreamer_layer/"><strong>GStreamer 레이어</strong><span>Neat이 어떤 부분을 추상화하는지, 그리고 원본 GStreamer의 세부 정보가 언제 중요한지 알아보세요.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>데이터 및 모델 계약</h2>
    <p>파이프라인이 사용하는 텐서, 메모리, 모델 계약을 이해합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/data_formats/"><strong>데이터 형식</strong><span>맵 형식 토큰을 텐서 레이아웃, 형태 및 평면 의미론에 매핑합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/dtype_contract/"><strong>데이터 유형 계약</strong><span>전처리, MLA, 후처리 단계에서 텐서 정밀도가 어떻게 변화하는지 확인합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/memory_model/"><strong>메모리 모델</strong><span>제로 복사 버퍼, 물리 주소, 캐시 동작 방식을 이해합니다.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-start">
    <h2>모델 런타임</h2>
    <p>컴파일된 모델 아티팩트와 이를 실행하는 하드웨어 백엔드에 대해 더 자세히 알아보세요.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mpk_contract/"><strong>MPK 계약</strong><span>컴파일된 모델 아카이브가 추론 단계와 계약을 어떻게 정의하는지 확인해 보세요.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/processor_backends/"><strong>프로세서 백엔드</strong><span>A65, CVU, MLA, MLASHM, APU, TVM 및 M4의 역할을 이해합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/cvu_kernels/"><strong>CVU 커널</strong><span>전처리 및 후처리 그래프 빌드 블록을 검토합니다.</span></a></li>
    </ul>
  </section>
</div>
