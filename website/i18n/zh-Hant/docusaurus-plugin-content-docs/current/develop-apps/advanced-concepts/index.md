---
title: "進階概念"
description: "設計更豐富的 Neat 應用程式時的細節考量"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/
---

# 進階概念

當您的應用程式需要超出基本 `Model`、`Graph` 和 `Run` 工作流程時，請使用這些頁面。它們解釋了更豐富的 Neat 應用程式背後的合約和執行階段行為。

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-app">
    <h2>應用程式設計</h2>
    <p>設計您的應用程式如何組成，以及它如何輸出結果。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/graphs/"><strong>圖</strong><span>建立模型、節點、具名輸入和輸出、分支，以及執行流程。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mipi-camera-input/"><strong>使用 MIPI 相機</strong><span>使用 libcamera 輸入和 CVU 預處理，建立一個由原始資料所擁有的相機到模型的圖。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/video_sender/"><strong>傳送影片</strong><span>透過 H.264 RTP/UDP，將 Neat 應用程式的視訊輸出串流傳輸。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/metadata_sender/"><strong>傳送 JSON 中繼資料</strong><span>透過 UDP JSON 格式發布結構化的應用程式中繼資料。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>執行模型</h2>
    <p>了解工作如何被排程、以執行緒方式處理，以及如何映射到管線層。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/timing_model/"><strong>時序模型</strong><span>了解同步和非同步執行、推送/拉取，以及工作發生的時機。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/threading/"><strong>執行緒模型</strong><span>了解有哪些執行緒存在，以及應用程式碼可能在哪裡執行。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/gstreamer_layer/"><strong>GStreamer 層級</strong><span>了解 Neat 抽象了哪些內容，以及何時原始的 GStreamer 細節變得重要。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>資料與模型合約</h2>
    <p>了解管線所依賴的張量、記憶體和模型合約。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/data_formats/"><strong>資料格式</strong><span>將地圖格式的標記對應到張量的佈局、形狀和平面語義。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/dtype_contract/"><strong>資料類型合約</strong><span>追蹤張量精確度在預處理、MLA 和後處理各階段的變化。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/memory_model/"><strong>記憶體模型</strong><span>了解零拷貝緩衝區、實體位址和快取行為。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-start">
    <h2>模型執行階段</h2>
    <p>深入探討已編譯的模型成品，以及執行這些成品的硬體後端。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mpk_contract/"><strong>MPK 合約</strong><span>請參閱編譯後的模型封存檔如何定義推論階段和合約。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/processor_backends/"><strong>處理器後端</strong><span>了解 A65、CVU、MLA、MLASHM、APU、TVM 和 M4 的職責。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/cvu_kernels/"><strong>CVU 核心</strong><span>檢視預處理和後處理圖的建構模組。</span></a></li>
    </ul>
  </section>
</div>
