---
title: "Palette Neat"
sidebar_position: 1
---

# Palette Neat

Palette Neat 是 SiMa.ai 為在 Modalix 上建置 AI 應用程式所提供的軟體開發工具組。
其中包含開發環境、執行階段程式庫、模型工具，以及 DevKit 驗證工作流程。這些元件共同
支援完整的開發流程：準備模型、建置應用程式，並在 Modalix 硬體上驗證結果。

您可以透過本概覽瞭解 Palette Neat 的主要組成，並選擇合適的環境設定、模型準備或
應用程式開發路徑。

![由 Neat SDK、Neat Library、Model Compiler、LLiMa 與 Modalix DevKit 組成的 Palette Neat 軟體堆疊](@site/../docs/images/neat-software-stack-animated.svg)

<div class="overview-section-label">開發者旅程</div>

:::tip 如果您是第一次使用 Palette Neat
使用 Palette Neat 開發應用程式有兩種支援方式：

- 若您需要效能更高的開發環境，特別是預計編譯模型或交叉編譯大型 C++ 程式碼，請選擇
  **[使用 Neat SDK](/getting-started/dev-environment/)**。
- 若您希望減少開發環境中的組成元件，尤其是不需要進行模型編譯時，請選擇
  **[直接在 DevKit 上開發](/getting-started/neat-library/)**。

若選擇 SDK 路徑，請先確認主機符合[主機需求](/getting-started/dev-environment/#host-requirements)，
再安裝 SDK。SDK 安裝會套用相容的預設值，因此只有在需要鎖定確切版本、個別升級元件，
或疑難排解版本不相容問題時，才需要查閱相容性參考資料。

**僅使用 SDK 的快速路徑（無 DevKit）：** 安裝 SDK 後，直接前往
[編譯模型](/compile-a-model/)。設定 SDK、DevKit Sync、Neat Library 與 PyNeat 都是
選擇性步驟，在配對 DevKit 前可以先略過。
:::

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>從這裡開始</h2>
    <p>準備主機、Neat SDK 與 DevKit，以進行本機開發和硬體驗證。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/"><strong>Neat SDK</strong><span>使用 SDK 進行高效能的主機式工作流程、模型編譯、C++ 建置和 DevKit 驗證。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/"><strong>Neat Library</strong><span>若要以較少的組成元件在 DevKit 上製作應用程式原型，可直接安裝執行階段與 PyNeat。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/compatibility/"><strong>相容性指南</strong><span>支援版本組合的參考指南。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>模型準備</h2>
    <p>將訓練完成的模型轉換成可部署並在 Modalix 硬體上執行的成品。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/compile-a-model/"><strong>編譯模型</strong><span>為 Modalix 編譯預先訓練的 ONNX 視覺模型或 GenAI 模型。</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-zoo/"><strong>使用預先編譯的模型</strong><span>使用可立即執行的模型成品快速開始。</span></a></li>
      <li><a class="overview-link-card" href="/genai-llima/"><strong>使用 LLiMa 的 GenAI</strong><span>在 Modalix 上編譯、測試及效能評測 GenAI 模型。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>建置應用程式</h2>
    <p>使用 Neat Library 執行模型，並組合正式環境的應用程式管線。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Hello Neat!</strong><span>執行您的第一個 Neat 應用程式，並驗證開發工作流程。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/"><strong>開發應用程式</strong><span>使用 C++ 或 PyNeat，以 Neat Library 建置 AI 應用程式。</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>教學課程</strong><span>依照引導式範例，學習實際的 Neat 應用程式模式。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>工具與參考資料</h2>
    <p>需要更多細節時，可使用支援工具與參考資料。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/tools/"><strong>工具</strong><span>sima-cli、Model Zoo 與 Insight。</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>參考資料</strong><span>瀏覽 API、疑難排解、環境變數、資料格式與版本資訊。</span></a></li>
      <li><a class="overview-link-card" href="/reference/troubleshooting/"><strong>疑難排解</strong><span>尋找設定、執行階段與管線問題的解決方法。</span></a></li>
    </ul>
  </section>
</div>
