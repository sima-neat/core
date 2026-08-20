---
title: "開發應用程式"
description: "在 Modalix 上使用 SiMa.ai，建立並執行人工智慧應用程式，讓一切變得 Neat。"
sidebar_position: 1
---

# 使用 SiMa.ai 開發應用程式，讓開發過程更 Neat。

<LanguageContent lang="cpp">

<div className="overview-workflow-image overview-workflow-image-light">

![使用 SiMa.ai Neat 組成的 C++ 應用程式工作流程](@site/../docs/develop-apps/images/neat-overview-workflow-cpp.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![使用 SiMa.ai Neat 組成的 C++ 應用程式工作流程](@site/../docs/develop-apps/images/neat-overview-workflow-cpp-dark.svg)

</div>

</LanguageContent>

<LanguageContent lang="py">

<div className="overview-workflow-image overview-workflow-image-light">

![使用 SiMa.ai Neat 組成的 Python 應用程式工作流程](@site/../docs/develop-apps/images/neat-overview-workflow-python.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![使用 SiMa.ai Neat 組成的 Python 應用程式工作流程](@site/../docs/develop-apps/images/neat-overview-workflow-python-dark.svg)

</div>

</LanguageContent>

## 什麼是 SiMa.ai Neat

SiMa.ai Neat 是一個應用程式開發框架，用於建置和執行在 SiMa.ai 平台上執行的 AI 應用程式。它提供 Python 和 C++ API，用於載入和執行編譯後的模型封存檔 (`.tar.gz`)，組成使用 Modalix 處理資源的端到端應用程式，以及管理執行階段執行。

在更廣泛的 SiMa.ai 軟體堆疊中，SiMa.ai Neat 位於應用程式層。它建立在 SiMa.ai 執行階段堆疊之上，並在其下方使用 GStreamer，因此開發人員可以專注於應用程式邏輯，而不是手動連接較低層級的執行階段元件。

若要採用最短的推論路徑，請將編譯後的模型封存檔載入為 `Model` 並直接執行。當應用程式需要多個輸入、處理階段、模型或輸出時，請將這些元件組成 `Graph`，再建置為 `Run`。同一組公開 API 同時支援傳統與代理式開發，因此團隊可以使用任一工作流程來檢視、擴充及維護應用程式。

### 選擇您的部署模型

- **在 Modalix DevKit 上執行** — 應用程式和 Neat 圖直接在設備上執行。使用 `simaai::neat` 或 `pyneat`，以及 `Model`、`Graph`、`Node` 和 `Run`。從 [執行／推論](/develop-apps/development-workflow/overview/) 開始。
- **使用 Modalix PCIe 擴充卡進行協同處理** — 應用程式在主機機器上執行，並將張量或圖像發送到擴充卡以進行模型執行。使用 `simaai::neat::pcie` 或 `pyneatpcie`。從 [PCIe 協同處理](/develop-apps/development-workflow/pcie-model/) 開始。

### C++ 或 PyNeat

對於直接在 Modalix DevKit 上執行的應用程式，SiMa.ai Neat 通過兩種語言介面提供相同的核心工作流程，因此您可以選擇適合您應用程式的介面：

- **PyNeat** — Python 綁定 (`pyneat`)。最適合快速迭代、筆記本、資料科學工作流程以及直接在 DevKit 上執行 Python 應用程式。
- **C++** — 本機 `simaai::neat` API。最適合大型應用程式、與現有 C++ 程式碼庫的緊密整合以及跨編譯的主機到 DevKit 工作流程。

兩者都使用相同的編譯模型成品和 Modalix 執行階段；以下的概念和頁面適用於任一者。PCIe 協同處理通過 `simaai::neat::pcie` 和 `pyneatpcie` 提供單獨的 C++ 和 Python 介面。

## 開發應用程式。 <span className="neat-heading-highlight">SiMa.ai Neat 它會為您繪製地圖。</span>

Modalix 將應用程式核心、視覺處理、機器學習加速、視訊引擎、共享記憶體和高速 I/O 整合在單一的 SoC 中。透過其 Python 和 C++ API，SiMa.ai Neat 提供一個程式設計模型，用於在系統中建置跨應用程式相關處理資源的應用程式。

從相機或網路串流建立一個端到端的流程，經過處理和推論，最終產生結果。SiMa.ai Neat 建置執行階段管線，在適用的情況下選擇加速的實作，並協調在 Modalix 上的執行和資料移動。您專注於應用程式，而 SiMa.ai Neat 則處理底層硬體和執行階段的複雜性。

<div className="overview-workflow-image modalix-application-map-desktop">

![對應至 MLSoC Modalix 平面設定的 SiMa.ai Neat 應用程式](@site/../docs/images/neat-modalix-floorplan-animated.svg)

</div>

<div className="overview-workflow-image modalix-application-map-mobile">

![對應至 MLSoC Modalix 平面設定之 SiMa.ai Neat 應用程式的行動版檢視](@site/../docs/images/neat-modalix-floorplan-mobile-animated.svg)

</div>

<p className="overview-figure-caption"><strong>說明性對應：</strong> 所選擇的路線取決於應用程式、模型以及可用的硬體加速功能。請參閱。 <a href="/develop-apps/advanced-concepts/processor_backends/">處理器後端</a> 用於技術對應。</p>

## 請描述您的應用程式。 <span className="agentic-heading-highlight">一位具備 Neat 技能的代理人會開發它。</span>

SiMa.ai Neat 透過內建的技能，支援即開即用的代理應用程式開發。 Neat 開發環境（以下簡稱為） Neat SDK)。這些技能讓程式碼代理程式能夠理解如何使用公開的 Python 和 C++ API，遵循既定的應用程式模式，並與其他程式碼協同運作。 Modalix 開發與驗證
工作流程。

建議的代理路徑可以建立一個應用程式，並在配對的
Modalix DevKit，檢查結果和診斷資訊，並最佳化實作方式。傳統的開發方式仍然是一種平行途徑，可透過相同的 API 進行直接控制。這兩種方式都能產生標準且可檢查的 SiMa.ai Neat 應用程式，因此您可以檢閱或修改代理程式開發的程式碼，並隨著應用程式的發展，在兩種工作流程之間切換。請參閱。 [設定 Neat SDK。](/getting-started/dev-environment/) 以促進具主動性的發展。

<div className="overview-workflow-image agentic-visual-desktop">

![程式設計代理程式建立、執行、診斷並改進 SiMa.ai Neat 應用程式](@site/../docs/images/agentic-development-loop-animated.svg)

</div>

<div className="overview-workflow-image agentic-visual-mobile">

![SiMa.ai Neat 代理式開發迴圈的行動版檢視](@site/../docs/images/agentic-development-loop-mobile-animated.svg)

</div>

## 需求

在建立應用程式之前，請完成「開始使用」設定：

- **針對部署模型進行安裝** — 對於 Modalix DevKit，請在 Neat SDK 中或直接在裝置上安裝 Neat Library。對於 PCIe 協同處理，請在主機機器上安裝 `core/pciehost`，並在 Modalix PCIe 卡上安裝相容的 Neat Library。
- **模型成品** — 使用來自 Model Zoo 的預先編譯模型，或將您自己的模型編譯成 Modalix 格式的檔案。
- **執行階段目標** — 在 DevKit 上執行原生應用程式，或在主機機器上直接建立並執行協同處理應用程式。在 Neat SDK 中跨編譯原生 C++ 應用程式時，請配對並同步 DevKit。

「Hello Neat！」頁面可幫助您執行第一個推論，開發流程頁面更詳細地說明了主要概念，而教學課程則展示了如何將這些概念應用於實際應用程式模式。

您可以瀏覽 [應用範例](https://developer.sima.ai/examples)，以研究、調整和執行完整的應用程式。

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>從這裡開始</h2>
    <p>從一個可運作的環境開始，並建立核心的 SiMa.ai Neat 應用程式工作流程。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>您好，Neat！</strong><span>執行一個簡化的 Neat 應用程式，並驗證開發迴圈。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/development-workflow/overview/"><strong>開發流程</strong><span>更深入地了解 `Model`、`Graph` 和 `Run` 的工作流程。</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>教學指南</strong><span>請按照提供的範例進行操作，這些範例會逐步說明實際操作流程。 SiMa.ai Neat 應用模式。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-explore">
    <h2>建造更多</h2>
    <p>當您準備好建立更豐富的應用程式或檢視 API 介面時，請使用這些章節。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/"><strong>進階概念</strong><span>了解圖、格式、記憶體、執行緒和執行階段行為。</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>參考</strong><span>瀏覽 C++、Python、Model Compiler、疑難排解，以及相關輔助資料。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>貢獻</strong><span>了解軟體架構、原始碼建置、測試期望以及程式碼庫規範。</span></a></li>
    </ul>
  </section>
</div>
