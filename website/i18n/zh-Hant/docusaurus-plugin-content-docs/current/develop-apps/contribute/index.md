---
title: "貢獻"
description: "關於 Neat 框架架構、建置、測試和發布流程的貢獻者指南。"
sidebar_position: 1
slug: /develop-apps/contribute/
---

# 貢獻

本節適用於對 Neat Library 程式碼庫本身進行修改的貢獻者。它說明了程式碼庫的結構、如何建置和測試修改，以及應用程式開發人員必須維持哪些合約的穩定性。

<div class="overview-section-label">貢獻者說明會</div>

**Neat Library** 是此儲存庫中的 C++/Python 函式庫和執行階段。
它會載入模型封存檔，組成和驗證管線，在 Modalix 硬體上執行，並公開公用應用程式 API。Neat SDK 和 DevKit Sync 是周圍的開發工作流程。

在修改此儲存庫時，請針對有助於人類和代理協助開發的框架屬性進行最佳化：明確的 API、確定性行為、結構化的診斷、嚴格的驗證以及穩定的公用合約。

<div class="overview-section-label">貢獻者原則</div>

- **確定性至關重要** — 保持元件名稱、產生的管線、報告和測試的可重現性。
- **可除錯性是首要考量** — 錯誤應產生結構化資料，而不僅僅是字串。
- **不允許靜默回退** — 不要隱藏模型輸入錯誤或硬體/執行階段失敗。
- **在執行前進行驗證** — 在執行階段之前，找出結構、上限、形狀和合約錯誤。
- **公開 API 必須保持穩定** — 在 `include/*` 下安裝的標頭檔，需要進行增量且相容的變更。
- **並行處理必須受到限制且可觀察** — 關閉時不應發生死鎖，且診斷工具必須是執行緒安全的。

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>從這裡開始</h2>
    <p>在修改程式碼之前，請先了解程式碼庫、命名規則和相關規範。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>架構</strong><span>學習程式碼庫的結構、模組的所有權、執行階段流程以及擴充點。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/naming/"><strong>命名合約</strong><span>始終如一地使用標準化的產品、API、套件和類型名稱。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/coding_standard/"><strong>程式碼規範</strong><span>請遵循 C++ 程式碼風格、公開 API、相容性以及檔案編寫規範。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>建置與測試</h2>
    <p>建立框架、驗證變更，並處理 Python 繫結。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/build/"><strong>建立</strong><span>從原始碼建構 Neat，然後選擇 CMake 設定檔或建構選項。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/test_requirements/"><strong>測試需求</strong><span>了解每種變更預期需要進行哪些測試和檔案更新。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/python_bindings/"><strong>Python 繫結</strong><span>作為貢獻者，建立、測試和封裝 PyNeat 的綁定。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>合約與內部事務</h2>
    <p>在變更模型封存檔、外掛合約或內部封裝時，請使用這些設定。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/mpk_contract/"><strong>MPK 合約</strong><span>了解模型封存檔的導入、驗證和安全規則。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/sima_plugin_json_truth_map/"><strong>外掛程式 JSON 真實資料對應</strong><span>檢閱已凍結的 SIMA 外掛程式 JSON 合約映射。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/appcomplex_workspace_packaging/"><strong>AppComplex 包裝</strong><span>建立並安裝具有閘道功能的應用程式群組工作區服務套件。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>發布與維護</h2>
    <p>將這些用於發布流程、清理計畫和長期設計指導。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/release-checklist/"><strong>發布檢查清單</strong><span>請遵循阻礙發布的條件、所需的檢查項目，以及可重複執行的步驟。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/error-taxonomy-rollout/"><strong>錯誤分類法推出</strong><span>追蹤結構化錯誤代碼的遷移和驗證狀態。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/agentic-workflow/"><strong>代理人工作流程</strong><span>了解為什麼 API 的結構設計是為了方便人類和 AI 輔助開發。</span></a></li>
    </ul>
  </section>
</div>
