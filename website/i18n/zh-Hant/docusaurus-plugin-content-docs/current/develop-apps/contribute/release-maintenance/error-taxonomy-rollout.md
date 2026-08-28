---
title: "錯誤分類法推出"
description: "結構化的錯誤碼遷移與驗證清單"
sidebar_position: 2
slug: /develop-apps/contribute/error-taxonomy-rollout
---

# 錯誤分類架構推出

此檢查清單追蹤核心和執行階段外掛程式中標準錯誤語義的推出進度。

## 標準程式碼

[`include/pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h) 是權威來源。[錯誤代碼目錄](/reference/error-codes) 必須記錄每個 C++ 常數、每個 Python `ERROR_*` 名稱，以及從粗略到具體程式碼的轉換。

## 執行片段

1. 建立錯誤分類架構
2. 建立/驗證程式碼
3. 執行階段提取程式碼
4. 圖形 I/O 解析器/開啟程式碼
5. 測試 + 檔案

## 相容性審查

- 將現有錯誤所傳回的確切程式碼的變更視為行為上的重大變更，即使沒有 C++ 或 Python 簽名的變更。
- 在公開的轉換表中記錄舊程式碼到新程式碼的對應關係。
- 僅針對沒有特定分類的錯誤，保留回退程式碼（`build.parse_launch`、`runtime.pull` 和 `runtime.element_failed`）。
- 在生產建構器中測試版本化的 `simaai-neat-error` 訊息金鑰，並透過核心解析實際的 `GstMessage`。

## 驗證檢查清單

- 在終端框架錯誤時，`NeatError.report().error_code` 不應為空。
- 在執行階段提取錯誤時，`PullError.code` 應包含值。
- 圖形包裝器錯誤應包含程式碼 + 內容 + 提示（沒有通用的回退文字）。
- JSON 解析失敗應包含 `offset=` 和 `near='...'`。
- 負面測試應確認每個錯誤分類的程式碼 + 穩定的訊息片段。
- 診斷檔案和架構檔案應包含故障排除流程：讀取 `error_code`、檢查 `repro_note`、檢查匯流排診斷，然後使用 `repro_gst_launch` 重新執行。
