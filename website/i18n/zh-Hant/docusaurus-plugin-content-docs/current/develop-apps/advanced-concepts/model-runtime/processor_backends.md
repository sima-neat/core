---
title: "處理器後端"
description: "A65、EV74 (CVU)、MLA、MLASHM、APU、TVM、M4——說明每個處理器所執行的功能，以及框架如何針對這些處理器進行最佳化。"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/processor_backends
---

# 處理器後端

Modalix SoC 具有多個處理器，每個處理器都適用於推論管線的不同部分。Neat 框架的規劃器會根據每個階段的功能，為每個階段選擇一個（或一條鏈）。本頁描述了每個後端及其在典型管線中出現的位置。

## A65 — 應用核心

執行 Linux 的標準 ARM A65 核心。框架的主要程序、所有應用程式碼以及大多數未加速的 GStreamer 元素都在這裡執行。

用於：

- `Graph` / `Run` 事件迴圈。
- 檔案/RTSP/網路 I/O。
- 診斷點和應用程式碼。
- 加速階段之間的輕量級連接。

## EV74 / CVU — 視覺運算單元

一種適合向量運算的 DSP 樣式處理器。框架在某些地方將其稱為 **EV74**，在其他地方則稱為 **CVU**（運算視覺單元）。用於 SIMD 結構但不足以證明需要 MLA 的核心：預處理（調整大小、色彩轉換、正規化）、鑲嵌/取消鑲嵌/量化/反量化邊界核心、融合預處理（通用預處理）和 BoxDecode 後處理。

EV74 的工作通過每個階段的 CVU 提交執行緒進行分配。核心二進位是模型封存檔（`lib/`）的一部分。

## MLA — 機器學習加速器

Modalix MLA。模型的編譯權重和圖位於此處。通過 MLA 提交執行緒進行分配，該執行緒接收來自 EV74（或直接來自量化階段）的鑲嵌輸入，並產生鑲嵌輸出，以便在下游取消鑲嵌。

MLA 工作有兩種形式：

- **MLA 推論** — 主要模型圖。
- **MLA 預處理/融合運算** — 當合約允許時，將預處理/後處理核心編譯到 MLA 中（[dtype 合約](/develop-apps/advanced-concepts/dtype_contract) 中的「MLA tess」欄）。

## MLASHM — MLA 共享記憶體

MLA 可以以最低延遲讀取的特殊記憶體區域。在可能的情況下，將分配給 MLA 輸入的緩衝區分配到 MLASHM 區段中。規劃器確保 EV74 端預處理直接寫入 MLASHM，以便 MLA 可以直接使用，而無需傳輸。

## APU — 音訊處理單元

音訊路徑以及某些從 SIMD-on-scalar 工作中受益的預處理階段使用。框架的音訊節點（重採樣、編解碼器）針對 APU。

## TVM — TVM 編譯的備用方案

對於 MLA 的編譯器無法生成的運算，框架可以回退到 TVM 編譯的 CPU 核心。在路徑計劃中顯示為 TVM 目標階段。比 MLA 執行速度慢，但當 MPK 合約描述 MLA 後端不支援的運算時，可以保證覆蓋。

## M4 — 協調核心

一個小型 Cortex-M4，用於低階協調 — A65 和加速器之間的 RPMsg、看門狗、硬體排序。應用程式碼永遠不會直接在 M4 上執行；框架通過作業系統層與其進行通信。

## 規劃器如何選擇

當建立圖時，路徑規劃器會遍歷每個階段並詢問：

1. **哪個處理器可以執行這個核心？** — MLA 推論會傳送到 MLA；預處理會傳送到 EV74；I/O 會傳送到 A65。
2. **達到目標的最低成本方式是什麼？** — 最小化傳輸次數（規劃器僅在無法避免的情況下插入 `ConversionKind::Transfer`）。
3. **相鄰的階段可以共享片段嗎？** — [記憶體模型](/develop-apps/advanced-concepts/memory_model) 決定了哪些是可行的；規劃器會使用它。

輸出是一個 `RouteGraph`，其中每個階段都包含一個目標處理器和一個片段策略。您可以透過 `Graph::describe()` 來檢查它。

## 更多閱讀

- 「處理器後端」— 請參閱設計深入探討的第 21 節和第 22 節。
- 「CVU 核心和圖目錄」— 請參閱 [CVU 核心](/develop-apps/advanced-concepts/cvu_kernels)。
- 「記憶體模型」— 請參閱 [記憶體模型](/develop-apps/advanced-concepts/memory_model)。
- [`Graph::describe()`](/reference/cppapi/classes/simaai-neat-graph) — 轉儲路由計畫。
