---
title: "開發流程"
description: "對 SiMa.ai Neat 開發流程進行概觀式的介紹，從安裝到部署。"
sidebar_position: 4
---

# 開發流程

本頁面提供一份概略的地圖，說明您實際上如何每天使用 SiMa.ai Neat。當您準備好時，每個步驟都會連結到更深入的頁面。

## 迴圈

典型的 Neat 開發週期如下：

1. **安裝** — 在您的主機或裝置上取得 `sima-neat` 套件，並可選擇性地安裝 `pyneat` Python 介面。
2. **試用 Hello Neat** — 編譯一個最簡化的範例，以確認函式庫是否已正確連接。
3. **選擇一個已編譯的模型** — Neat 會使用一個已編譯的模型封存檔（`.tar.gz`，通常稱為 MPK）。您可以從 Model Zoo 中選擇一個，或者使用 Model Compiler 編譯您自己的模型。
4. **撰寫 `Model` / `Graph` / `Run`** — 載入模型，建構圖，並以最適合該任務的最小執行階段路徑來執行它。
5. **執行並檢查** — 提供輸入，獲取輸出，並使用 `GraphReport` / `MeasureReport` 來驗證行為。
6. **透過教學逐步學習** — 從單次推論開始，進而學習如何建立管線、多輸入模型、多串流圖，以及達到生產級別的錯誤處理能力。
7. **部署** — 將您的應用程式與目標裝置上已安裝的 Neat 函式庫連結。

## 選擇模型、圖或執行

從最小的、能夠解決問題的表面開始。當應用程式的運作機制變得更複雜時，您可以隨時升級。

| 如果您需要…… | 從以下開始 | 為什麼 | 下一站 |
| --- | --- | --- | --- |
| 執行一次已編譯的模型。 | `Model.run(...)` | 從成品到輸出張量之間的最短路徑。 | [執行您的第一個模型](/tutorials/run-your-first-model) |
| 在已連接的 Modalix PCIe 卡上執行模型。 | `pcie::Model` / `pyneatpcie.Model` | 在不建立原生卡片應用程式的情況下，於主機端進行協同處理。 | [PCIe 協同處理](/develop-apps/development-workflow/pcie-model) |
| 檢查範本合約或路線。 | `Model` | 規格、中繼資料和路由資訊會告訴您模型接受和輸出的內容。 | [模型](/develop-apps/development-workflow/model) |
| 將模型新增到應用程式流程中。 | `Graph` | 命名輸入和輸出，組成節點，並保持拓撲結構的明確性。 | [圖](/develop-apps/development-workflow/graph) |
| 在不同時間點重複使用同一個圖。 | `graph.build(...)` → `Run` | 提供推送/拉取、關閉/排空控制、測量和佇列策略功能。 | [執行圖](/develop-apps/development-workflow/pipeline) |
| 同時處理多個資料流，或追求最高的處理量。 | `RunOptions` + 測量 | 調整傳輸量需要佇列策略、封包丟棄計數器，以及針對每個資料流的證據。 | [調整吞吐量與佇列深度](/tutorials/tune-throughput-and-queues) |

如果這是您第一次進行實作練習，請從 [教學指南](/tutorials) 中的預先檢查清單開始，然後在新增圖形相關機制之前，先執行一個模型。

## 核心概念概覽

「開發流程」頁面會詳細說明上述各個環節。以下是重點摘要：

- [模型](/develop-apps/development-workflow/model) — 載入已編譯的模型套件，並將其公開為可執行的單元。
- [PCIe 協同處理](/develop-apps/development-workflow/pcie-model) — 在主機應用程式中執行已編譯的模型，並在連接的 Modalix PCIe 卡上執行。
- [生成式 AI 模型](/develop-apps/development-workflow/genai-model) — 直接執行生成式模型，或透過 HTTP 提供服務。
- [張量與樣本](/develop-apps/development-workflow/core_types)——這是各個階段之間傳遞的資料和中繼資料封包。
- [執行／推論](/develop-apps/development-workflow/overview) — 以同步方式執行 (`run`) 或以非同步方式執行 (`push` / `pull`)。
- [圖](/develop-apps/development-workflow/graph) — 將模型階段、節點、輸入和輸出組合成一個應用程式圖。
- [執行圖形](/develop-apps/development-workflow/pipeline) — 將圖嵌入到即時 `Run` 中，然後進行推送、拉取、排空、測量和調整，以最佳化吞吐量。
- [節點](/develop-apps/development-workflow/node)——圖的基本組成單元。

如果您只想先學習一頁內容，請從 [執行／推論概觀](/develop-apps/development-workflow/overview) 開始——它將 `Model`、`Graph` 和 `Run` 串聯起來。

## 接下來要去哪裡？

針對新使用者的逐步入門指南：

- [Neat SDK](/getting-started/dev-environment/) — 安裝 Neat SDK，配對一個 DevKit，並在搭載 `dk` 的硬體上執行。
- [建立](/develop-apps/contribute/build) — 使用 `build.sh`（貢獻者工作流程）從原始碼建立 Neat。
- [您好，Neat！](/develop-apps/hello-neat/minimal)——一個簡潔的 CMake 應用程式，它會連結到已安裝的函式庫。
- [教學指南](/tutorials) — 循序漸進的教學章節，內容從「第一個模型」到「生產管線」不等。

當您需要更深入的資訊時，請參考以下資料：

- [執行／推論](/develop-apps/development-workflow/overview)——將`Model`、`Graph`、`Run`、節點和輸入/輸出逐一分解的概念。
- [C++ 參考資料](/reference/cppapi) — 提供已安裝標頭檔的完整 API 介面。
- [Python 參考資料](/reference/pythonapi) — `pyneat` 繫結參考資料。

## 您寫的內容與 Neat 提供的內容之間的差異

Neat 擁有執行階段：模型載入、驗證、管線建構、排程、清理和診斷。您擁有應用程式程式碼，該程式碼將輸入連接到輸出，並對結果做出反應。界線是 `include/` 中的公開 API，它被視為**穩定**的——您可以升級 Neat，而無需重寫您的應用程式。

如果您只記得 Neat 中的三行程式碼，請記住以下內容：

```cpp
simaai::neat::Model      model(mpk_path);
simaai::neat::TensorList outputs = model.run(input_tensors, /*timeout_ms=*/2000);
simaai::neat::Mapping    view = outputs[0].map_read();  // inspect the output bytes
```

本檔案中所有其他內容——包括圖、執行句柄、非同步佇列和多串流應用程式——都是對那段核心三行故事的擴展，並且受到控制。
