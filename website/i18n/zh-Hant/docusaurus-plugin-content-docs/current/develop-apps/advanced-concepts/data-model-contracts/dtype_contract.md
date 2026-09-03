---
title: "dtype 規範"
description: "張量資料類型、量化、鑲嵌以及公開酬載合約如何相互配合？"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/dtype_contract
---

# 資料類型合約

一個 Neat 模型路徑有兩種合約：

- **公開合約**：您的應用程式透過 `Tensor`、`Sample`、`InputOptions`、模型規格和圖形端點看到。
- **模型路徑合約**：Neat 從編譯後的模型封存檔和選定的預處理/後處理路徑中解析。

不要假設每個公開邊界都是 FP32。有些邊界攜帶圖像、編碼媒體、封裝的檢測有效負載、INT8 張量、BF16 張量或應用程式定義的張量語義。首先檢查規格；規格就是合約。

在路徑內部，當編譯後的模型合約需要時，Neat 會插入量化、鑲嵌、類型轉換、取消鑲嵌、反量化和後處理階段。

## 四種 MLA 輸入案例

模型封存檔會告訴 Neat 關於第一個 MLA 階段的兩個重要資訊：

- MLA 輸入資料類型，通常為 **BF16** 或 **INT8**；
- MLA 側鑲嵌是否已包含在編譯後的內核中。

這產生了四個預處理圖形族：

| MLA 資料類型 | MLA 鑲嵌 | 預處理圖形族 | Neat 在 MLA 之前插入的內容 |
|---|---|---|---|
| BF16 | 是 | `Preproc` | 調整大小、色彩轉換、正規化。MLA 階段內部進行鑲嵌。 |
| BF16 | 否 | `Tess` | 調整大小、色彩轉換、正規化、鑲嵌。 |
| INT8 | 是 | `Quant` | 調整大小、色彩轉換、正規化、量化。MLA 階段內部進行鑲嵌。 |
| INT8 | 否 | `QuantTess` | 調整大小、色彩轉換、正規化、量化、鑲嵌。 |

檢查 [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) 以查看規劃器選擇了什麼。

## 鑲嵌的含義

鑲嵌將張量位元組排列成 MLA 輸入暫存區預期的圖塊幾何形狀。這是一種佈局轉換：相同的邏輯張量，但記憶體順序不同。

匹配的取消鑲嵌在 MLA 輸出之後發生，當路徑需要將自然張量佈局傳回到下一個階段或應用程式時。

## 邊界升級

Neat 可以在四種案例的資料類型決策之上添加更高層級的路徑階段：

- **通用預處理**：使用 `PreprocessOptions` 在推理之前應用調整大小、色彩、佈局、正規化、量化、鑲嵌或明確的轉換意圖。
- **框解碼**：為需要檢測後處理階段的模型解碼檢測頭。應用程式使用 `BoxDecodeType` 選擇族，例如 `YoloV8`，以及過濾欄位，例如 `score_threshold`、`nms_iou_threshold` 和 `top_k`。

這些升級會改變執行的內核以及應用程式接收到的輸出合約。例如，原始模型輸出張量和解碼後的檢測張量不是相同的公開合約。

## 這對應用程式程式碼有什麼影響

- 在分配輸入或解碼輸出之前，請檢查 `model.input_specs()` 和 `model.output_specs()`。
- 使用 `ModelOptions.preprocess` 來指定您提供的輸入類型：圖像輸入、張量輸入、調整大小、顏色、佈局、正規化、量化或鑲嵌意圖。
- 使用 `model.resolved_preprocess_plan()` / `model.preprocess_plan()` 來查看 Neat 根據您的選項以及模型封存檔所規劃的內容。
- 不要假設輸出 dtype、形狀或佈局。請閱讀輸出規格，並在需要時，閱讀傳回的張量元資料。
- 僅當輸出合約是匹配的打包格式時，才解碼框、姿勢或分割。
- 除非有公開的規格或張量明確地顯示它們，否則將 INT8/BF16/鑲嵌詳細資訊視為路由行為。

不要想太多。閱讀合約，然後移動位元組。

## 有目的地解碼輸出

使用與輸出合約匹配的解碼輔助函數。

| 輸出合約 | C++ | Python |
|---|---|---|
| 原始張量 | 直接使用傳回的 `Tensor` / `TensorList` | 直接使用傳回的張量，或使用 `to_numpy(...)` / `to_torch(...)` |
| 打包框 | `simaai::neat::decode_bbox(...)` | `pyneat.decode_bbox(...)` |
| 打包姿勢 | `simaai::neat::decode_pose(...)` | `pyneat.decode_pose(...)` |
| 打包分割 | `simaai::neat::decode_segmentation(...)` | `pyneat.decode_segmentation(...)` |

解碼後的框使用一個 float32 `[N, 6]` 張量，其中包含 `x1`、`y1`、`x2`、`y2`、`score` 和 `class_id` 這些欄位。姿勢和分割解碼器傳回框以及用於關鍵點或遮罩的特定任務張量。

## 保留坐標元資料

檢測坐標通常需要預處理元資料，才能將其從模型空間映射回源框架空間。在使用 letterbox、調整大小、ROI 列表、渲染或檢測解碼時，請通過圖保留元資料。

相關元資料可以包括目標大小、縮放大小、填充、顏色轉換、軸置換、正規化、量化、鑲嵌、ROI 窗口以及每個 ROI 的仿射變換。

如果解碼後的框位於錯誤的位置，請在責怪 NMS 之前檢查元資料傳播。請參閱 [資料格式](/develop-apps/advanced-concepts/data_formats#preprocess-metadata-and-roi-breadcrumbs) 和 [預處理感興趣區域 (ROI) 清單](/reference/preproc_roi)。

## 相關類型

- [`PreprocessOptions`](/reference/cppapi/structs/simaai-neat-preprocessoptions) — 應用程式預處理意圖。
- [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) — 規劃器編譯的內容。
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — 選擇的預處理族。
- [`Tensor`](/reference/{lsa}/structs/simaai-neat-tensor) — 公開的張量有效載荷和元資料。
- [`Sample`](/reference/{lsa}/structs/simaai-neat-sample) — 有效載荷加上執行階段元資料。

## 更多閱讀

- [張量與樣本](/develop-apps/development-workflow/core_types)
- [資料格式](/develop-apps/advanced-concepts/data_formats)
