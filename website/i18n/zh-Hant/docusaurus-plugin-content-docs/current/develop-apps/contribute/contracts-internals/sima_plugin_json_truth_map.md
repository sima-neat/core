---
title: "SIMA 外掛程式 JSON 真值對應表"
description: "用於模型管線 SIMA 階段的已凍結 JSON 欄位使用地圖"
sidebar_position: 2
slug: /develop-apps/contribute/sima_plugin_json_truth_map
---

# SIMA 外掛程式 JSON 真值對應表（已凍結）

_上次更新：2026-02-17_

本檔案規定了模型管線 SIMA 階段中 JSON 欄位的用法，以便對其移除進行控制和測試。

## 1. 確定範圍和外掛程式矩陣

### 1.1 涵蓋範圍（模型管線）

- `simaaiprocesscvu` 用於：
  - 預處理階段 (`kernel=preproc`)
  - 量化/Tess 階段 (`kernel=quanttess`)
  - 用於 detess/dequant 的後處理階段包裝器（模型序列中的 `kernel=detessdequant`，後端元件仍然是 `simaaiprocesscvu`，位於 `src/nodes/sima/DetessDequant.cpp`）。
- `simaaiprocessmla`
- `simaaiboxdecode`（通用型解碼器）
- detess/dequant/tess 酬載階段，位於 `tmp/gst/*` 中：
  - `detessdequant`
  - `detessellate`
  - `quantize`
  - `slicedequant`

### 1.2 不屬於本範圍內

在此對照表的範圍內，不包含通用 CVU 應用程式外掛程式與自訂圖工具，包括但不限於：

- `overlay`、`genericrender`、`argmax`、`nms*`、`groupkeypoints`、`distancecalculation`、`cv_process`、`cvresize`、`fastbev*`、`PyGast-plugins/*`、已棄用的外掛程式，以及自訂應用程式/測試架構。

### 1.3 已處理的原始碼樹

靜態提取涵蓋了兩個請求的樹：

- `tmp/gst_plugins_source/gst/*`
- `tmp/gst/*`

檢查後視鏡：

- 兩棵樹都一樣：`genericboxdecode`、`detessdequant`、`detessellate`、`quantize`、`slicedequant`。
- 已分歧：`processcvu`、`processmla`

### 1.4 外掛程式矩陣

| 外掛程式／階段 | 目前所需的 JSON 鍵（目前的程式碼路徑） | 可推導出的金鑰 | 執行階段屬性（不應為靜態 JSON） | 僅適用於 MLA 格式的鍵 |
|---|---|---|---|---|
| `simaaiprocesscvu`（預處理／量化／後處理包裝器） | 先進行推論：如果有的話，連線會來自 `ConfigManager::getBuffers()`；JSON `input_buffers`/`output_memory_order` 僅作為備用方案。 | `input_width`/`input_height` 可以來自於圖 200/202 的標題/執行階段；布線陣列可以從 CM 中繼資料中合成。 | 在每個影格中重新協商執行階段的尺寸；框架建置不再重新編寫每個階段的 JSON 連接欄位。 | quant/tess 和後續路徑會間接使用 MLA 張量的形狀欄位。`input_depth`, `slice_*`) |
| `simaaiprocessmla` | `simaai__params`、`model_path`、`batch_size`；儘管 `outputs[*]` 仍然是首選，但當輸出形狀欄位可以推斷出片段大小時，它不再是強制要求。 | 輸出片段的大小可以從 `output_*`/`slice_*` + dtype 推導得出。 | `input_segment_name` 是一種可選的執行階段接線輔助工具；模型路徑可以從封裝中衍生。 | `outputs`，`data_type`，`output_*`，`slice_*`，量化參數。 |
| `simaaiboxdecode` | 後端設定載入器實際上需要：`buffers.output.size`、`memory.next_cpu`、`system.out_buf_queue`；目前，類別計數的解析取決於實作版本。 | 可以從 `input_depth`/ `slice_depth` + `num_in_tensor` + `decode_type` 推導出 `num_classes`（新的來源邏輯）。 | `buffers.input[*].name` 已從上游重新設定；閾值/前 N 個值通常是執行階段可調整的參數。 | `input_*`, `slice_*`, `data_type`, `num_in_tensor` |
| `detessdequant`（舊版獨立的 GST 元素） | `simaai__params`，加上解析器欄位：`orig_img_width`、`orig_img_height`、`frame_width`、`frame_height`、`num_in_tensor`、`next_cpu`、`no_of_outbuf`、`out_sz`、`input_*`、`slice_*`、`q_scale`、`q_zp`。 | 外掛中沒有；較高層級的形狀推論存在於 `StageConfig` 中。 | 上游緩衝區命名/CPU 路由是在封裝流程的執行階段進行的。 | `input_*`, `slice_*`, `q_scale`, `q_zp`, `num_in_tensor` |
| `detessellate` 負載 (`tmp/gst/detessellate`) | 接受 `de_tess.*` 或 root/static-contract 的等效值（`input_*`、`slice_*`/ `output_*`）；`buffers.input[0].offset` 為可選參數（預設值為 0）。 | 張量數量和維度可以從「manifest」階段的靜態欄位中合成。 | 輸入的名稱/路徑應該是在執行階段動態設定，而不是靜態設定。 | 形狀/切片合約 |
| `quantize` 酬載 (`tmp/gst/quantize`) | `quant_scale`，`zero_point`（JSON 備用方案） | 根據輸入緩衝區的大小推算出的輸入元素數量。 | 現在會先從上游取得 `q_scale`/`q_zp` 中繼資料，然後再改用 JSON 格式。 | 不適用（通用量化） |
| `slicedequant` 負載 (`tmp/gst/slicedequant`) | 讀取區段（`slice_dequant`/`simaai__params`/root），用於形狀回退；量化 JSON 僅用於回退。 | 從執行階段的中繼資料中取得量化資訊；維度可以從宣告的靜態合約中推導而出。 | 執行階段的中繼資料（`q_scale`/`q_zp`）是首選的傳輸方式。 | MLA 輸出形狀/數量合約 |

### 1.5 傳輸目前使用的資訊清單內容

- 管線上下文類型：`sima.model.manifest.v1`
- 符合 ABI 規範的外掛程式存取：位於 `include/gst/SimaPluginStaticManifestAbi.h` 中的 `manifest_accessor_v1`。
- 階段查找鍵：
  - `element_name`（預設值）
  - `logical_stage_id`（來自設定後的 `stage-id` 或 `stage_id` 管線屬性）
- 既有系統 `manifest_json` 字串會保留以供後續轉換使用。
- 框架節點/模型片段建構器現在會針對 SIMA 模型路徑輸出 `stage-id=<element-name>`。
  外掛程式設計得非常合理，因此即使應用了額外的名稱轉換，邏輯查詢的結果仍然是可預測的。

## 2. 必要的關鍵真理圖

### 2.1 靜態提取方法

從明確的存取點中提取：

- `json["..."]`
- `contains("...")`
- 語法分析器輔助函式（例如：`parser_get_int`、`parser_get_double_array`等）。

重要證據地點：

- `tmp/gst/processcvu/gstsimaaiprocesscvu.cpp:1667`
- `tmp/gst/processmla/gstsimaaiprocessmla.cpp:579`
- `tmp/gst/genericboxdecode/payload.cpp:61`
- `tmp/gst/detessdequant/gstsimaaidetessdequant.cpp:276`
- `tmp/gst/detessellate/detessellate.cpp:361`
- `tmp/gst/quantize/payload.cpp:124`
- `tmp/gst/slicedequant/payload.cpp:57`

執行階段的連線/推論證據：

- `src/nodes/sima/Preproc.cpp:245`
- `src/nodes/sima/DetessDequant.cpp:238`
- `src/nodes/sima/SimaBoxDecode.cpp:158`
- `src/pipeline/runtime/StageConfig.cpp:296`
- `src/pipeline/runtime/StageConfig.cpp:411`

### 2.2 使用的動態故障注入方法

對於已註冊的外掛程式（`simaaiprocesscvu`、`simaaiprocessmla`、`simaaiboxdecode`、`detessdequant`）：

- 基準值：`gst-launch-1.0 ... num-buffers=0`
- 一次變更一個欄位（移除欄位）
- 記錄啟動／執行階段失敗的行為和訊息。
- 請注意：動態結果反映了目前在此主機上已註冊的執行階段外掛程式。

對於未註冊的有效載荷階段（`detessellate`、`quantize`、`slicedequant`）：

- 在這個執行階段中，沒有可用的直接 GST 元素（`gst-inspect-1.0` 報告指出缺少）。
- 因此，動態金鑰移除功能在本階段僅限於靜態/來源分類。

### 2.3 分類地圖（已凍結）

### `simaaiprocesscvu`

- 強制要求：
  - 無論是 CM 緩衝區推論成功，還是使用 JSON 備用方案，都必須包含以下內容：
    - `input_buffers`
    - `output_memory_order`
    - 每個輸入的 `memories[*].segment_name`。
    - 每個輸入 `memories[*].graph_input_name`。
- 軟體/可設定為預設值：
  - `graph_name`
  - `input_width`、`input_height`（可選的 JSON 尺寸）
- 重複/衍生：
  - `input_buffers[*].name`（在執行階段連接）
  - 現在，從「manifest context」中取得的 `sink_pad_tensor_index_map`，是進行確定性多輸入映射時的首選方法。
  - 從 caps/runtime 取得預處理後的維度。
- 僅用於除錯：
  - `debug` 執行時不需要的樣式欄位

動態證據：

- 如果 CM 推論失敗且缺少 JSON 結構，則啟動程序會因匯流排錯誤而失敗。
- 如果 CM 推論成功，則可以省略 `input_buffers`/`output_memory_order`。
- 如果上下文表明存在模型管理的、具有多個輸入的階段，並且缺少或存在歧義的 `sink_pad_tensor_index_map`，則啟動會因匯流排錯誤而失敗。

### `simaaiprocessmla`

- 強制要求：
  - `simaai__params`
  - `model_path`
  - `batch_size`
  - `outputs[*].name`
  - `outputs[*].size`
  - `batch_sz_model`，當 `batch_size != 1` 時。
- 軟體/可設定為預設值：
  - `input_segment_name`
- 重複/衍生：
  - 可以在較高層級的模型中繼資料中推斷出輸出維度/類型。
- 僅用於除錯：
  - 並非關鍵

動態證據：

- 移除 `model_path` -> 發生未處理的 `nlohmann::json` 類型錯誤 (`ec=134`)
- `batch_size=2`，並移除 `batch_sz_model`，導致出現未處理的類型錯誤（`ec=134`）。
- 移除 `outputs` -> 啟動時可以通過 `num-buffers=0`，但執行階段`num-buffers=1`) 觸發 SIGSEGV 迴圈路徑

### `simaaiboxdecode`

- 強制要求（目前的執行階段行為）：
  - `buffers.output.size`
  - `memory.next_cpu`
  - `system.out_buf_queue`
- 軟體/可設定為預設值（取決於實作版本）：
  - 在較新的原始碼中，`num_classes` 的值可能會被推斷或回溯，但目前的執行階段可能僅會發出警告。
  - `decode_type` 在目前的執行階段可能會降級為類型不符警告。
- 重複/衍生：
  - `buffers.input[*].name` 執行階段連線。
  - 對於已知的解碼族群，可以從張量的形狀（`input_depth`/ `slice_depth`）推導出 `num_classes`。
- 僅用於除錯：
  - `system.debug`, `system.dump_data`

動態證據：

- 移除 `memory.next_cpu` -> 發生未處理的類型錯誤，程式中止 (`ec=134`)
- 移除 `system.out_buf_queue`，導致未處理的類型錯誤而中止 (`ec=134`)。
- 移除 `num_classes` -> 變更為非嚴重錯誤警告：`JSON type mismatch` (`ec=0`)

### `detessdequant`（舊版獨立的 GST 外掛程式）

- 強制要求：
  - `simaai__params` 物件
  - 解析器鍵：`orig_img_width`、`orig_img_height`、`frame_width`、`frame_height`、
    `num_in_tensor`, `next_cpu`, `no_of_outbuf`, `out_sz`,
    `input_height`, `input_width`, `input_depth`,
    `slice_height`, `slice_width`, `slice_depth`,
    `q_scale`, `q_zp`
- 軟體/可設定為預設值：
  - 目前程式碼中沒有。
- 重複/衍生：
  - 某些框架/原始尺寸欄位屬於中繼資料層級，應該可以從中推導出相關資訊。
- 僅用於除錯：
  - `debug`、`dump_data`、`inpath`、`ibufname`、`n_request` 等。

動態證據：

- 移除 `simaai__params` -> SIGSEGV 迴圈路徑（逾時）
- 移除 `num_in_tensor` -> SIGSEGV 迴圈路徑（逾時）

### `detessellate` 負載

- 強制要求：
  - 已解決輸入/切片張量欄位（來自 `de_tess.*` 或根目錄/靜態合約所產生的金鑰）。
- 軟體/可設定為預設值：
  - `buffers.input[0].offset`（預設值為 `0`）
  - `num_in_tensor`（如果省略，則從向量大小推導得出）。
- 重複/衍生：
  - 形狀向量可以從已定義階段的靜態張量中推導出來。
- 僅用於除錯：
  - 無

### `quantize` 有效負載

- 強制要求：
  - `quant_scale`
  - `zero_point`
- 軟體/可設定為預設值：
  - 目前程式碼中沒有。
- 重複/衍生：
  - 從輸入位元大小推導而來的張量元素數量。
- 僅用於除錯：
  - 無

### `slicedequant` 負載

- 強制要求：
  - 用於去量化的量化參數（`q_scale`、`q_zp`），可從執行階段的中繼資料或預設設定中取得。
  - 從分段/根/靜態合約合成中解析出的張量切片維度（`input_*`、`output_depth`/`slice_depth`）。
- 軟體/可設定為預設值：
  - 用於量化和形狀關鍵影格的純量與向量編碼
- 重複/衍生：
  - 在執行階段，優先使用量化後的元資料；JSON 僅作為備用方案。
- 僅用於除錯：
  - 不適用

## 3. 可控制的移除門（從這張凍結地圖中移除）

任何 JSON 欄位的移除都必須包含以下所有項目：

1. 更新此領域的真相地圖分類。
2. 新增/更新一個故障注入測試案例：
   - 啟動失敗時，必須明確顯示錯誤訊息（例如：匯流排錯誤），絕對不能導致程式崩潰。
3. 對於可推導或衍生欄位：
   - 先實作推論功能，
   - 保留「大寫/外掛選項」的備用方案。
   - 僅在仍然需要時，才將 JSON 作為最後的備用選項。
4. 請將僅包含 MLA 格式的 JSON 檔案保持簡潔：
   - 只有未解決的 MLA 中繼資料（形狀/大小/數量參數）還未處理。
5. 請務必啟用嚴格的持續整合閘道：
   - `unit_sima_plugin_manifest_strict_model_pipeline_test`
   - `unit_sima_plugin_manifest_strict_fallback_test`
   - 以及對應的 Vulcan CI 測試流程，請參閱 `.github/workflows/vulcan-ci.yml`。

## 4. 目前發現的風險

- 即使缺少必要的金鑰，某些路徑仍然會導致未處理的 `nlohmann::json` 例外狀況或 SIGSEGV 錯誤，而不是匯流排錯誤。
- `detessdequant` 舊版路徑容易因缺少解析器金鑰而導致程式崩潰。
- `slicedequant` 完全忽略 JSON 格式，並使用編譯後的常數。
- 在重新建構時，如果沒有將 `.so` 檔案從 `tmp/gst/*/build` 複製到 `deps/gst-plugins`，執行階段/原始碼的差異可能會再次出現。
