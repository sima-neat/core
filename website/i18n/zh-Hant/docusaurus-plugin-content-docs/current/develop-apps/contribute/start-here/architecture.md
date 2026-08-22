---
title: "架構"
description: "儲存庫架構與設計"
sidebar_position: 1
slug: /develop-apps/contribute/architecture
---

# 儲存庫架構與設計

本頁面是為貢獻者提供的，他們需要了解函式庫的結構、責任歸屬，以及如何在不破壞其模組和執行階段合約的情況下擴展框架。

---

## 框架與環境

「Neat」一詞用於描述兩個相關但獨立的概念：

- **Neat Library：** 本儲存庫中的 C++/Python 函式庫和執行階段。它載入模型套件、組成管線、驗證合約、在 Modalix 硬體上執行，並公開 API。
- **Neat SDK / 環境：** 圍繞框架的容器化開發工作流程，包括 DevKit Sync、共享工作區和代理工具。

在修改此儲存庫時，請針對支援人和代理的框架屬性進行最佳化：明確的 API、確定性行為、結構化的診斷、嚴格的驗證和穩定的公共合約。

---

## 本函式庫的用途

### 主要使用者
希望：
- 從可重複使用的建置模塊中組裝管線（無需編寫原始 GStreamer 程式碼）
- 儘早驗證管線（適合 CI），並快速了解失敗原因
- 透過 `appsink` 在 C++ 中執行管線並消費幀
- 可選擇透過 RTSP 提供管線（透過 `gst-rtsp-server`）
- 透過適合張量的輸出提供機器學習程式碼，而無需編寫 GStreamer 程式碼

## 套件所有權

選定的核心成品是 Neat、LLiMa 和一起安裝的內部 Debian 套件的權威來源。核心使用並轉發該成品，而無需選擇或重寫相依性版本。成品之外的套件仍由平台擁有；不相容的平台必須進行更新，而不是由核心或 LLiMa 進行修復。

### 常見工作流程
- **解碼/導入：** 檔案或 RTSP -> 拆分/多工/解析 -> 解碼 -> 轉換/caps -> appsink -> C++ 消費者
- **驗證：** 建置 + 解析 + 預滾動（暫停）以儘早捕獲協商問題
- **提供 RTSP：** 使用 `appsrc` 將合成幀推送到 RTSP 伺服器管線
- **影像/影片張量適配器：** 影像/影片/RTSP -> 解碼 -> 轉換/縮放 -> `add_output_tensor(...)` -> `Run::pull_tensors()`
- **教學：** 從 [教學指南](/tutorials) 開始，以獲得可執行的、有序的學習路徑

### 標準生產管線（權威來源）
此儲存庫的標準「生產路徑」為：**輸入 -> 預處理 -> MLA -> 後處理**。權威來源位於：`tests/e2e_pipelines/obj_detection/sync_yolov8_test.cpp`。

當此測試發生更改時，請更新 README + 架構，以保持檔案的一致性。

### 心理模型（業務邏輯 <-> 管線黏合劑）
您的應用程式保留業務邏輯；框架擁有管線黏合劑。

```text
Business logic
    |
    v
Nodes/Graph fragments  ->  GStreamer fragments  ->  caps negotiation  ->  runtime (Run)
    |                                                           |
    +-----------------------------------------------------------+
                                Sample / Tensor
```

## 核心概念

此框架刻意圍繞少數核心概念設計。大多數使用者程式碼只會與 `Model`、`Graph`、`Run`、`Tensor` 和 `Sample` 互動；較底層的貢獻者也會使用 `Node`、可重複使用的 Graph 片段、MPK 合約解析及圖的內部結構。

| 概念 | 角色 |
|---|---|
| 模型封存檔 | 一個封裝的 `.tar.gz` 成品，其中包含 MPK 推論合約、外掛程式私有設定、模型二進位檔和核心成品。 |
| `Model` | 用於載入 `.tar.gz` 模型封存檔的公開載入器。它會解析 MPK 合約、執行路徑規劃、公開模型階段，並提供簡單的 `run(...)` / `Graph` 組成進入點。 |
| `Tensor` | 帶有 dtype、shape、layout、儲存、裝置和語義中繼資料的類型化數值有效載荷。 |
| `Sample` | 圍繞張量、張量列表或捆綁包的執行階段/媒體封套。在讀取欄位之前，請檢查 `Sample::kind`。 |
| `Node` | 一個原子管線階段，它會產生一個確定性的 GStreamer 片段和所擁有的元素名稱。 |
| 可重複使用的 Graph 片段 | 預先建立的 `Graph`，可展開為多個節點，例如已解碼的 RTSP 輸入或模型階段。 |
| `Graph` | 組裝和驗證邊界。節點、模型和可重複使用的 Graph 片段會組成經過協商且可建置的管線。 |
| `Run` | 由 `Graph::build(...)` 傳回的即時管線句柄；擁有推送/拉取/執行階段生命週期。 |
| 圖 | 使用建構器圖來在單個管線內進行 DAG 組成；使用執行階段圖來協調跨管線的階段/執行。 |

從左到右閱讀它們之間的關係：

```text
model archive on disk -> Model -> Graph fragments/Nodes -> Graph -> Run
                                           |
                                           v
                                  Tensor/Sample flow
```

`Model` 是使用者入門的起點，但它並非獨立的執行引擎。它會解析為模型圖的片段/節點，這些片段/節點可以新增到 `Graph` 中。`Graph` 是核心的組裝概念；`Run` 是建置完成後產生的即時物件。

---

## 對於貢獻者的設計原則

這些是框架背後的關鍵架構原則。在決定實作選項時，請使用它們。

- **確定性優先。** 維持元件名稱、產生的管線字串、序列化的管線資料、報告欄位和測試的可重現性。診斷和代理迴圈取決於穩定的識別碼。
- **可除錯性是首要考量。** 錯誤應產生結構化資料，而不僅僅是字串：`GraphReport.error_code`、`repro_note`、匯流排訊息和可重現的後端管線。
- **不允許靜默回退。** 不要透過靜默地轉換格式、變更圖族群、回退到 CPU 或忽略外掛程式錯誤來隱藏模型輸入錯誤或硬體/執行階段失敗。
- **在執行前進行驗證。** 偏好在執行階段執行緒啟動或取得硬體資源之前進行結構、上限、形狀和合約驗證。
- **MPK 合約是模型的事實來源。** 核心路由、dtype、形狀、量化和階段決策必須來自 `mpk.json` / `*_mpk.json`。每個階段的 JSON 檔案都是外掛程式專用的。
- **邏輯等級和執行階段幾何形狀是分開的。** 核心會保留 MPK 撰寫的 `frame_shape` 作為邏輯輸出合約，並在需要時推導出明確的 MLA 幾何形狀。只有在已宣告的位元組範圍識別出唯一的解釋時，才會接受等級 2 的形狀作為 NC 或 HW。模糊或不一致的合約會在模型載入期間失敗。
- **公開 API 應保持穩定。** `include/*` 下的公開標頭會安裝並受到支援。優先採用增量變更和棄用路徑，避免破壞簽章的變更。
- **並行性必須受到限制且可觀察。** 串流執行緒的工作應盡可能輕量；探測端診斷需要原子操作或等效的執行緒安全處理；關閉時不應發生死鎖。

---

## 模型執行路徑

對於以模型為基礎的管線，高階路徑如下：

```text
input Sample/Tensor
  -> optional preprocessing / format normalization
  -> MLA inference stages selected from MPK contract
  -> optional postprocessing / box decode
  -> output Sample/Tensor
```

使用者可見的 `Model` 介面，其設計目的就是比 MLA 硬體介面更為友善。MLA 可能需要 INT8/BF16 和鑲嵌佈局，而使用者程式碼通常使用 FP32 和標準張量佈局。此框架透過以資訊檔為基礎的配接器階段來彌合此差距。

預處理和後處理是明確的框架階段/選項。如果發生格式不匹配、缺少必要的預處理中繼資料、MLA 派遣器不可用、模型封存檔或 MPK 契約無效，或 caps 協商失敗，則應以可操作的結構化錯誤形式呈現，而不是以隱藏的執行階段修正來處理。

---

## 儲存庫佈局

### 高階結構
- `include/` -- 公開標頭檔（支援的 API 介面）
- `src/` -- 實作
- `docs/` -- 檔案（本檔案）
- `examples/` -- 小型可執行範例
- `tests/` -- 單位/整合測試
- `python/` -- `pyneat` 套件來源、nanobind 繫結和 Python 測試
- `old_*` -- 舊版單體實作快照，用於參考/遷移

### 公開標頭樹 (`include/`)
公開標頭檔位於 `include/<module>/...` 下。
範例：`include/pipeline/Graph.h`、`include/model/Model.h`。

公開便利的進入點標頭檔：
- `include/neat.h`（總攬）
- `include/neat/runtime.h`
- `include/neat/models.h`
- `include/neat/nodes.h`
- `include/neat/node_groups.h`

有意沒有 `include/neat/graph.h` 公開總攬。需要較低層級圖基礎的執行階段/編譯器測試，會直接包含較窄的 `include/graph/...` 標頭檔。應用程式、範例和公開檔案應使用來自 `<neat.h>` 的單一公開 `simaai::neat::Graph`。

### 內部標頭檔和執行階段外掛程式路徑

位於 `include/` 下的公開標頭檔會被安裝並視為穩定的 API。
位於 `src/**/internal` 下的內部標頭檔不會被安裝；範例/教學應該僅使用公開 API。

執行階段環境注意事項：

- 如果使用 `deps/gst-plugins` 中的捆綁 GStreamer 外掛程式，請將 `GST_PLUGIN_PATH` 和/或 `GST_PLUGIN_PATH_1_0` 設定為包含該目錄。
- 如果使用 `cmake --install` 安裝，外掛程式會放置在 `${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/sima-neat/gst-plugins` 下。
  將該路徑新增到 `GST_PLUGIN_PATH` 和/或 `GST_PLUGIN_PATH_1_0`。
- 使用 `scripts/use_neatdecoder.sh` 為目前的 shell 設定外掛程式路徑。
- 如果要以系統範圍方式安裝外掛程式，請重新建置系統 GStreamer 快取。

---

## 規劃與穩定 (API 介面)

| 區域/API | 狀態 | 備註 |
|---|---|---|
| 核心管線 API (`Graph`、`Run`、`Tensor`、`Sample`) | 穩定 | 主要支援的 C++ 介面。 |
| 建置器內部元件 (`Node`、私有節點向量輔助工具、`GraphPrinter`) | 內部 | 僅使用 STL，支援 GStreamer 之前的組合。 |
| 模型 API (`Model`、可重複使用的 Graph 片段) | 穩定 | 標準模型檔案整合路徑。 |
| `include/policy/*` | 穩定 | 最少的已驗證策略合約和預設值 (`Decoder`、`Encoder`、`Memory`、`RTSP`)。 |
| `include/nodes/groups/ImageToH264RtspGroup.h` | 已規劃 | 空的佔位符群組。 |
| Python 綁定 (`python/`、`pyneat`) | Beta | 基於 Nanobind 的綁定和封裝位於儲存庫中；API 介面著重於 `Tensor`、`Graph/Run`、`Model`，以及核心節點/群組輔助工具。 |

---

## 模組和職責

### `builder/` -- 節點合約和私有線性組合支援（不使用 GStreamer）
**目的：** 定義如何從邏輯部分組裝管線。

主要類型：
- `Node` -- 每個管線建置區塊所實作的介面
- 私有節點向量輔助工具和 `GraphPrinter` -- 組合工具和診斷

**規則：** 建置器必須主要使用 STL。它不應擁有 GStreamer 執行階段物件。

---

### `nodes/` -- 已定義類型的管線建置區塊
**目的：** 提供隨即可用的節點實作，這些節點會產生確定性的 GStreamer 片段。

範例：
- `nodes/io/HttpSource`、`nodes/io/RTSPInput`、`nodes/io/StillImageInput`
- `nodes/common/*`（Caps、Queue、Output 等）
- `nodes/sima/*` (SiMa.ai 解碼/編碼/解析/酬載節點)
- `nodes/rtp/*`（depay/payload 輔助工具）
- `nodes/groups/*`（常見的多節點配方）

**合約：**
每個節點都必須產生：
- `backend_fragment(index)` -- 在特定索引處，此節點的 GStreamer 片段
- `element_names(index)` -- 此節點所擁有的確定性元素名稱（用於診斷和強制執行）

---

### `gst/` -- 簡潔的 GStreamer 工具
**目的：** 針對常見的 GStreamer 模式提供小型包裝函式/輔助工具。

範例：
- 初始化 (`GstInit`)
- 解析啟動字串 (`GstParseLaunch`)
- 匯流排排清/字串化 (`GstBusWatch`)
- caps 輔助工具/元素內省 (`GstHelpers`、`GstIntrospection`)
- 填充點/探測輔助工具 (`GstPadTap`)

**規則：** `gst/` 不應依賴 `pipeline/`（以避免依賴循環和「工具層」膨脹）。

---

### `pipeline/` -- 執行階段協調和公開 API
**目的：** 擁有執行階段生命週期：建置 -> 解析 -> 執行 -> 消耗 -> 拆解，並提供診斷資訊。

主要類型：
- `Graph` -- 使用者的主要進入點
- `Run` -- 具有推入/拉出 API 的執行管線處理器
- `Sample` -- 拉取操作所傳回的結構化輸出酬載
- `GraphReport` -- 針對失敗、停頓和重現的結構化診斷資訊
- `Errors` -- 異常 (`NeatError`) 嵌入報告

#### 錯誤語義合約

`GraphReport.error_code` 是標準的機器可讀錯誤碼欄位。框架中的
執行階段/建置/I/O 路徑將終端錯誤映射到穩定的程式碼分類：

- `misconfig.pipeline_shape`
- `misconfig.caps`
- `misconfig.input_shape`
- `misconfig.input_capacity`
- `misconfig.media_caps`
- `misconfig.tensor_dtype_missing`
- `misconfig.option_out_of_range`
- `build.parse_launch`
- `build.pipeline_syntax`
- `build.plugin_missing`
- `build.property_invalid`
- `runtime.pull`
- `runtime.element_failed`
- `runtime.output_timeout`
- `io.parse`
- `io.open`
- `io.file_not_found`
- `io.permission_denied`
- `io.rtsp_connection_failed`
- `io.camera_not_found`
- `codec.*`、`resource.*`、`infra.*` 和 `internal.*`

GStreamer 錯誤會通過一個內部解析器、分類器和渲染器。
分類器優先使用帶版本號的 Neat 診斷 ID，然後使用原生
GStreamer 域/程式碼和元件工廠，然後縮小舊版外掛程式的相容性映射。
未知的錯誤會使用 `runtime.element_failed`；除非實際協商失敗，否則不會
將其報告為 `misconfig.media_caps`。當管線發布多個錯誤時，會渲染最特定的根本原因，並且每個錯誤都會
保留在匯流排日誌中。

`GraphReport.repro_note` 是面向使用者的摘要。生產渲染包含
以純文字形式呈現的原因、相關的觀察/預期值、具體的
使用者操作以及穩定的診斷 ID。原始外掛程式字串、來源位置和
GStreamer 域/程式碼僅用於除錯。在 `NeatError` 建立時，會添加一次括號中的公共程式碼。
`GraphReport.bus` 是外掛程式/執行階段錯誤詳細資訊的真實來源。
對於建置（輸入）流程，`GraphReport.build_adaptation` 會記錄已解析的形狀原則/功能、種子/最大限制的來源、位元組保護的來源以及應用/跳過的調整動作。
對於非拋出型執行階段拉取，`PullError.code` 使用相同的分類法。
輸入串流工作程式失敗會保留類型化的錯誤程式碼並報告到
工作程式執行緒邊界，因此 `Run::pull()` 和 Python 例外翻譯器
會顯示相同的 `NeatError`。

支援優先順序是：
1. 依 `error_code` 分組
2. 閱讀 `repro_note`
3. 檢查第一個終端 `bus` 錯誤
4. 使用 `repro_gst_launch` 重新執行

#### 內部管線診斷
在 `src/pipeline/internal/`（僅限內部使用）下：
- `Diagnostics.h` -- 執行階段使用的共享診斷類型：
  - `DiagCtx`（匯流排日誌 + 節點報告 + 邊界/元件計數器）
  - `BoundaryFlowCounters`（從串流執行緒更新的原子計數器）
  - `ElementTimingCounters`（原子每元件計算時間）
  - `ElementFlowCounters`（原子每元件流量統計資訊）
- `GstDiagnosticsUtil.h` -- 用於格式化和收集 GStreamer 診斷資訊的輔助程式

#### SIMA 靜態資訊清單上下文合約
對於模型管線，框架會建置靜態階段/張量合約資料，並將其注入為
管線層級的 `GstContext`：

- 環境類型：`sima.model.manifest.v1`
- 環境欄位：
  - `manifest_version`
  - `manifest_json`（舊版相容性負載）
  - `manifest_accessor_v1`（ABI 安全的存取器表格指標）
  - 可選的 `session_id`、`model_id`
- 模型資訊的所有權/生命週期與管線的生命週期相關聯；外掛程式借用指標並複製它們需要的內容。
- 儲存庫邊界：此儲存庫不得新增建置時對外掛程式/分派器儲存庫的依賴。整合僅透過介面進行（執行階段 `GstContext`、屬性、caps/meta，以及 C-ABI 合約）。

已遷移欄位的解析器優先順序是確定性的：

1. 從合約/執行階段訊號推斷（形狀/meta/caps）
2. 環境/預設/屬性路徑
3. 嚴格的匯流排錯誤（永遠不會中止/SIGSEGV）

`StageTransformRuleRegistry`（內部）是單一的映射表格，它告訴解析器哪些非 MLA 階段繼承來自 MLA 輸入與 MLA 輸出的張量合約，以及何時傳播輸出量化。這使得預/後推導過程明確且可測試。

對於使用聚合器範本的已遷移 SIMA 外掛程式，執行階段設定現在遵循環境/屬性驅動的解析：
1. 階段靜態欄位來自模型資訊環境
2. 執行階段旋鈕來自屬性/環境預設值
3. 未解析的必要欄位會明確失敗（框架中沒有階段 JSON 備援）

對於 `simaaiprocesscvu`，CM 衍生的接線首先進行推斷，並且使用環境 `sink_pad_tensor_index_map`
進行確定性的多輸入映射；舊版輸入緩衝區名稱僅作為備援。

如果提供 `logical_stage_id`，則會從 `stage-id`/`stage_id` 管線屬性中解析，否則它會回退到元素名稱。
SIMA 模型路徑片段建構器預設會在 `simaaiprocesscvu`、`simaaiprocessmla` 和 `simaaiboxdecode` 元素上設定 `stage-id`。

##### YOLO26 BoxDecode 類別計數合約

對於模型管理的 YOLO26 偵測、姿勢和分割路徑，MPK 類別標頭深度是權威的類別計數。`Model::Options::num_classes = 0` 選擇該推斷值。一個正值必須與其匹配；如果出現矛盾，則會在合約建構期間失敗，並報告已設定的值、MPK 衍生的值和解碼類型。這可防止使用無效的類別計數來解釋分組的原始標頭佈局。SSD 和預先 YOLO26 非姿勢 YOLO 系列保留其現有的明確覆寫行為，而姿勢和 SuperPoint 解碼器保留其系列特定的規則。

##### SuperPoint BoxDecode 合約

SuperPoint 使用與其他模型管理的 BoxDecode 系列相同的 MPK 到靜態模型資訊邊界，並具有以下其他不變性：

- MPK 記錄擁有檢測器-logits 和描述符-grid 張量識別碼、儲存表示法、dtype/shape 資訊、數值設定檔來源，以及可選的明確 NMS 和邊界控制。核心永遠不會從張量值中識別這些角色。
- 核心將精確地將一個張量繫結到每個角色，驗證設定檔指紋和支援的表示法，應用明確的 `Model::Options::superpoint` 覆寫，並且僅解決省略的設定檔預設值。更改設定檔會重新計算其衍生的預設值，同時保留 MPK 或 API 明確設定的控制項。
- 版本化的靜態資訊清單 ABI 攜帶已解決的合約到 `simaaiboxdecode`。外掛程式僅在設定期間借用資訊清單指標，並且必須複製執行階段所需的任何狀態；核心在管線的整個生命週期內保留資訊清單的所有權。
- 實際輸出使用 `FEATURE_POINTS_V1` 線格式和特徵語義中繼資料。`FEATURE_POINTS_LEGACY_A65_V0` 僅在明確選擇以用於相容性時才可用；使用者不應從緩衝區大小推斷任何一種格式。

---

### `contracts/` -- 驗證規則
**目的：** 編碼「有效管線的樣子」，而不僅僅是「gst_parse_launch 成功」。

範例：
- 驗證器介面和註冊表
- 結構化的 `ValidationReport`

此層可用於 CI，並在執行階段之前捕獲問題。

---

### `policy/` -- 可供使用者調整的行為
**目的：** 將可調整的參數（預設值、記憶體限制、編碼器/解碼器/RTSP 策略選項）集中化。

目標是使「旋鈕」明確且易於發現，而不是隱藏在分散的程式碼中。

---

### 模型封存檔整合
**目的：** 透過 `Model` 載入 `.tar.gz` 模型封存檔，並將解析後的 MPK 推論合約轉換為可路由的 Graph 片段。

安全的封存檔載入器是內部實作細節；應用程式程式碼應建立 `Model` 並組合 `model.graph()` 或特定階段的片段。

常見用法：

```cpp
simaai::neat::Model model("resnet_50.tar.gz");
simaai::neat::Graph graph;
graph.add(model.graph());
```

---

### 模型階段片段
**目的：** 組成 `Model` 公開的預處理、推論、後處理或完整流程，而不公開內部封存檔載入器。

主要 API：
- `Model::preprocess()`
- `Model::inference()`
- `Model::postprocess()`
- `Model::graph()`

這用於混合流程，其中預處理僅執行一次，而 MLA/BoxDecode 則在單獨的圖或執行緒中執行。

---

### 工作執行位置（CPU / CVU / MLA）
處理器路由由 MPK 合約（模型封存檔中定義的 CVU/MLA 階段）以及可選的執行階段覆寫決定：

* `Model::Options` 控制預處理、後處理、命名和緩衝區選擇。
* `SIMA_MLA_NEXT_CPU` 可以在某些設定中覆寫 MLA 的下一個階段。
* 管線節點本身是宣告式的；實際執行發生在 GStreamer 外掛程式及其設定中。

實際影響：更多的緩衝區和明確的路由可以提高吞吐量，而封包類型不匹配或緩衝區大小不足會在協商期間快速失敗。

---

## 執行階段模型（執行方式）

### 初始化
所有執行階段進入點都呼叫單一的安全初始化例程：
- `gst_init_once()`（執行緒安全，`std::call_once`）

此外，執行階段路徑可能會驗證是否存在所需的外掛程式：
- `require_element("appsink", ...)` 等。

### 建立管線
透過新增 `Node` 物件和可重複使用的 Graph 片段來建立 `Graph`。對於 RTSP，請使用能辨識編解碼器的片段，以便在解碼前對來源進行解封包和解析：

```cpp
simaai::neat::nodes::groups::RtspDecodedInputOptions source;
source.url = "rtsp://example/live";
source.codec = simaai::neat::nodes::groups::RtspCodec::H265;
source.source_fps = 30;

simaai::neat::Graph graph;
graph.add(simaai::neat::nodes::groups::RtspDecodedInput(source));
graph.add(simaai::neat::nodes::Output());
```

內部：

1. 圖強制每個邏輯組合節點使用一個節點物件。重複的 `connect()` 呼叫可以重複使用該索引節點進行扇出。
2. 一個組合的變更會以一個單元的形式提交，或者完全回滾。
3. 圖會要求每個節點提供 `backend_fragment(i)`，並使用 `!` 將片段串連起來。
4. 它可選擇性地在節點之間插入**邊界標記**：

   * `identity name=sima_b<i> silent=true`
5. 它會分析精確的 `name=` 繫結，使用 GStreamer 進行一次解析，並清點建構的物件樹。在下游設定之前，如果名稱重複或遺漏，則會失敗。
6. 它會建立一個 `DiagCtx`：

   * `node_reports`，用於可重現性
   * `boundaries` 作為 `BoundaryFlowCounters`（原子變數）

### 推送/拉取執行階段模型

`Run` 擁有輸入/輸出佇列和一個輸入執行緒：

* `push(...)` 將輸入排入佇列（根據 `RunOptions::overflow_policy` 進行阻塞或丟棄）
* `pull(...)` 從應用程式接收器中取出 `Sample` 輸出
* `try_push(...)` 是非阻塞的（如果佇列已滿，則傳回 false）

這支援完全非同步的管線（生產者/消費者分離），以及單次流程（`Graph::run(...)`）。

### 解碼器許可生命週期

在選擇單一管線或連接圖的執行階段之前，核心會掃描編譯後的執行計畫，以查找型別為 H.264/H.265 的 `SimaDecode` 節點。所有符合條件的解碼器都將作為一個群組許可，並且由此產生的保留將由頂層 `Run` 擁有，直到其管線工作者停止為止。這同樣適用於線性 `Graph::add(...)` 管線、普通連接的片段和融合的即時分支。

許可需要已知的解碼器寬度、高度和幀率。核心永遠不會發明一個幀率。不完整的合約或不可用的可選許可端點會產生警告，並且計畫保持不變；使用 `SIMA_DECODER_ADMISSION_REQUIRE=1`，任何一種情況都會在解碼器硬體啟動之前失敗。容量拒絕和格式錯誤的租賃回應始終會失敗。

### 即時扇入降低

應用程式使用普通 `Graph::connect(...)` 描述即時邊緣，並使用普通 `Graph::build(...)` 將其具體化。`GraphLinkOptions` 攜帶最新的流策略、流識別碼、保留的佇列深度欄位和可選的原始幀許可限制。最新的流策略始終為每個流保留一個待處理樣本。

執行圖編譯器（而不是應用程式）決定是否可以將即時多來源扇入融合到一個 GStreamer 管線中。符合條件的私有、無輸入的來源分支將與其按流多工器和消費者一起降低，以便解碼的設備緩衝區不會跨越應用程式接收器/應用程式來源邊界。不符合條件的最新流拓撲結構將保持分割狀態。嵌套的已融合的來源片段在可以遞迴地保留其分支之前，將始終不符合條件。

### 內部邊界計時

一個邏輯 `Graph` 可以降低到幾個 GStreamer 管線片段。核心在它們之間的每個內部邊界處注入一個 `appsrc`。

**一個注入的邊界會傳輸它所接收的時間軸，並且永遠不會自行產生時間戳。只有一個公開的、應用程式擁有的 `Input` 會產生時間戳。**

`appsrc` 會從其自身區段的執行時間中產生時間戳，因此產生時間戳的邊界會為扇出中的每個分支提供不同的時鐘。視訊 RTP 接著會停止與描述相同影格的模型輸出中繼資料同步，而且沒有任何應用程式可以更正它：降低操作會消耗應用程式宣告的 `Input` 節點，因此應用程式設定的 `InputOptions` 不會傳遞到注入的邊界。

在新增區段實體化路徑時：

* 使用 `injected_boundary_input_options(...)` 建立注入選項，這是此不變量的唯一位置。
* 保持 `is_live = true`。清除它會導致即時區段停止。
* 維持公開的 `InputOptions::do_timestamp` 預設值，以便推送的 `cv::Mat`（不帶有 PTS）仍然可以在進入時接收到時間戳。

邊界會以零拷貝方式轉發保留的 `GstBuffer`，因此已存在的時間戳會在跨越時仍然存在。拒絕產生時間戳永遠無法移除它。

### 輸入合約的專業化

某些複合管線節點具有多個安全的後端表示形式。圖編譯器會從靜態建立的 `OutputSpec` 中對這些節點進行專業化；它不會修改公開的圖，也不會從第一個執行階段樣本中推斷出永久拓撲。`Derived` 或 `Authoritative` 合約可以選擇最佳化的表示形式。`Hint`、未知的格式/記憶體或缺少後端功能會選擇保守的表示形式。

例如，原始 `VideoSender` 僅在系統或 SiMaAI 記憶體中存在穩定的 NV12 合約，並且當 `neatencoder` 宣告其唯讀 `input-layout-aware=true` 功能時，才會省略其 NV12 轉換。`OutputSpec` 目前不包含平面步幅和偏移量，因此沒有記憶體域會繞過該功能閘。如果缺少或功能為 false，則會將其視為不受支援，因此核心可以與較舊的內部元件包保持安全。

原始視訊幾何形狀和物理儲存佈局仍然是獨立的合約。`OutputSpec` 和 caps 描述可見的寬度和高度；核心不得將這些值四捨五入到編碼器區塊、DMA 步幅或表面高度對齊。佈局感知外掛程式會從 `GstVideoMeta` 或 `GstVideoInfo` 衍生物理平面偏移量和步幅，在物理合約不相容時重新封裝，並將編碼器/硬體授權留給編碼器服務。這保留了精確的解碼幾何形狀，同時將特定於裝置的對齊方式排除在公開的圖 API 之外。

### 解析與啟動

該函式庫主要使用：

* `gst_parse_launch(pipeline_string, &err)`

這提供了靈活性和可除錯性（您可以透過 `gst-launch-1.0` 重新執行完全相同的字串）。

### 執行

典型流程（`Graph::build()` / `Run`）：

1. 執行合約（例如，對於 `build()` + 拉取操作，執行「sink last」）。
2. 建立管線字串（+ 可選的邊界）。
3. 解析管線。
4. 可選地執行元件命名合約。
5. 附加可選的邊界探測器。
6. 將管線設定為 `PLAYING`。
7. 傳回一個 `Run` 控點，用於推/拉控制。

### 幀的生命週期（簡潔語言）
1. **建立：** 節點變成一個確定性的 gst-launch 字串。
2. **協商：** GStreamer 在元件之間協商 caps（格式、大小、記憶體）。
3. **執行：** 輸入被推送到管線中（或從來源拉取）。
4. **樣本：** Appsink 將一個 `Sample` / `Tensor` 傳回您的程式碼。
5. **錯誤：** 任何協商或執行階段失敗都會變成一個 `NeatError`，並附帶一個 `GraphReport`。

Caps 協商是自動的；失敗會在早期（驗證/預滾動）或在執行階段發生，並提供您可以重現的診斷資訊（`describe_backend()` + 報告）。

### 相機設定的所有權

`CameraInput` 會在其相機 caps 之後，並在任何即時佇列之前，立即放置 `neatcamerabridge`。 在協商期間，橋接器會使用標準池來回應上游的 `GST_QUERY_ALLOCATION`，並請求 `GstVideoMeta`。 該池會分配來自一個打包的 SiMaAI 設定的已驗證平面，並為每個平面匯出一個 DMA-BUF。 一個相容的 `libcamerasrc` 會將這些 DMA-BUF 匯入到 ISP 捕獲佇列中。 然後，橋接器會解包相同的打包設定，以進行下游處理。 嚴格模式會拒絕任何不滿足該合約的緩衝區；CPU 複製仍然是一種明確的相容性後備方案。

面向應用程式的捕獲深度與核心的私有 CSI 到 ISP RAW 傳輸環以及後續的 GStreamer 佇列都是獨立的。 可選的 `capture_buffer_count` 參數，用於 `CameraInputWithCaptureBuffers`，用於控制在 ISP 輸出、libcamera 和應用程式之間保留的緩衝區。 `queue_depth` 和 `leaky_queue` 分別控制下游延遲和幀丟棄策略。 可選的相容性複製池會按需增長，並且不受該佇列深度的限制，因此，漏洩佇列可以應用其丟棄策略，而無需首先使上游橋接器停頓。

### 拆解

拆解是故意防禦性的。
某些外掛程式堆疊可能會在狀態變更時掛起；執行階段更傾向於避免使主機程序/CI 陷入死鎖。

常見的模式是：

* 發送 EOS
* 設定 `GST_STATE_NULL`
* 取消引用物件
* 應用逾時保護（如果需要，則洩漏而不是掛起）

---

## SimaAI 並行

SimaAI 外掛程式支援每個程序中的多個管線。 如果您同時執行多個管線，請使用 `GraphOptions` 或 `Model` 名稱字尾/字首來使元件名稱唯一，以避免 GStreamer 名稱衝突。

---

## 限制與安全性

* **輸入格式必須與 caps 匹配**：`InputOptions` 和模型設定必須就格式/寬度/高度達成一致。
  如果格式不匹配，則會在協商期間或推送輸入時立即失敗。
* **基於功能的動態輸入**：僅當已建構的圖宣佈具有動態功能時，才允許在執行階段重新協商。`FullyDynamic` 圖可以重新協商原始影片的幾何形狀/格式/幀率/媒體 caps；`IngressDynamicCvuOnly` 允許幾何形狀的更改，並且僅當建構時的下游合約檢查證明輸出行為穩定時，才允許格式的更改。
* **在有效範圍內動態調整**：`max_*` 是硬性上限；如果未設定 `max_*`，則 `width/height/depth` 作為隱含的上限。
* **模型與圖的預設值**：這兩種流程現在都透過 `src/pipeline/internal/InputPolicy.*` 解析種子/最大值/位元組保護原則；`Model` 仍然應用其記錄的基於元資料的預設值（例如 1920x1080 的上限），而 `Graph` 則保持由節點選項驅動，除非已進行設定。
* **`caps_override` 具有權威性**：如果已設定，則會阻止重新協商，並且形狀的更改需要重新建構。

| 流程 | 種子預設值 | 最大值預設值 | 位元組保護預設值 |
|---|---|---|---|
| `Model` | 如果存在，則為預處理元資料，否則從使用者格式/選項推斷 | 明確的 `input_max_*`；否則為原則預設值（例如 `1920x1080`，基於格式的深度） | 明確的 `RunOptions.max_input_bytes`，否則為 `InputPolicy` 中的有界估計值或彈性預設值 |
| `Graph` | 輸入節點選項和/或種子輸入樣本 | 明確的 `max_*`；否則為隱含的種子 `width/height/depth`（如果已提供） | 明確的 `RunOptions.max_input_bytes`，否則為 `InputPolicy` 中的有界估計值或彈性預設值 |

* **SimaAI 並行處理**：多個管線可以在進程內運行；請保持元素名稱的唯一性。

---

## 每幀屬性傳播

來源將 `Sample::attributes` 作為嵌套結構附加到 `GstSimaMeta` 中。保留緩衝區的元素會自然地攜帶元資料；在邊界處分配或重新使用緩衝區的核心會深度複製屬性並清除過時的值。`neatdecoder` 在解碼之前快照相同的幀上下文，並透過守護程式提供的關聯 ID 進行還原，因此重新排序或丟棄的幀無法將屬性轉移到另一個輸出。已協商的解碼器/守護程式協定擁有該關聯合約；舊版解碼器協定仍然僅為 FIFO。

支援的使用者介面路徑和限制記錄在 [每幀屬性](../../advanced-concepts/data-model-contracts/frame_attributes.md) 中。

---

## 執行緒和所有權模型

### 執行緒

* **GStreamer 串流執行緒**：pad 探測、解碼、排程
* **使用者執行緒**：`appsink` 輪詢 + 定期匯流排清空
* **RTSP 伺服器執行緒**：用於 `gst-rtsp-server` 模式的 GLib 主迴圈

### 所有權規則（GStreamer 物件）

* GStreamer 物件會進行參考計數。
* 如果您將 `GstObject*` 儲存在超出其取得範圍的位置，則必須 `gst_object_ref()` 它。
* 完成後，請始終 `gst_object_unref()` 一次。

### 診斷執行緒安全性（重要）

Pad probe 在串流執行緒上執行，因此從 probe 取得的診斷資訊必須是無鎖的。

設計如下：

* `BoundaryFlowCounters` 儲存原子變數。
* pad probe 僅執行原子 `fetch_add()` / `store()` 操作。
* 報告使用 `BoundaryFlowCounters::snapshot()` 將原子變數轉換為 `BoundaryFlowStats`（普通整數）。

這可避免資料競用，同時保持 probe 的效能。

---

## 診斷與可觀察性

### `DiagCtx` 擷取：

* 管線字串（用於重現）
* 節點報告（每個節點產生了什麼）
* 匯流排訊息（在互斥鎖下）
* 邊界流量計數器（原子變數）
* 元素計時 + 流量計數器（原子變數）

### 邊界流量 probe

啟用時，執行階段會將 pad probe 附加到邊界 `identity` 元素。
它們追蹤：

* 緩衝區計數（輸入/輸出）
* 最後看到的 PTS（奈秒）
* 最後看到的牆面時鐘時間（單調遞增，微秒）

這用於產生「可能停頓」摘要：

* 「我們上次看到活動在時間 T 處進入/離開邊界 X」

### 元素計時 probe

啟用時（`SIMA_GST_ELEMENT_TIMINGS=1`），執行階段會將 sink+src pad probe 附加到每個元素的**所有 pad**（靜態、動態和請求），並記錄每個緩衝區的 `src_ts - sink_ts`。這產生每個元素的計算計時，而無需依賴外掛程式檢測。

對於替換緩衝區的元素，實作會回退到 `GstSimaMeta` 關聯（框架 ID/串流 ID），並記錄 `missed_in`/`missed_out` 計數器。

### 元素流量 probe

啟用時（`SIMA_GST_FLOW_DEBUG=1`），執行階段會將每個元素的 pad probe 附加到每個元素，以追蹤緩衝區/位元組計數和 caps 變更，從而為圖中每個外掛程式提供輸送量上下文。

### 匯流排記錄和錯誤

執行階段會將匯流排訊息排入 `DiagCtx`。
如果發生錯誤訊息（`GST_MESSAGE_ERROR`），它會拋出 `NeatError`，其中包含 `GraphReport` 和重現提示。

### DOT 轉儲

如果啟用，執行階段可以透過 `gst_debug_bin_to_dot_file_with_ts(...)` 將 DOT 圖輸出到已設定的目錄。

### 偵錯流程（生產環境）
1. **重現管線**：`Graph::describe_backend()` 或 `last_pipeline()`。
2. **擷取報告**：`MeasureReport::to_text()` 或 `NeatError::report()`。
3. **啟用目標 probe**：
   - `SIMA_GST_BOUNDARY_PROBES=1`，用於定位停頓
   - `SIMA_GST_ELEMENT_TIMINGS=1`，用於每個元素的計時
   - `SIMA_GST_FLOW_DEBUG=1`，用於每個元素的流量計數器
4. **產生 DOT 圖**：設定 `SIMA_GST_DOT_DIR` 並重現。
5. **加強驗證**：`SIMA_GST_ENFORCE_NAMES=1` 並驗證預先緩衝逾時。

---

## 輸出處理

`Run::pull()` 會產生一個 `Sample`，其中可能包含：

* 一個 `Tensor` 有效載體（`SampleKind::Tensor`）
* 一個包含多個輸出的集合（`SampleKind::Bundle`）

在您希望使用張量負載而非完整的 `Sample` 封包時，對於以機器學習為中心的流程，請使用 `Run::pull_tensors(...)`。

---

## 管線序列化（儲存/載入）

管線可以儲存並還原為 JSON 格式：

* `Graph::save(path)` 會寫入包含節點類型/標籤/片段/元素的版本化 JSON
* `Graph::load(path)` 會透過 `ConfiguredNode` 封裝來重新建立節點

目前的結構設計是簡潔且可重現的，並且可以演進到更豐富的節點設定。這也可用於未來綁定和工具的橋樑。

---

## UX 輔助工具

* `Graph::describe()` 使用 `GraphPrinter` 來呈現可供人類閱讀的節點清單
* `Graph::describe_backend()` 會傳回 gst-launch 字串，以便快速進行偵錯

---

## 元素命名與確定性

確定性的元素名稱是核心設計原則，因為它們可以實現：

* `gst_bin_get_by_name()` 用於接收器和關鍵元素
* 穩定的探測器附加
* 穩定的診斷和可重現性
* 可選的命名合約強制執行（「每個元素都屬於某個節點」）

**節點作者必須確保：**

* 片段包含穩定的 `name=` 欄位，以便在需要可檢索元素時使用
* `element_names()` 傳回片段建立的每個明確元素名稱
* 宣告和具名填充參考保持同步

名稱完整性是 `build()` 的一部分，並且不依賴於先前的 `validate()` 呼叫。名稱在一個已實體的管線區段中是唯一的，因為框架查找使用遞迴簡短名稱。單獨解析的連接區段可以使用相同的名稱。框架會拒絕衝突，而不是重新命名它們，因為名稱可以參與填充和路由表達式。

取決於輸入的連接區段可以在第一個輸入上實體化。因此，它們的建置失敗會在第一次 `push()` 或 `pull()` 上報告，並保留原始的 `GraphReport`。

---

## 階段命名和連接

框架現在將外掛程式 JSON 視為外掛程式擁有的資料，並且在管線建置期間不會重新編寫或驗證每個階段的 JSON 欄位。

連接的真相來源：

1. 來自節點片段的確定性 GStreamer 元素名稱。
2. SIMA 模型路徑元素上的 `stage-id`。
3. 用於靜態階段/張量合約查找的 `sima.model.manifest.v1` 上下文。

影響：

* 建置不再修改 `node_name` / `input_buffers[*].name` / `buffers.input[*].name`。
* 建置不再執行基於 JSON 的連接檢查。
* 名稱轉換僅適用於元素名稱。

對於模型管理的圖執行，階段解析由 `stage-id` + 清單上下文驅動。對於非模型圖執行，明確的外掛程式屬性是執行階段控制平面。

---

## 驗證與合約

驗證的目的是在執行階段之前及早發現問題：

* `validate()` 可以解析並預先滾動（PAUSED），以檢測協商停頓
* `contracts/` 提供用於「管線正確性」的結構化驗證器

強制性的最終啟動名稱檢查也會在標準建置路徑中執行。`ValidateOptions` 控制額外的驗證工作，而不是是否強制執行名稱完整性。

對於連接的圖，`validate()` 會編譯端點拓撲，但不會為依賴輸入的區段建立啟動字串。每個區段在其實際輸入合約可用且區段實體化時，都會收到強制檢查。

預期的行為：

* 執行階段流程在發生致命錯誤時會拋出異常
* 驗證流程會傳回結構化的報告（適合 CI）

### SSD BoxDecode 合約解析

SSD 模型封包會在圖編譯期間，針對私有登錄中的精確術後頭部合約進行驗證。解析器會比較每個有序的邏輯定位和置信度 H/W/C 形狀；它不會對層級進行排序、使用模型名稱或接受通用的類似 SSD 的預設值。目前已註冊的模型為 SSD300-v1、SSD-Mobile-300-v1、SSD-Mobile-320-v1 和 SSDlite-Mobile-320-v1。

已解析的模型擁有分數啟動、置信度通道順序、背景類別、允許的類別選擇、所需的 300x300 或 320x320 模型框架以及 Stretch 預處理。核心將模型作為內部 `SsdRecipeId` 儲存；公開和外掛程式 ABI 解碼類型仍然是 `BoxDecodeType::Ssd` / `ssd`，這是已部署物件解碼器支援的標記。如果頭部幾何形狀不受支援或格式不正確、啟動衝突、類別選擇無效、非 Stretch 調整大小或模型框架錯誤，則在管線啟動之前會發生錯誤。模型發現僅在編譯時進行，並且不會增加每個框架的工作量。

---

## RTSP 伺服器模式

`run_rtsp()` 使用 `gst-rtsp-server`：

* 伺服器在專用的執行緒中執行，並具有 GLib 主要迴圈
* 在 `media-configure` 中，程式碼會依名稱找到 `appsrc` 並設定 caps/屬性
* 框架會定期推送（基於計時器），並帶有明確的時間戳記

根據工廠設定，每個客戶端都可能獲得自己的媒體實例。

---

## 環境/設定旋鈕

執行階段支援由環境驅動的偵錯旋鈕：

* `SIMA_GST_DOT_DIR` – 對於失敗/偵錯，寫入 DOT 圖
* `SIMA_GST_BOUNDARY_PROBES` – 啟用邊界流程計數器
* `SIMA_GST_STAGE_TIMINGS` – 啟用階段計時探測
* `SIMA_GST_ELEMENT_TIMINGS` – 啟用元素計時探測
* `SIMA_GST_FLOW_DEBUG` – 啟用每個元素的流程計數器
* `SIMA_GST_ENFORCE_NAMES` – 強制執行命名合約
* `SIMA_GST_RUN_INPUT_TIMEOUT_MS` – 執行/建置輸入路徑的輸入逾時
* `SIMA_GST_VALIDATE_TIMEOUT_MS` – 預滾動的驗證逾時
* `SIMA_GST_VALIDATE_INSERT_BOUNDARIES` – 在 validate() 期間插入邊界
* `SIMA_GST_RUN_INSERT_BOUNDARIES` – 在 build/run() 期間插入邊界
* `SIMA_GST_TEARDOWN_TIMEOUT_MS` – 等待 NULL 狀態（毫秒）
* `SIMA_GST_TEARDOWN_REAPER_MS` – reaper 重試間隔（毫秒）
* `SIMA_GST_TEARDOWN_ASYNC` – 跳過等待，延遲到 reaper

這些控制項有意地設計在公開 API 之外，因此您可以在 CI 或現場環境中啟用它們，而無需重新編譯。
在 `src/pipeline/internal/*` 中有額外的低階偵錯旗標（輸入資料流記錄、樣本轉儲、池偵錯）。除非您需要深入的診斷，否則請將這些內容排除在使用者檔案之外。

---

## PCIe 主機執行階段邊界

獨立打包的 PCIe 主機 API 具有兩個公開層級：

* `pcie::Model` 是一個與原始碼相容的便利 API，用於單個模型。
* `pcie::Runtime` 是一個卡片範圍的多模型協調器，旨在支援一個精簡的 OAAX C ABI 介面。

執行階段公開邏輯模型 ID、呼叫者提供的請求 ID、非阻塞佇列、從任何模型檢索、批次載入、獨立卸載和冪等清理。硬體佇列 ID 仍然是一個實作細節。目前的 Modalix 實作會將精確的一個已載入的模型指派給四個 PCIe 佇列中的每一個，並繼續透過虛擬乙太網路 SSH/SCP 控制路徑傳輸模型封存檔。

推論請求關聯使用在 PCIe/卡片往返過程中編碼在 `GstSimaHostMeta.frame-id` 中的帶符號的 32 位元 OAAX 請求 ID。位元模式是不透明的，並且必須在公開完成時保持不變。執行階段擁有模型到佇列的註冊表，並彙總每個佇列的結果；卡片端的 `pcie-pipeline-builder` 仍然是一個程序和一個模型圖，每個佇列一個。

卡片傳輸在舊版命名的 `GstSimaMeta.pcie-buffer-id` 欄位中攜帶一個單獨的私有請求令牌。它識別了確切的驅動程式請求和正在處理的信用；普通外掛程式可以不變地轉發它，但不得檢查或製造它。`stream-id` 仍然是輸出路由金鑰，而 `frame-id` 仍然是應用程式關聯金鑰。只有 `neatpciesink` 才會解析請求令牌。成功的作業會傳回一個 DATA 回應。解碼器丟棄、刷新、重新啟動或下游失敗會傳回一個關聯的 `NEAT_PCIE_FRAME_RETURN_ERROR`，它會釋放相同的主機信用，並使用可操作的錯誤終止受影響的主機管線，而不是讓它處於阻塞狀態。

標準化的 OAAX `runtime_*` C 符號是一個介面，位於此原生 API 之上。OAAX 所有權規則、狀態碼和最後錯誤儲存應位於該介面中，而不是 C++ API 中。

---

## 如何擴展程式庫

### 新增一個節點

1. 在 `include/nodes/<category>/<YourNode>.h` 中建立一個標頭檔
2. 在 `src/nodes/<category>/<YourNode>.cpp` 中實作
3. 確保：

   * `backend_fragment(i)` 有效且確定性
   * 所有重要的元素都已命名，並由 `element_names(i)` 傳回
4. 新增測試（理想情況下，新增以下測試之一）：

   * 解析/驗證測試
   * 使用簡單的來源/接收管線進行執行/建置測試

### 新增執行階段診斷

* 最好將欄位新增到 `DiagCtx` 和 `GraphReport`
* 如果更新來自串流執行緒，請使用 **原子**（或其他無鎖機制）
* 轉換為純快照類型以進行報告

---

## 依賴規則（不可協商）

* `builder/` 不應依賴 GStreamer 或 `pipeline/`。
* `gst/` 不應依賴 `pipeline/`。
* `nodes/` 不應依賴 `pipeline/`（節點是在建置時的描述，而不是執行階段的協調器）。
* `pipeline/` 是協調器，可以依賴 `gst/`、`builder/`、`nodes/`、`contracts/`、`policy/` 和模型內部元件。

這有助於保持架構的模組化，並防止循環依賴。

---

## 測試與範例

* `examples/` 顯示典型的端到端使用模式：

  * 解碼 RTSP
  * 執行模型封存檔
  * 執行 RTSP 伺服器
* `tests/` 驗證關鍵行為：

  * 檔案讀取路徑
  * 群組擴展等效性（輸入群組）
  * 張量輸出路徑 + 儲存/載入往返
  * `model_resnet50_multi_test` 驗證具有多個圖形/執行個體的模型準確性。

在新增功能時，請優先新增以下測試：

* 以確定性方式重現管線字串。
* 驗證 caps 協商假設。
* 確保失敗產生有用的 `GraphReport` 診斷資訊。

---

## 檔案一致性保護

保持檔案和程式碼一致：

* 如果您變更公用標頭 (`include/*`)，請更新 README + 架構。
* 如果您變更標準的生產管線測試 (`tests/e2e_pipelines/obj_detection/sync_yolov8_test.cpp`)，請同時更新檔案。
* 如果您新增新的環境參數，請將它們新增到「環境/設定參數」部分。

---

## 設計原則

1. **確定性優先**

   * 穩定的元件名稱、穩定的管線字串、穩定的報告

2. **可除錯性是首要考量**

   * 匯流排日誌、DOT 轉儲、邊界探測、清晰的重現步驟

3. **安全的並行處理**

   * 串流執行緒探測僅觸及原子操作（快照產生純文字報告）

4. **永遠不要讓程序停止回應**

   * 關閉操作是防禦性的；避免在損壞的外掛程式堆疊上無限期地阻塞

5. **保持公用 API 的穩定性**

   * 內部重構不應破壞使用者程式碼，除非有意進行版本控制
