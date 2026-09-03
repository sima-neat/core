---
title: "疑難排解"
description: "針對新錯誤，先從解決症狀開始 Neat 使用者最常點擊"
sidebar_position: 5
---

# 疑難排解

每個條目都以「**症狀 → 原因 → 解決方案**」的形式呈現。症狀標題是確切的錯誤訊息——請在本頁中搜尋（Ctrl-F），以找到您看到的訊息。每個條目都經過驗證，以確保與目前的原始碼一致，或在 DevKit 上進行重現。

如果您不確定從何開始，請跳至 [當您遇到問題時：診斷。](#when-youre-stuck-diagnostics)。

## 安裝與環境設定

### 1. `pyneat is not importable. Either Neat is not installed, or the venv is not activated.`

:::info 原因
`pyneat` 虛擬環境未啟用，或者您正在執行的環境中未安裝 wheel 套件。
:::

:::tip 修復
在執行任何 Python 程式碼之前，請先啟動 DevKit 環境：
```bash
source ~/pyneat/bin/activate
```
:::

### 2. GST 外掛程式載入失敗：`undefined symbol: _ZN16simaaidispatcher14DispatcherBase14submitPrepared...`

:::info 原因
Neat 執行階段共用函式庫不在動態載入器路徑中，因此 GStreamer 外掛程式無法在載入時解析執行階段符號。
:::

:::tip 修復
在啟動前，請將執行階段目錄置於 `LD_LIBRARY_PATH`。
```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/neat/runtime:$LD_LIBRARY_PATH
```
:::

### 3. 缺少模型封存檔 — `sima-cli modelzoo` 尚未執行。

:::info 原因
您程式碼中參考的 `.tar.gz` 模型封存檔（或 `SIMA_YOLO_TAR` / `SIMA_RESNET50_TAR` / `SIMA_MODEL_TAR`）在磁碟上不存在。
:::

:::tip 修復
請從 Model Zoo 下載。
```bash
sima-cli modelzoo get yolo_v8s     # or resnet_50, etc.
```
:::

## 建立

### 4. 無法找到 `find_package(SimaNeat CONFIG)` 封包。

:::info 原因
CMake 無法找到 `SimaNeatConfig.cmake`（已安裝於 `lib/cmake/SimaNeat/`）。在原生 DevKit 安裝中，它位於預設的系統路徑中；但在 SDK 交叉編譯中，sysroot 並未位於 `CMAKE_PREFIX_PATH` 中。
:::

:::tip 修復
匯出 `SYSROOT`，並讓您的 `CMakeLists` 將其新增到前置路徑中（[您好，這是 Neat 範本。](/develop-apps/hello-neat/minimal) 這樣做）：
```cmake
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu")
endif()
find_package(SimaNeat REQUIRED CONFIG)
```
:::

## 載入模型並進行設定。

### 5. `failed to read image: <path>`

:::info 原因
OpenCV（`cv2.imread` / `cv::imread`）傳回 null 值——表示檔案不存在、無法讀取，或不是可解碼的影像。
:::

:::tip 修復
在建構輸入張量之前，請驗證路徑，並確認該檔案是否為有效的 JPEG/PNG 格式。
:::

### 6. `reason=topk must be > 0`（來自 `boxdecode`）

:::info 原因
偵測模型的 `ModelOptions.top_k` 參數設定為 `0`；框檢測階段需要一個正數上限。
:::

:::tip 修復
設定一個正數 `top_k`（教學影片中使用 `100`）：
```python
opt.top_k = 100
```
（訊息來自 EV74 封包解碼外掛程式。）
:::

### 7. `preproc_upsample_not_supported`

:::info 原因
原始影像的尺寸小於模型的輸入解析度，因此預處理步驟必須進行**升頻採樣**，但較舊的 EV74 預處理韌體無法執行此操作（它僅支援降頻採樣）。
:::

:::tip 修復
請提供一張與模型輸入大小相等或更大的原始影像（例如，對於 YOLOv8，至少為 640×640），或者將 `neat-ev74-firmware` 更新為包含升頻採樣核心的建置版本。
（此訊息來自 EV74 預處理外掛程式/韌體。）
:::

### 8. 低 `score_threshold` → 後處理延遲峰值

:::info 原因
檢測閾值越低，通過閾值篩選的候選框數量就越多，而且非最大值抑制 (NMS) 的計算成本會隨著存留的框的數量大致呈**平方**級數增長。
:::

:::tip 修復
僅將閾值降低到足以捕捉到弱檢測值的程度，並使用 `top_k` 來限制最壞的情況。請參閱 [讀取偵測框](/tutorials/read-detection-boxes)。
:::

## 執行推論。

### 9. `misconfig.media_caps … Internal data stream error … reason not-negotiated (-4)`

:::info 原因
對於原始影像輸入，預處理階段並未啟用，或者輸入類型未聲明，因此無法在應用程式來源 (appsrc) 和第一階段之間進行協商。
:::

:::tip 修復
在 `ModelOptions` 中宣告影像輸入和預處理設定：
```python
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
```
:::

### 10. `No channel available (all candidate channel opens failed)`

:::info 原因
EV74 派遣器嘗試排程一個內核，但已載入的韌體並未實作該內核——通常是因為 `neat-runtime` 和 `neat-ev74-firmware` **不是相同的版本**（內部雜湊值不符），例如部分更新。
:::

:::tip 修復
一起安裝匹配的 `neat-*` 套件（哈希值相同）；確認在執行階段和韌體中顯示的哈希值是否相同。請參閱 [相容性 → 版本相符的集合](/getting-started/compatibility#the-version-matched-set-firmware--runtime)。
*（此訊息來自 EV74 派遣器。）*
:::

### 11. `frame=N rtsp_timeout`

:::info 原因
RTSP 拉取逾時——URL 錯誤或串流未傳輸影格。
:::

:::tip 修復
驗證 RTSP URL 是否可連線且正在積極地進行串流；檢查傳輸方式（TCP 或 UDP）。請參閱 [播放 RTSP 串流。](/tutorials/consume-rtsp-stream)。
:::

### 12. `CameraInput strict zero-copy requires external-buffer-mode`

:::info 原因
`CameraInputOptions::allow_cpu_fallback` 預設為 false，因此 Neat 需要端到端 SiMaAI/裝置零拷貝支援。或者 `libcamerasrc` 沒有宣告通用的 `external-buffer-mode` 屬性，或者已安裝的記憶體函式庫無法將其設定匯出為 DMA-BUFs。
:::

:::tip 修復
在安裝相容的相機和記憶體套件時，請務必維持嚴格的零複製。如果您必須在沒有 DMA-BUF 匯出的情況下使用相機堆疊，請明確選擇相容性橋接：

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::CameraInputOptions camera;
camera.allow_cpu_fallback = true;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera = pyneat.CameraInputOptions()
camera.allow_cpu_fallback = True
```

</CodeTab>
</CodeTabs>

自適應模式仍然將下游的 CVU/MLA 階段的資料傳輸到 SiMaAI 記憶體。只有在上游相機緩衝區尚未被 EV74 使用時，才會在相機橋接器處進行複製。
:::

### 13. `misconfig.media_caps … libcamerasrc … not-negotiated (-4)`

:::info 原因
您所要求的相機設定與相機堆疊能夠產生的模式不符，或者板載疊加層/驅動程式未正確地設定相機。
:::

:::tip 修復
驗證外部影像是否具有相同的格式、解析度和幀率。 Neat:

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

如果上述方法無效，請先檢查覆蓋層、纜線、感測器驅動程式或相機模式。使用 [Modalix DevKit MIPI 相機介面指南](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) 來確認 `.dtbo` 和 libcamera 的驗證路徑。如果驗證通過，請將其與您的 `CameraInputOptions` 進行比較。
:::

### 14. 相機拍攝的畫面呈現出綠色、紫色或過於強烈的色調。

:::info 原因
目前的畫面是以錯誤的像素格式或色彩轉換方式進行解讀。最常見的錯誤是將 `NV12` 格式的相機畫面視為 RGB/BGR 格式的位元組。如果相同的色調在 Neat 之前就已經出現，那麼問題很可能出在相機 ISP 的調整或 libcamera 管線。
:::

:::tip 修復
請確保相機鏡頭蓋和模型預處理格式保持一致：

- 請求推薦模型的路徑，格式為 `camera.format = "NV12"`。
- 設定 `preprocess.color_convert.input_format = PreprocessColorFormat::NV12`。
- 在正式發布版本中，請避免使用 CPU 進行 `videoconvert`/`videoscale`；
- 執行一個簡短的 `gst-launch-1.0 libcamerasrc ... ! videoconvert ! jpegenc` 僅進行簡單測試，以確認是否已存在著色效果。 Neat.
:::

### 15. MIPI 攝影機教學中的 `frame=N output_timeout`。

:::info 原因
在教學課程的逾時設定結束之前，沒有任何輸出樣本傳送到應用程式。在「相機到模型」的圖中，這可能表示相機沒有傳輸畫面、Caps 協商失敗、模型路徑仍在啟動中，或者下游階段（例如 BoxDecode）沒有產生輸出。
:::

:::tip 修復
首先驗證僅使用相機的路徑。然後重新執行教學，並設定較長的逾時時間，以及啟用後端輸出：

<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/model.tar.gz --frames 2 --decode none \
  --pull-timeout-ms 15000 --print-backend
</ShellCommand>

生產流程應包含 `libcamerasrc`、`neatcamerabridge`（當啟用回退機制時）、`neatprocesscvu`、`neatprocessmla` 和 `appsink`。對於 BoxDecode 路由，還請確認 `--decode` 標記和閾值與模型封存檔相符。
:::

### 16. 圖的處理速度慢，或者即時影格會遺失。

:::info 原因
這個圖的處理速度跟不上。常見的原因包括：拉取迴圈無法跟上速度、輸出樣本保留時間過長、在熱門路徑中進行逐幀記錄、佇列策略與來源不符，或即時串流沒有明確的丟棄/新鮮度策略。
:::

:::tip 修復
使用一個可重複使用的 `Run`，然後明確說明執行階段原則：

- 對於需要即時處理且資料時效性至關重要的即時輸入，請使用 `RunPreset::Realtime` / `pyneat.RunPreset.Realtime`。
- 對於需要處理每個輸入檔案的批次或檔案處理，請使用 `RunPreset::Reliable` / `pyneat.RunPreset.Reliable`。
- 當應用程式不應因佇列已滿而停止運作時，請使用 `try_push(...)`。
- 將 `on_input_drop` 設定為依據 `stream_id`、`frame_id`、`port_name` 以及原因來計算丟棄的次數。
- 持續拉取。如果輸出佇列已滿，可能會影響整個圖的效能。
- 在推送更多內容之前，先釋放或複製輸出，以防應用程式可能保留由執行階段支援的緩衝區。

對於多串流圖，請保留 `stream_id` 和 `frame_id`，並檢查每個串流的輸出計數。彙總的 FPS 可能會掩蓋掉一個資源不足的串流。請參閱 [執行圖形分析 → 調整輸送量，並確保結果真實可靠](/develop-apps/development-workflow/pipeline#tune-throughput-without-lying-to-yourself)。
:::

### 17. `unknown input/output name`、`no unambiguous default input` 或 `no unambiguous default output`

:::info 原因
這個圖有命名的端點，而且應用程式推送或拉取了錯誤的名稱，或者在具有多個可能端點的圖上使用了未命名的 `push(...)` / `pull(...)`。
:::

:::tip 修復
在推送或拉取程式碼之前，請檢查程式碼中的命名是否正確：

```python
run = graph.build()
print("inputs:", run.input_names())
print("outputs:", run.output_names())
```

然後使用確切的端點名稱：

```python
run.push("image", [tensor])
sample = run.pull("detections", timeout_ms=2000)
```

`Graph("name")` 是一個診斷標籤。它不會建立端點。端點來自 `nodes.input("name")` 和 `nodes.output("name")`。
:::

### 18. 在逾時之前，`pull(...)` 不會產生任何輸出。

:::info 原因
沒有任何範例在逾時前產生所需的輸出。該圖可能仍在執行中，輸出名稱可能不正確，輸入可能受到回壓，圖可能已關閉，或者可能發生執行階段錯誤。
:::

:::tip 修復
將逾時、關閉和錯誤事件分開處理。在 C++ 中，請使用結構化拉取超載：

```cpp
simaai::neat::Sample sample;
simaai::neat::PullError error;

switch (run.pull("detections", /*timeout_ms=*/1000, sample, &error)) {
case simaai::neat::PullStatus::Ok:
  break;
case simaai::neat::PullStatus::Timeout:
  // Keep waiting, push more input, or report timeout.
  break;
case simaai::neat::PullStatus::Closed:
  // End of stream.
  break;
case simaai::neat::PullStatus::Error:
  std::cerr << error.code << ": " << error.message << "\n";
  break;
}
```

此外，請檢查 `run.last_error()`、端點名稱、輸入資料類型/佈局/格式，以及您的應用程式是否持續從每個輸出分支提取資料。
:::

### 19. 舊程式碼片段在執行時，可能會因為以下原因而失敗：`push_timeout_ms`、`pull_or_throw`、根層級的 `input_max_*`，或 `boxdecode_original_*`。

:::info 原因
這段程式碼是針對較舊的選項介面或私人/內部路徑所撰寫。目前的應用程式程式碼應使用公開的 `ModelOptions`、`RunOptions` 和 `Run` API。
:::

:::tip 修復
請使用目前的公開名稱：

- 使用 `RunOptions.queue_depth`、`overflow_policy` 和 `try_push(...)` 來處理輸入壓力。
- 請改用 `pull(...)` 或結構化的 `PullStatus` 覆載，取代 `pull_or_throw`。
- 如果舊程式碼設定了根層級的 `input_max_*` 欄位，請將動態輸入限制移至 `ModelOptions.preprocess.input_max_width`、`input_max_height` 和 `input_max_depth` 下，並且僅在您確實需要設定邊界時才進行設定。
- 對於 BoxDecode 座標映射，請優先使用預處理後的元資料。請勿在新範例中設定已棄用的原始大小欄位。

如果您複製的頁面仍然顯示舊的拼字，請將其視為過時的檔案，並提交一份檔案錯誤報告，以便下一個讀者不會遇到同樣的問題。
:::

## 張量與 Python 的互通性

### 20. `… expects a TensorList; pass [tensor] instead of a single Tensor`

:::info 原因
一個未經處理的 `Tensor`（或 `Sample`）被傳遞給 `run` / `push` / `build`；API 要求提供明確的列表——這是故意的，並非錯誤。
:::

:::tip 修復
將其包裝：`model.run([tensor])`、`run.push([tensor])`、`graph.build([tensor])`。
:::

### 21. `image-mode Tensor input requires explicit image format metadata`

:::info 原因
一個接受圖像輸入的模型接收到一個沒有像素格式的張量，因此 Neat 無法解析位元組設定。
:::

:::tip 修復
使用明確的格式建立張量：`pyneat.Tensor.from_numpy(arr, image_format=pyneat.PixelFormat.RGB)`。
:::

### 22. `byte_format tensors cannot also specify image_format`

:::info 原因
一個張量是透過結合 `byte_format=`（不透明位元組）和 `image_format=`（像素）來建構的——這兩者是互斥的。
:::

:::tip 修復
通過其中一個，但不能同時通過兩個。
:::

## 來自另一個堆疊。

- **「我的 `.engine` / `.blob` / `.dlc` / `.hef` 呢？」**— Neat 會載入一個 `.tar.gz` 模型封存檔；這相當於編譯後的成品。
- 「我該如何將工作固定到 CUDA 串流／OpenCL 佇列中？」——您不需要這樣做；透過非同步方式將生產者和消費者分開。 `push`/`pull` 並調整 `RunOptions` 相反。
- **「為什麼吞吐量低於標示的 TOPS？」**——通常是由於主機負載、佇列資源不足、輸出反壓或捨棄策略所導致，而非加速器本身的問題。請參閱 [執行圖。](/develop-apps/development-workflow/pipeline)。

## 當您遇到問題時：診斷。

在猜測之前，請先查看以下內容。

**檢查管線/執行（Python 和 C++）：**
- `graph.validate()` → 一個 `GraphReport` — 在您建構之前，會驗證連線是否符合內建的合約。請檢查其 `error_code`。
- `graph.describe()` → 以文字形式呈現已解析的管線（節點名稱 + 封裝鏈）。
- `run.input_names()` / `run.output_names()` 指的是在執行階段推送/拉取時所接受的名稱。
- `run.start_measurement()` / `MeasureReport` → 包含計數器、延遲、輸入串流遙測資料、外掛程式/邊緣節點定時資訊，以及可選的電力資料。
- `run.json(...)` / `run.save_json(...)` 或 C++ `save_run_json(...)` → 在樣本移動完成後執行證據收集。
- `NeatError::report()`：當執行過程中發生錯誤時，會提供結構化的錯誤詳細資訊。


### 收集一份支援資料包。

如果您需要其他開發人員或 SiMa.ai 支援，請提供其他開發人員可以重現問題的證據。請包含：

- Neat 版本/建構資訊：Python `pyneat.build_info()` 或 C++ `sima_neat_version()`、`sima_neat_platform_version()` 和 `sima_neat_abi_version()`；
- 模型成品名稱、模型路徑，以及其製作方式；
- 能重現錯誤的最簡短、可執行的程式碼片段；
- 輸入形狀、資料類型、佈局、像素格式、酬載系列，以及圖是否為應用程式推送或由來源擁有。
- 來自 `run.input_names()` 和 `run.output_names()` 的端點名稱；
- `GraphReport`：來自 `graph.validate()` 或 `NeatError::report()` 的 JSON 資料；
- 在樣本完成執行流程後，執行從 `run.save_json(...)` 或 C++ `save_run_json(...)` 匯出 JSON 檔案。
- 當問題是延遲、吞吐量、封包遺失或功耗時，測量輸出結果。

對於多串流問題，也請納入每個串流的輸入計數、接受計數、輸出計數和丟棄計數。彙總的 FPS 可能會掩蓋其中一個串流出現問題的情況。

當您收集 `GraphReport` 時，請保留說明發生情況的欄位：

- `error_code`和`repro_note`；
- `pipeline_string`;
- `bus`;
- `repro_gst_launch`和`repro_env`；
- `dot_paths`和`caps_dump`；
- 當存在邊界探測時，顯示 `boundaries` / `BoundaryFlowStats`。
- 針對已植入的 `build(input, ...)` 失敗案例進行 `build_adaptation`。
- 執行 JSON 匯出，用於記錄執行後計數器和指標。

**開啟框架的偵錯輸出**，使用 `SIMA_DEBUG_PROFILE`，這是一個以逗號分隔的元件清單，用於追蹤。使用 `all` 以追蹤所有元件，或者縮小範圍：
```bash
export SIMA_DEBUG_PROFILE=all                 # everything
export SIMA_DEBUG_PROFILE=graph,gst,pipeline  # just these areas
```
已知的元件：`pipeline`、`graph`、`gst`、`appsink`、`inputstream`、`tensor`。預設情況下未啟用（沒有除錯輸出）。

**轉儲 GStreamer 圖**，以便視覺化檢查 caps 發生錯誤的位置：
```bash
export SIMA_GST_DOT_DIR=/tmp     # writes .dot graphs on build/failure; default: off
```

## 錯誤碼

`NeatError`（以及 `GraphReport::error_code`/ `PullError::code`）會回報一個 `domain.reason` 錯誤碼。框架明確定義了這些錯誤碼——啟用該錯誤碼，然後閱讀訊息以獲取詳細資訊。

| 程式碼 | 觸發條件 |
|---|---|
| 無法開啟檔案或裝置路徑 `io.open` | — 檔案遺失、權限不足，或缺少核心裝置（例如 `/dev/rpmsg*`）。|
| `io.parse` | JSON/設定檔解析錯誤——通常是 MPK 合約或各階段設定檔出現問題。|
| `misconfig.pipeline_shape` | 管線的幾何形狀或最終名稱的完整性有誤——例如，匯流點數量不正確、存在迴圈、缺少終端 `Output`，或元素名稱重複。|
| `misconfig.caps` | 在開始串流之前，一個大寫字母覆寫或相鄰節點合約未能通過框架驗證。|
| `misconfig.media_caps` | 在相鄰的媒體階段之間，執行階段 GStreamer 的協商失敗。|
| `misconfig.input_shape` | 輸入的張量違反了模型的規範（秩、空間維度、通道數量）。|
| `misconfig.runtime_abi_mismatch` | 框架/執行階段外掛程式 ABI 不匹配——通常是混合了 `pyneat` 和執行階段成品。|
| `build.plugin_missing` | 缺少必要的 GStreamer 元素或程式碼解碼器外掛程式。|
| `build.property_invalid` | 一個 GStreamer 元素屬性名稱或值無效。 |
| 自訂的 GStreamer 片段的語法無效。| `build.pipeline_syntax` |
| `build.parse_launch` | 由於無法更精確地分類，因此無法判斷 `gst_parse_launch` 失敗的原因。|
| `runtime.pull` | 某次拉取操作失敗，但沒有提供更具體的上游/根本原因代碼。|
| `infra.dispatcher_unavailable` | 無法取得 MLA/EV74/A65 訊息分派器——可能是韌體未載入、缺少授權或硬體故障。沒有 CPU 備援機制。|

這是一份簡短的疑難排解指南。請針對每個錯誤代碼以及 C++/Python 常數名稱，參考 [完整的錯誤碼目錄](/reference/error-codes)。
