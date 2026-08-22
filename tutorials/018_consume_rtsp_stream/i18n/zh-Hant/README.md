# 018 處理即時 RTSP 串流

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 5-10 minutes |
| Model | None |
| Labels | rtsp, h264, h265, streaming, input-group, live-input |

## Concept

將即時 H.264 或 H.265 RTSP 串流附加到一個 `Graph`，並使用
`RtspDecodedInput` 區塊。該區塊連接到 RTSP，選擇匹配的
RTP 解包器和解析器，並將串流解碼為原始影格。

## Walkthrough

這是第一個章節，其中輸入來自*程式外部*。在之前的章節中，我們會產生測試圖像或從磁碟讀取檔案；在這裡，影格會從網路串流中持續不斷地傳入，而您會以盡可能快的速度處理它們。這個機制是一個可重複使用的 `Graph` 程式碼片段，即 `RtspDecodedInput`，它將整個 RTSP 到原始影格的前端整合到單一介面中。

本章刻意停留在「提取已解碼的影格」這個步驟。將它們輸入到 `Model` 中，會在其他地方進行說明（001 適用於單次模型執行，007 適用於將模型插入到管線中，015 適用於將模型嵌入到圖中）。到本章結束時，您將連接到一個 RTSP URL，並列印每個已解碼影格的張量形狀——這證明了串流正在進行。

這僅僅是一個*消費者*。要發布一個串流，請執行一個獨立的 RTSP 伺服器（例如 `mediamtx`），並將 `--url` 指向它。

### 設定 RTSP 使用者端 {#step-configure-rtsp}

`RtspDecodedInputOptions` 設定來源和解碼器。`url` 選擇
`rtsp://...` 來源。`codec` 選擇編碼格式。H.264 是預設值；
本教學還接受 `avc`、`h265` 和 `hevc`，其中 AVC 等於 H.264，而 HEVC 等於 H.265。

當您已經知道來源的影格速率時，請設定 `source_fps`。如果您省略它，此
教學將使用 OpenCV 開啟 RTSP 來源，讀取其報告的 FPS，並將
檢測到的值提供給 `RtspDecodedInput`。該群組本身不會探測
URL。只有探測路徑需要 OpenCV，並且 Python 版本會在
需要時導入它，因此提供 `--source-fps` 時，無需使用它；要進行探測，請使用
`pip install opencv-python` 安裝它。對於 H.265，Neat 將此值傳播到
已剖析的串流功能和解碼器設定中。它不會更改影格速率。
H.265 串流必須使用 HEVC Main 設定檔、8 位元、4:2:0 輸入。

設定 `tcp = true` 會透過 TCP 傳輸 RTP。TCP 可以保留順序並重新傳輸
遺失的片段，與 UDP 相比，這可以減少可見的遺失，但可能會增加
恢復遺失資料時的延遲。

### 組合圖 {#step-compose-graph}

建立一個 `Graph` 僅有兩個階段： `RtspDecodedInput` 片段（來源）和一個簡單的 `Output` 節點（拉取端點）。新增此片段是一個單一 `add(...)` —它會在內部擴展到連接/解封包/解碼元件，因此您的設定會維持在預期層級。由於輸入源自管線的*內部*，因此我們稱之為 `build(RunOptions{})` 過載，且無需採樣：沒有可用的框架。 `build()` 一開始就提供，因為串流會產生這些資料。

### 擷取已解碼的影格 {#step-pull-frames}

在程式執行時，迴圈會不斷重複。 `pull(...)` 搭配逾時設定。每次成功的提取都會產生一個 `Sample` 其張量是一個已解碼的影格。本教學使用預設的 NV12 輸出，以邏輯方式呈現。 `[H, W]` 帶有 Y 和 UV 平面中繼資料的張量。一個不會傳回任何內容（或傳回一個空的張量）的指令會顯示 `frame=N rtsp_timeout` 並終止迴圈——這通常表示 URL 錯誤或串流未正常運作。逾時設定可防止死掉的串流導致程式當機。

**C++：** 提取一個影格 `tensors_from_sample(*sample, true)`；迴圈會在讀取資料之前檢查清單是否為空。 `shape`。

**Python：** 從第一個條目中讀取一個影格。 `sample.tensors` 在列印其形狀之前。

## Run

本章會使用即時 RTSP 串流，因此您必須提供一個可存取的
`--url`。如果您沒有攝影機，請透過 RTSP 伺服器發布相容的影片，並指向 `--url` 在它上面執行。執行 **Python** 和 **C++（預先建置）** 指令，從 **Neat 安裝根目錄（包含 `share/` 以及
`lib/`）；從**原始碼**開始執行**建置**指令，指令從**儲存庫的根目錄**執行。

自動化的教學回歸測試會執行這兩種編解碼器。它會讀取第一個可用的 URL。 `SIMANEAT_TEST_RTSP_H264_URL` 或 `SIMANEAT_TEST_RTSP_H264_URLS`，以及
來自 `SIMANEAT_TEST_RTSP_H265_URL` 或 `SIMANEAT_TEST_RTSP_H265_URLS`。該測試會探查每個來源，並將其偵測到的 FPS 值傳送到 RTSP 群組。

**Python:**
```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --source-fps 30 --frames 5
```

針對 H.265：

```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --codec hevc --source-fps 30 --frames 5
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_018_consume_rtsp_stream
./build/tutorials-standalone/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

預期輸出（外觀取決於資料流的解析度和解碼器格式）：

```text
frame=0 shape=[720, 1280]
frame=1 shape=[720, 1280]
frame=2 shape=[720, 1280]
frame=3 shape=[720, 1280]
frame=4 shape=[720, 1280]
```

如果無法存取串流，您將會看到 `frame=0 rtsp_timeout`。若要將本章的 C++ 原始程式碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始程式碼檔案
- C++：`tutorials/018_consume_rtsp_stream/consume_rtsp_stream.cpp`
- Python：`tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py`
- Python OpenCV FPS 探測器：`tutorials/018_consume_rtsp_stream/probe_rtsp_fps.py`
