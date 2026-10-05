# 029 透過 PCIe 執行 GenAI 模型

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | Qwen3-0.6B-Autoround-a16w4, LFM2-VL-450M-Autoround-a16w4, whisper-small-a16w8 |
| Labels | PCIe, GenAI, LLM, VLM, Whisper |

## Concept

從主機應用程式在 Modalix PCIe 卡上執行文字生成、圖片問答與語音轉錄。模型檔案、請求與結果均透過 PCIe 傳輸；應用程式使用公開的 GenAI API，而非 GStreamer 管線。

## Walkthrough

每個範例都會載入主機上已準備好的 LLiMa 模型目錄，執行具有上限的請求，然後關閉模型。請先分別執行各範例。它們使用相同的連線設定，但請求輸入不同。

### 載入 LLM {#step-load-llm}

在 C++ 中建立 `pcie::genai::GenAIModel`，或在 Python 中建立 `pyneatpcie.genai.GenAIModel`。建立時會將模型載入卡上，不需要 `build()` 或選擇佇列。設定 `connection.card_id` 以選擇卡。可使用模型的絕對路徑，或相對於目前工作目錄的路徑。

### 傳送文字請求 {#step-text-request}

設定 `prompt` 與 `max_new_tokens`，然後呼叫 `run()`。結果包含回答文字與生成指標。範例預設最多生成 128 個權杖；可使用 `--max-tokens` 更改上限。

### 串流回答 {#step-stream-answer}

向 LLM 範例傳入 `--stream`，即可改為逐一處理 `TokenSample`。每個文字片段到達時立即印出；最終樣本包含完整的生成指標。每個模型同一時間只能有一個進行中的請求。

### 關閉模型 {#step-close-model}

C++ 在推論後明確呼叫 `close()`，若例外中斷範例，則由解構函式進行清理。Python 使用內容管理器，在離開區塊時關閉模型，發生錯誤時亦然。連線逾時均有上限：啟動 15 分鐘，請求 5 分鐘。

### 準備 RGB 圖片 {#step-prepare-image}

VLM 範例在主機上使用 OpenCV 讀取圖片，將 BGR 轉換為 RGB，並產生形狀為 `[H, W, 3]` 的連續 UInt8 像素。C++ 將擁有像素資料的向量包裝為 PCIe 張量；Python 則提供 NumPy 陣列。請勿正規化像素，也不要改用卡上的本機圖片路徑。

### 詢問圖片內容 {#step-image-request}

載入 VLM，將 RGB 輸入與提示一起放入 `request.images`。模型會自行處理圖片前處理。傳回的文字就是問題的回答；PCIe 請求沒有圖片檔案欄位。

### 使用 Whisper 轉錄語音 {#step-audio-request}

載入 Whisper 目錄，並將 `audio_file` 設為主機音訊路徑。使用 `language="auto"` 偵測語言，或透過 `--language` 指定來源語言。預設請求會進行轉錄；`--translate` 則選擇翻譯為英文。讀取轉錄文字、語言與選用的無語音機率。請勿在 ASR 請求中加入聊天提示。

## Run

請依照[安裝 PCIe 主機](/getting-started/neat-library/pcie-host/)與[教學準備](/tutorials/before-you-run)操作。使用互相相容且提供 PCIe GenAI API 的主機與卡端套件。以下所有命令都在主機上，從解壓縮後的 PCIe extras 根目錄執行。

### 下載已準備好的模型

這些 SiMa.ai Model Zoo 儲存庫已包含編譯好的 LLiMa 成品：[Qwen3](https://huggingface.co/simaai/Qwen3-0.6B-Autoround-a16w4)、[LFM2-VL](https://huggingface.co/simaai/LFM2-VL-450M-Autoround-a16w4) 與 [Whisper](https://huggingface.co/simaai/whisper-small-a16w8)。只下載您打算執行的模型。請閱讀模型卡，並使用與已安裝平台相容的成品。

```bash
source ~/pyneatpcie/bin/activate
python -m pip install huggingface_hub opencv-python
mkdir -p models
hf download simaai/Qwen3-0.6B-Autoround-a16w4 \
  --local-dir models/Qwen3-0.6B-Autoround-a16w4
hf download simaai/LFM2-VL-450M-Autoround-a16w4 \
  --local-dir models/LFM2-VL-450M-Autoround-a16w4
hf download simaai/whisper-small-a16w8 \
  --local-dir models/whisper-small-a16w8
```

保留每個完整目錄，包括分詞器、設定與編譯好的成品。不需要解包或手動複製到卡上。原始 Hugging Face 檢查點並不是已準備好的 LLiMa 目錄；對於自訂模型，請先依照[使用 LLiMa 的 GenAI](/genai-llima/)操作，再執行本教學。

### 準備媒體

VLM 命令使用隨附的街景圖片。若要執行 Whisper，請在主機上錄製或提供一小段語音。安裝 FFmpeg 後，可將現有錄音轉換為 16 kHz 單聲道 WAV：

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le speech.wav
```

請勿使用靜音來判斷轉錄品質。範例可透過 `--image` 或 `--audio` 接受其他圖片或音訊路徑。

**Python:**

```bash
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_llm.py \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_vlm.py \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_whisper.py \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./lib/sima-pcie-host/tutorials/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./lib/sima-pcie-host/tutorials/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_029_run_llm
./build.sh --target tutorial_029_run_vlm
./build.sh --target tutorial_029_run_whisper
./build/tutorials-standalone/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./build/tutorials-standalone/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./build/tutorials-standalone/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

從原始碼建置 C++ 需要 PCIe 開發套件與 OpenCV 開發檔案。所有範例預設使用卡 0 與連線的預設 SSH 使用者 root。若安裝環境不同，請加入 `--card N`、`--user sima`、`--ssh-key PATH` 或 `--card-host ADDRESS`。請使用已設定的 SSH 帳號；範例不會設定存取權限。

您應該看到 LLM 回答、場景描述，或與錄音內容相符的轉錄文字。用詞取決於模型與輸入；僅成功執行並不代表已完成品質檢查。

## In Practice

獨立模型應使用個別控制代碼，並考量卡上的記憶體與運算資源。聊天時透過 `messages` 而非 `prompt` 提供歷史紀錄；獨立呼叫不會記住先前對話。若要取消串流，請呼叫 `cancel()` 並讀取至最終樣本，再重複使用模型。

API 概觀請參閱 [PCIe GenAI API](/develop-apps/development-workflow/genai-model/pcie-api)。獨立執行的 GenAI 教學使用不同的命名空間與本機模型執行方式；請勿在此替換成那些模型控制代碼。

## 原始碼檔案

- `run_llm.cpp` / `run_llm.py`
- `run_vlm.cpp` / `run_vlm.py`
- `run_whisper.cpp` / `run_whisper.py`
- `tutorial_args.h`（共用 C++ 命令列解析）
- `../assets/street-scene.png`
