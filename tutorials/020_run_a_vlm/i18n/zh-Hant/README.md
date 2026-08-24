# 020 使用直接 API 執行 VLM

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, vlm, image, cache, multimodal |

## Concept

針對相同的圖片提出重複的問題，而無需為每個請求重新編碼該圖片。

## Walkthrough

視覺語言模型可以接受文字加上圖片張量。對於一個問題，將圖片直接附加到 `GenerationRequest.images`。對於重複的問題，只需編碼一次圖片，並在後續請求中重複使用快取的圖片嵌入。

### 載入 VLM 和圖片 {#step-load-inputs}

從已部署的 LLiMa 模型目錄中載入一個 `VisionLanguageModel`，並從磁碟中解碼一個圖片。

**C++：** 使用 OpenCV 讀取圖片。Neat 將三通道 `cv::Mat` 輸入視為 BGR，並在內部將其轉換為 RGB。

**Python：** 使用 OpenCV 解碼，將 BGR 轉換為 RGB，並將 NumPy 陣列傳遞到請求中。

### 使用直接圖片提問 {#step-direct-image}

將圖片直接附加到第一個請求。這是最簡單的方法，通常足以用於一次性的視覺問題。

### 快取圖片嵌入 {#step-cache-image}

呼叫 `encode(...)`，以在模型中快取圖片嵌入。如果圖片已接受並快取，則該呼叫會傳回 `true`。

### 提出後續問題 {#step-follow-up}

在每個應重複使用快取圖片的請求中，設定 `use_cached_images = true`。您可以針對相同的快取圖片提出多個問題。未設定該旗標的請求會正常運作：僅文字請求不使用圖片，直接圖片請求使用其自身的 `images`，而另一個 `encode(...)` 呼叫會取代快取圖片。

### 將圖片附加到聊天訊息 {#step-message-image}

當您使用 `messages` 時，將圖片附加到需要它們的使用者訊息。這樣可以將圖片與其對應的確切文字放在一起。

## Run

在 Modalix DevKit 上，使用 LLiMa CLI 從 Hugging Face 下載 LFM2-VL 1.6B VLM：

```bash
llima pull LFM2-VL-1.6B-a16w4
```

在 Modalix 上使用 DevKit 本機模型目錄和本機圖片執行教學：

**Python:**
```bash
python3 share/sima-neat/tutorials/020_run_a_vlm/run_a_vlm.py \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_020_run_a_vlm
./build/tutorials-standalone/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

預期的輸出結果是，針對直接的圖片請求，產生一個答案；針對後續的請求，重複使用快取的圖片，產生多個答案；以及針對訊息層級的圖片請求，產生一個答案。

## In Practice

當使用者針對相同的畫面、產品圖片、圖表或檔案頁面提出多個問題時，請使用圖片快取。避免在每次請求都使用不同的圖片時進行快取，因為直接圖片的路徑更簡單，並且能讓提示狀態更清晰。

某些模型系列可能不支援快取重複使用。在這種情況下，請在每次請求中使用直接圖片。

當您建立對話時，並且只有一條訊息應該包含圖片時，請使用 `ChatMessage.images`。對於更簡單的單一提示格式，請使用頂層的 `GenerationRequest.images`。

## 原始程式碼檔案
- C++：`tutorials/020_run_a_vlm/run_a_vlm.cpp`
- Python：`tutorials/020_run_a_vlm/run_a_vlm.py`
