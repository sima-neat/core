# 022 將 GenAI 整合到圖中

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, graph, composition, streaming, advanced |

## Concept

當 LLM、VLM 或 ASR 的運作是更大 Neat 圖中的一個階段時，請使用 GenAI 圖片段。

## Walkthrough

大多數 GenAI 應用程式都應該從直接的模型 API 開始。當 GenAI 需要與其他 Neat 階段並列，例如命名輸入、命名輸出、路由或應用程式層級的協調時，圖的組合就會變得有用。

### 建立 GenAI 圖片段 {#step-create-fragment}

建立特定任務的模型處理器，設定圖片段選項，並建立一個公開的 `Graph` 片段。

視覺語言片段會公開 `prompt`、`image` 和 `use_cached_image` 輸入，以及 `tokens`、`done`、`encoded` 和 `error` 輸出。語音轉錄片段會公開 `audio` 和 `audio_path` 輸入，以及 `tokens`、`done` 和 `error` 輸出。

`SpeechTranscriberOptions` 預設為自動語言偵測和轉錄。在 C++ 中，將 `task` 設定為 `ASRTask::Translate`，或在 Python 中設定為 `ASRTask.Translate`，以將語音翻譯成英文。其 `done` 封包會報告偵測到的來源語言，並且在可用時，還會報告 `no_speech_prob` 和 `avg_logprob`。

### 將片段新增到應用程式圖 {#step-compose-graph}

將片段新增到更大的應用程式圖中。該片段會保留其公開端點名稱，因此應用程式程式碼可以按名稱推送和提取。

### 建置並推送圖的輸入 {#step-push-prompt}

將圖建置為 `Run`，將圖像樣本推送至 `image` 輸入，然後將文字樣本推送至 `prompt` 輸入，並讓 GenAI 階段產生 token。

### 提取 token 和完成中繼資料 {#step-pull-results}

從 `tokens` 提取，直到收到 `done` 樣本為止。`done` 樣本是一個封包，其中包含已產生 token 數量和完成原因等欄位。

## Run

在 Modalix DevKit 上，使用 LLiMa CLI 從 Hugging Face 下載 LFM2-VL 1.6B VLM：

```bash
llima pull LFM2-VL-1.6B-a16w4
```

在 Modalix 上執行教學，並使用 DevKit 本機模型目錄和本機圖像：

**Python:**
```bash
python3 share/sima-neat/tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_022_compose_genai_into_graph
./build/tutorials-standalone/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

預期的輸出會顯示圖的描述，以及從 `tokens` 輸出中提取的串流式答案。

## In Practice

當 GenAI 是大型應用程式圖的一部分時，請使用此模式。對於簡單的請求/回應應用程式程式碼，請保留對 `GenAIModel`、`VisionLanguageModel` 和 `ASRModel` 的直接呼叫。

## 原始檔案
- C++：`tutorials/022_compose_genai_into_graph/compose_genai_into_graph.cpp`
- Python：`tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py`
