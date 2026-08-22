# 021 部署 GenAI 模型

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Beginner |
| Estimated Read Time | 15-20 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4, Qwen3-VL-4B-Instruct-GPTQ-a16w4, whisper-small-a16w8 |
| Labels | genai, server, llm, vlm, asr, http |

## Concept

在 Neat GenAI 伺服器後方部署多個 GenAI 模型，以便 UI、服務或遠端客戶端可以從單一程序調用 LLM、VLM 和 ASR 端點。

## Walkthrough

對於大多數應用程式，請從 `GenAIServer` 及其與 OpenAI 相容的 `POST /v1/chat/completions` 端點開始。當嵌入的應用程式邏輯應在同一程序中擁有模型調用時，請使用直接的 `model.run(request)` 調用。

請參閱 [GenAI 伺服器參考](/develop-apps/development-workflow/genai-model/genai-server) 以獲取完整的端點和請求合約。

### 設定伺服器 {#step-configure-server}

選擇主機和端口。預設主機為 `0.0.0.0`，它接受來自可以訪問 Modalix 裝置的其他機器上的連接。

### 註冊模型目錄 {#step-register-models}

使用已部署的名稱新增每個已部署的模型目錄。本教學課程將註冊 `llm`、`vlm` 和 `asr`；已部署的名稱是客戶端在 `model` 欄位中發送的名稱。

### 開始部署 {#step-start-serving}

對於阻塞型前景程序，請調用 `serve()`，或者當您的應用程式擁有程序的其餘生命週期時，請調用 `start()`。

伺服器啟動後，使用 `GET /v1/models` 驗證已註冊的模型名稱：

```bash
curl http://<modalix-ip>:9998/v1/models
```

回應應包含在本教學課程中註冊的已部署名稱：`llm`、`vlm` 和 `asr`。

## Run

在 Modalix DevKit 上，使用 LLiMa CLI 從 Hugging Face 下載 LLM、VLM 和 ASR 模型：

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
llima pull whisper-small-a16w8
```

在 Modalix 上啟動伺服器，其中包含所有三個 DevKit 本機模型目錄：

**Python:**
```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/serve_genai_models.py \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_021_serve_genai_models
./build/tutorials-standalone/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

如果您只想在開發期間部署子集，請刪除 `--vlm` 或 `--asr`。

伺服器運行後，首先驗證是否已註冊所有已部署的名稱：

```bash
curl http://<modalix-ip>:9998/v1/models
```

然後從客戶端呼叫端點。將 `<modalix-ip>` 替換為您的 Modalix 裝置的 IP 位址或主機名稱。
以下請求客戶端使用 Python `requests`，串流回應，並列印伺服器端 TTFT 以及報告時的平均、最小值和最大每詞彙單元的 TPS。

### 向 LLM 發送文字請求

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_text.py \
  --server-ip <modalix-ip> \
  --model llm \
  "Give me three tips for designing a small REST API."
```

### 向 LLM 發送工具呼叫請求

與 OpenAI 相容的 `POST /v1/chat/completions` 端點和與 Ollama 相容的
`POST /api/chat` 端點接受 `tools` 陣列中的函數定義。每個項目
都必須具有 `type: "function"`、一個 `function` 物件，以及一個非空字串
`function.name`。函數描述和 JSON 結構描述參數可以包含在
`function` 物件中：

```bash
curl http://<modalix-ip>:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "stream": false
  }'
```

將 `tool_choice` 設置為 `"auto"`，以讓模型選擇已宣告的工具，或設置為 `"none"`
以停用工具提示和解析。省略 `tool_choice`，或將其設置為
`null`，在 `tools` 非空時，其行為與 `"auto"` 相同。格式不正確的工具定義、
非陣列的 `tools`，以及不受支援的 `tool_choice` 值或類型，將傳回 HTTP 400 錯誤，並顯示
`invalid_request_error`。相同的工具定義和工具選擇驗證
適用於在推理開始之前對直接的 `GenerationRequest` 呼叫。

### 向 VLM 發送文字和圖像請求

請求腳本以 base64 編碼圖像，並將其作為與 OpenAI 相容的 `image_url` 內容部分發送。

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_image.py \
  --server-ip <modalix-ip> \
  --model vlm \
  image.jpg \
  "What is the main subject of this image?"
```

### 向 ASR 模型發送音訊請求

轉錄客戶端預設為自動來源語言檢測。當已知來源語言時，請使用
`--language`：

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  speech.wav
```

要將語音翻譯成英文，請新增 `--translate`。客戶端將相同的
多部分請求發送到 `POST /v1/audio/translations`：

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  --translate \
  speech-in-another-language.wav
```

轉錄使用 `POST /v1/audio/transcriptions`。這兩個路由都支援
`stream=true`；提供的客戶端會串流文字，並列印檢測到的來源
語言、`no_speech_prob` 和 `avg_logprob`，來自最終事件。較高的
`no_speech_prob` 表示 Whisper 認為輸入不太可能
包含語音。`avg_logprob` 是生成詞彙單元的平均對數機率，其中較高的（不太負）值表示更可靠的解碼。

## In Practice

當網路邊界對應用程式有幫助時，請使用伺服器。對於同一進程中的低負載應用程式碼，請使用直接的 `GenAIModel`、`VisionLanguageModel` 和 `ASRModel` 呼叫。

對於一般應用程式，請執行一個 `GenAIServer` 進程，並使用多個已部署的模型名稱。如果 DevKit 具有足夠的記憶體，則多個伺服器進程可以綁定不同的端口，但它們會載入自己的模型實例，並且仍然共享相同的 MLA 硬體閘道器，因此不應將其視為一種增加硬體吞吐量的方法。

`/v1/models` 端點是最快速的初步測試：如果它傳回已部署的名稱，則表示伺服器可訪問，並且模型登錄已填充。

## 原始程式碼檔案
- C++：`tutorials/021_serve_genai_models/serve_genai_models.cpp`
- Python：`tutorials/021_serve_genai_models/serve_genai_models.py`
- 請求客戶端：
  - `tutorials/021_serve_genai_models/request_chat_completion_text.py`
  - `tutorials/021_serve_genai_models/request_chat_completion_image.py`
  - `tutorials/021_serve_genai_models/request_audio_transcription.py`
