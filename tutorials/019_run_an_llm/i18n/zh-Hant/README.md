# 019 使用直接 API 執行 LLM

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Intermediate |
| Estimated Read Time | 10 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4 |
| Labels | genai, llm, chat, history, streaming |

## Concept

載入 GenAI 模型目錄，發送一個簡單的提示，新增一個系統提示，然後將相同的模式擴展到對話歷史記錄和串流中。

## Walkthrough

經典的 `Model` 教程使用 `.tar.gz` MPK 檔案。GenAI 模型改用 LLiMa 模型目錄和 `neat::genai` API。從最小的請求開始：載入一個模型，設定 `request.prompt`，執行它，然後列印答案。一旦成功，當您需要對話狀態時，切換到 `request.messages`。

### 載入模型目錄 {#step-load-model}

將 `GenAIModel` 指向已部署的 LLiMa 模型目錄。本教程使用 `GenAIModel`，因為它可以自動檢測目錄是否為 LLM、VLM 或 ASR 模型。

**C++：** 從模型路徑建構 `simaai::neat::genai::GenAIModel`。

**Python：** 從模型路徑建構 `pyneat.genai.GenAIModel`。

### 發送一個提示 {#step-send-prompt}

使用 `prompt` 和令牌預算建構 `GenerationRequest`。這是用於一次性問題、測試和腳本的最短路徑。

### 定義一個系統提示 {#step-system-prompt}

使用一個簡短的系統指令來引導模型的行為。您可以將其附加到一個簡單的提示請求中，使用 `system_prompt`；當您切換到對話歷史記錄時，將相同的指令作為 `system` 訊息放入訊息列表中。

### 切換到訊息 {#step-store-history}

對於對話樣式的請求，使用 `messages` 而不是 `prompt`：從一個系統訊息和一個使用者訊息開始，執行請求，然後儲存助理的回應。模型本身不會記住之前的 `run()` 呼叫；您的應用程式擁有訊息歷史記錄。

### 使用歷史記錄提出後續問題 {#step-follow-up}

附加另一個使用者訊息，發送更新後的訊息列表，然後讀取答案。模型現在可以看到您的應用程式保留的完整對話。

### 串流輸出答案 {#step-stream-answer}

對於 UI 樣式的輸出，呼叫 `stream()` 並迭代傳回的 `GenerationStream`。每個令牌樣本都包含最新的文字片段。

## Run

在 Modalix DevKit 上，使用 LLiMa CLI 從 Hugging Face 下載一個 LLM，例如 Qwen3 4B：

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

使用 DevKit 本機模型目錄執行 Modalix 上的教學：

**Python:**
```bash
python3 share/sima-neat/tutorials/019_run_an_llm/run_an_llm.py \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_019_run_an_llm
./build/tutorials-standalone/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

預期的輸出是一個簡單的提示回應、一個系統提示的回應、一個具有上下文意識的後續回應，以及一個串流式的最終回應。

## In Practice

僅保留您的應用程式所需的訊息歷史記錄量。過長的歷史記錄會消耗上下文標記，並增加產生第一個標記所需的時間。對於持續對話應用程式，請將對話儲存在模型物件之外，並為每次互動重新建構 `GenerationRequest.messages`。

## 原始程式碼檔案
- C++：`tutorials/019_run_an_llm/run_an_llm.cpp`
- Python：`tutorials/019_run_an_llm/run_an_llm.py`
