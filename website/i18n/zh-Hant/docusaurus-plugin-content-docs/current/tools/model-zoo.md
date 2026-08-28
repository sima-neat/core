---
title: "Model Zoo"
sidebar_position: 5
---

# Model Zoo

Model Zoo 是一套經過精心挑選的預先編譯和量化模型集合，可直接在 SiMa 裝置上執行。

適合在以下情況使用：

- 評估模型在 Modalix 硬體上的準確性和效能
- 避免對已知模型進行手動編譯和量化
- 從已驗證的模型成品開始
- 選擇針對特定硬體目標所建構的模型

Model Zoo 提供已預先編譯的模型成品，適用於 Neat C++ 和 PyNeat 應用程式，但不包含 GenAI 模型。

列出可用的模型：

```bash
sima-cli modelzoo list
```

下載前先檢視模型：

```bash
sima-cli modelzoo describe yolov5
```

下載模型成品：

```bash
sima-cli modelzoo get yolov5s
```

模型名稱可能會因版本而異。當您不確定該使用哪個模型識別碼時，請先使用 `sima-cli modelzoo list`。

如需指令的詳細資訊，請參閱 [`sima-cli modelzoo`](/tools/sima-cli/modelzoo/) 參考檔案。

## GenAI 模型

對於 GenAI，SiMa.ai 在 [Hugging Face](https://huggingface.co/simaai) 上提供預先編譯的 LLM、VLM 和 ASR 模型集合。請使用 LLiMa CLI 下載：

```bash
llima pull <model_name>
```

例如：

```bash
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

下載 GenAI 模型後，請使用 LLiMa 執行階段在 DevKit 上執行該模型。
有關設定和執行階段指令，請參閱 [搭配 LLiMa 的生成式 AI](/genai-llima/)。
