---
title: "Model Browser"
sidebar_position: 4
---

# Model Browser

[Palette Model Browser](https://developer.sima.ai/models/) 列出 SiMa.ai 針對 Modalix 編譯並量測的模型。每個模型頁面都會顯示在 Modalix DevKit 上量測到的效能與準確度，並提供已編譯的模型供下載。

適用於下列情況：

- 為您的任務尋找模型，例如物件偵測或文字生成
- 比較模型及其變體的效能與準確度
- 下載可直接搭配 Neat 執行的已編譯模型
- 使用公開的指令碼自行重新建置模型

## 使用 sima-cli 下載模型

`sima-cli` 可從終端機讀取相同的模型。您不需要帳號。

搜尋模型：

```bash
sima-cli models list -q rf_detr_nano
```

選取模型，然後選取變體。變體是編譯模型的一種方式，以其權重與啟動值的位元寬度命名。此命令會顯示模型卡，其中包含量測到的準確度與效能，接著詢問是否要下載模型。

如果您已知道模型的 ID 與變體，可直接下載：

```bash
sima-cli models download --id rf_detr_nano --variant w8a8_w16a16
```

此命令會將模型儲存在 `models/<model id>/<variant>/` 之下。若要儲存到其他位置，請加上 `--output <folder>`。

授權條款禁止再散布的模型無法下載。請依照下一節的說明，使用其指令碼重新建置。

如需命令的詳細資訊，請參閱 [`sima-cli models`](/tools/sima-cli/models/) 參考資料。

## 重現模型

每個模型都附有用來建置及量測該模型的指令碼，並公開於 [models 存放庫](https://github.com/sima-neat/models)。每個模型都有一個 `models/<model id>/` 資料夾，每個步驟各有一個指令碼：

| 步驟 | 指令碼 | 執行位置 |
| --- | --- | --- |
| 將官方模型匯出為 ONNX | `convert.py` | 您的電腦 |
| 量化並編譯 | `compile.py` | Palette SDK 中的 Model Compiler |
| 量測效能 | `performance.py` | Modalix DevKit |
| 量測準確度 | `accuracy.py` | Modalix DevKit |

同一個資料夾中的 `model.yaml` 檔案會在 `build_profiles` 之下列出您可以建置的變體，以及每個步驟所需的資料集。

[入門指南](https://developer.sima.ai/models/docs/getting-started)會逐步說明每個步驟並提供可直接複製的命令，同時示範如何將您的結果與已發布的結果比較。
