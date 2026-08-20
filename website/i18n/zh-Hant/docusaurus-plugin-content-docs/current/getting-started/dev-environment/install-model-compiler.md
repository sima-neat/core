---
title: "安裝 Model Compiler。"
description: "在 Neat SDK 中或在支援的獨立主機上安裝 Model Compiler。"
sidebar_position: 5
---

:::tip 僅在單獨安裝 Model Compiler 時，才從這裡開始。
Model Compiler 在 SDK 安裝期間，安裝是選擇性的。只有在您跳過該提示、想要安裝較新且相容的模型編譯器，或需要安裝時，才使用此頁面。 Model Compiler 在支援的主機上，於 SDK 之外進行。
:::

Model Compiler 會對 ONNX 模型進行量化和編譯，以便它們可以在 SiMa.ai 的 MLA 上執行。當您自行編譯或量化模型（包括 GenAI 模型）時，這是**必要**的步驟；如果僅使用預先編譯的模型套件，則此步驟是**可選**的。

在安裝/設定 SDK 的過程中，`sima-cli` 會提示您將相應的模型編譯器安裝為擴充功能，置於 Neat SDK 內部。您也可以稍後再安裝，無論是在 Neat SDK 容器內部，還是以獨立方式安裝在受支援的 Ubuntu 主機上。
有關受支援的版本組合和獨立主機需求，請參閱 [相容性](/getting-started/compatibility/#model-compiler)。

## 在 SDK 內部安裝

如果您在設定 SDK 時跳過 Model Compiler，請稍後從 Neat SDK 內部安裝它。執行與您的 Neat SDK 容器架構相符的指令。若要檢查，請在 SDK shell 中執行 `uname -m`：`aarch64` 表示使用 `arm64` 指令，而 `x86_64` 表示使用 `amd64` 指令。

針對 `amd64` Neat SDK 容器：

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli install -v 2.1.2 tools/model-compiler/amd64
</ShellCommand>

針對 `arm64` Neat SDK 容器：

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli install -v 2.1.2 tools/model-compiler/arm64
</ShellCommand>

安裝完成後，請從 Neat SDK 的指令列介面啟動編譯器環境：

<ShellCommand prompt="username@neat-sdk-latest">
activate-model-compiler
</ShellCommand>

若要傳回預設的 Neat SDK 介面，請執行：

<ShellCommand prompt="username@neat-sdk-latest">
deactivate-model-compiler
</ShellCommand>

## 在獨立的主機上安裝

僅支援在 [相容性](/getting-started/compatibility/#model-compiler) 中列出的主機環境上進行獨立安裝。從支援的主機環境中執行對應的 `sima-cli install` 命令。若要檢查主機架構，請執行 `uname -m`：`x86_64` 使用 `amd64` 命令，而 `aarch64` 使用 `arm64` 命令。

針對在 `amd64` 主機上執行的 Model Compiler 2.1.2：

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.1.2 tools/model-compiler/amd64
</ShellCommand>

針對在 `arm64` 主機上執行的 Model Compiler 2.1.2：

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.1.2 tools/model-compiler/arm64
</ShellCommand>

針對在 `amd64` 主機上執行的 Model Compiler 2.0.0：

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.0.0 tools/model-compiler/amd64
</ShellCommand>

## 下一步

繼續[編譯模型](/compile-a-model/)。
