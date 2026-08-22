---
title: "兩步驟 SDK 安裝"
description: "適用於 SDK 2.0.0、2.1.2 和 2.1.2.1 的舊版兩步驟安裝流程。"
sidebar_position: 20
---

僅針對需要單獨拉取映像檔和執行設定指令的 SDK 版本使用此參考資料。這適用於 SDK 2.0.0、SDK 2.1.2.0 和 SDK 2.1.2.1。較新的 SDK 版本使用簡化的套件安裝流程，該流程已在 [安裝環境。](/getting-started/dev-environment/install-the-environment/) 中記錄。

舊版 SDK 安裝包含兩個步驟：

1. 拉取 SDK 容器映像。
2. 執行 SDK 設定，以建立工作區，並可選擇與 DevKit 配對。

## 拉取 SDK 映像檔。

從主機上執行與您 SDK 版本相符的映像檔安裝指令。

對於 SDK 2.0.0：

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.0.0
</ShellCommand>

對於 SDK 2.1.2 或 2.1.2.1，請使用與該版本配套的 2.1 映像標籤。

首次安裝可能需要幾分鐘，因為它會下載 SDK 容器映像。

## 執行 SDK 設定。

如果您的 DevKit 可以從主機存取，請在設定期間進行配對：

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

如果您的 DevKit 尚未連線，請在未配對的情況下設定 SDK 工作區：

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

在設定過程中，`sima-cli` 可能會要求您選擇已安裝的 SDK 映像檔、選擇主機工作目錄，以及設定 SDK 擴充目錄。使用 `--devkit` 時，設定程序還會要求您提供 DevKit 的連線資訊，並設定 DevKit 同步功能。

## 開啟 SDK Shell。

設定完成後，請開啟 SDK 指令列：

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

## 相容性

SDK 2.0.0 適用於 DevKit 軟體 2.0.0。SDK 2.1.2 和 2.1.2.1 適用於 DevKit 軟體 2.1.2。如需最新的相容性指引，請參閱 [相容性](/getting-started/compatibility/)。
