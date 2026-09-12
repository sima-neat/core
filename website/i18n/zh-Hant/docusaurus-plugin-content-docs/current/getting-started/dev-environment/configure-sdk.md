---
title: "設定 SDK"
description: "安裝後，您可以變更 Neat SDK 工作區、DevKit 配對方式或設定選項。"
sidebar_position: 4
---

:::tip 只有在變更 SDK 設定時，才從這裡開始。
SDK 安裝指令已經會下載 SDK 映像檔並進行設定。如果您需要變更 SDK 設定，例如工作區位置、DevKit 配對、SDK 擴充功能或其他設定選項，請使用此頁面。
:::

當您需要設定現有的 Neat SDK 容器時，請直接執行 `sima-cli sdk setup`。

## 使用 DevKit 配對功能進行設定

當您的 DevKit 可以從主機存取，且您想要新增或更新 DevKit 配對時，請使用此指令：

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

請事先準備好：

- DevKit 的 IP 位址；
- 如果需要安裝或設定 NFS，請提供您的主機管理員密碼；
- 當系統提示時，請提供 DevKit 使用者的認證資訊。預設使用者為 `sima`，預設密碼為 `edgeai`。

使用 `--devkit`，設定會啟用 DevKit Sync。它會透過 NFS 匯出您的主機工作區，並預設將其掛載到 DevKit 上，作為 `/workspace`。

## 在未配對 DevKit 的情況下進行設定

當您想要更新 SDK 設定，但尚未連線到 DevKit 時，請使用此指令：

<ShellCommand prompt="host">
sima-cli sdk setup
</ShellCommand>

在設定過程中，`sima-cli` 可能會要求您：

- 選擇一個主機工作區目錄。除非您需要不同的工作區，否則請接受預設值；
- 選擇一個 SDK 擴充目錄；
- 選擇是否安裝 Model Compiler。

您需要安裝 Model Compiler，以便自行編譯或量化模型。如果您僅使用預先編譯的模型套件，則可以選擇不安裝。如果您打算編譯或量化模型，請在此處安裝它，並在提示時完成 `sima-cli login`。

## 下一步

繼續前往 [DevKit Sync](/getting-started/dev-environment/devkit-sync/)，以了解工作區共用、配對更新、rsync 備援以及 `dk` 命令的詳細資訊。
