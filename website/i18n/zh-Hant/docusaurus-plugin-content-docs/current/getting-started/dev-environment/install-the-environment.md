---
title: "安裝開發環境"
description: "安裝並設定 Neat SDK 容器"
sidebar_position: 2
---

安裝最新的 Neat 開發環境（以下稱 Neat SDK）。它會以容器形式在主機上執行。

## 必要條件

- 確認主機符合[主機需求](/getting-started/dev-environment/#host-requirements)，包括安裝所需的系統管理員或 `sudo` 權限。
- 依照 [sima-cli 安裝指南](/tools/sima-cli/)，在主機上安裝 `sima-cli`。
- 完成所用平台的主機特定設定：
  - [Ubuntu 主機注意事項](/reference/elxr-sdk-host-setup/ubuntu)
  - [Windows 主機注意事項](/reference/elxr-sdk-host-setup/windows)
  - [macOS 主機注意事項](/reference/elxr-sdk-host-setup/macos)

若稍後要使用 DevKit Sync，還需要：

- 執行與 Neat SDK 相容軟體的 DevKit。詳情請參閱[相容性指南](/getting-started/compatibility/)。
- 位於同一個網路且允許 NFS 流量的主機與 DevKit。
- DevKit IP 位址。

## 安裝

安裝目前的 Neat SDK 2.1 發行通道：

<ShellCommand prompt="user-host-machine">
sima-cli neat install sdk@release-2.1
</ShellCommand>

首次安裝可能需要數分鐘，因為必須下載 Neat SDK 容器映像。下載完成後，安裝程式會開始設定 SDK，並詢問是否要與 Modalix DevKit 配對，以及是否要在 SDK 中安裝相容的 Model Compiler。

若選擇與 DevKit 配對，請在提示時輸入 DevKit IP 位址。設定流程會設定 SDK 工作區、啟動 SDK 容器並設定 DevKit Sync。若略過配對，系統仍會建立 SDK 工作區，之後也可以再進行配對。

`release-2.1` 套件會追蹤 2.1 系列最新的 Neat SDK 修補程式版本。目前版本為 Neat SDK 2.1.3.0，與 DevKit 軟體 2.1.3 相容。

設定期間，`sima-cli` 也會詢問是否要在 SDK 中安裝相容的 Model Compiler。若您會自行編譯或量化模型，請接受提示；不需要另外選擇版本。若只執行預先編譯的模型套件，則可略過。若要稍後安裝、固定特定修補版本或使用獨立主機，請參閱[安裝 Model Compiler](/getting-started/dev-environment/install-model-compiler/)及[相容性指南](/getting-started/compatibility/)。

:::note 舊版 SDK 使用傳統的兩階段安裝流程
若舊版 SDK 需要分別執行映像提取與設定命令，請參閱[兩階段 SDK 安裝](/reference/two-step-sdk-installation/)。
:::

若要在安裝後變更 SDK 設定，請參閱[設定 SDK](/getting-started/dev-environment/configure-sdk/)。受限網路環境請參閱[離線安裝](/getting-started/dev-environment/offline-installation/)。

## 访问 SDK

設定成功後，可以從終端機、Chrome 瀏覽器或 VS Code 访问 SDK。

### 使用 SDK Shell

使用下列命令開啟 Neat SDK shell：

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

### 使用 Chrome 瀏覽器

Neat Insight 由 SDK 內部提供，可從瀏覽器開啟。請在 SDK shell 中執行：

<ShellCommand prompt="username@neat-sdk-latest">
neat
</ShellCommand>

命令輸出的 **Web Access** 區段包含 Insight 與瀏覽器版 VS Code 的本機及遠端 URL。本機存取是從執行 SDK 的同一部電腦開啟 URL；遠端存取則是從另一部電腦開啟。若瀏覽器與 SDK 位於同一主機，建議使用本機 `127.0.0.1` URL，因為即使主機的網路 IP 改變仍可運作。遠端存取請使用由 `NFS_SERVER_HOST_IP` 衍生的 URL。VS Code URL 包含已設定的存取權杖，並會開啟設定的工作區。詳情請參閱 [Insight](/tools/insight/)。

### 使用 VS Code

可以透過 SDK Code UI 從瀏覽器使用 VS Code。SDK 安裝結束時，`sima-cli` 會輸出類似下列的 `codeUI` URL：

<ShellCommand prompt="user-host-machine">
codeUI      | https://192.168.76.4:10000/?tkn=gA5CS...&folder=/workspace
</ShellCommand>

在瀏覽器中開啟該 URL，即可在 SDK 工作區中作業。SDK 會在瀏覽器 Code UI 中預先安裝 Codex 與 Claude Code 擴充功能。

若偏好瀏覽器以外的方式，也可以使用原生 VS Code。使用 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 將 VS Code 連線至 SDK 容器。

在 SDK 容器中執行一次 `sima-cli login`，讓 SDK 可以擷取預先建置模型等資產。

## 升級

若要重新安裝或升級至目前的 SDK 套件，請從主機重新執行上述安裝命令：

<ShellCommand prompt="user-host-machine">
sima-cli neat install sdk@release-2.1
</ShellCommand>

若要更新現有 Neat SDK 容器內的 Neat 程式庫，請從容器 shell 執行 Neat CLI：

<ShellCommand prompt="username@neat-sdk-latest">
neat update
</ShellCommand>

這會更新目前 Neat SDK 中已安裝的 Neat 程式庫元件。若需要容器層級變更，這無法取代完整的容器映像升級。

若稍後刪除或重新建立 Neat SDK 容器，請在新容器內再次執行 `neat update`。

## 解除安裝

若要移除已安裝的 SDK 容器，請執行：

<ShellCommand prompt="user-host-machine">
sima-cli sdk remove
</ShellCommand>

## 後續步驟

- **要在 SDK 中編譯模型嗎？** 繼續前往完全在 SDK 中執行且不需要 DevKit 的[編譯模型](/compile-a-model/)。
- **要配對 DevKit 嗎？** 設定 [DevKit Sync](/getting-started/dev-environment/devkit-sync/)，以共用 SDK 工作區並在硬體上執行 `dk` 命令。
