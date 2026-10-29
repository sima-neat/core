---
title: 離線安裝
description: 在目標主機無網際網路連線時安裝 SDK 和 Model Compiler 套件
sidebar_position: 6
---

:::tip 何時需要離線安裝
當目標主機無法從 SiMa.ai 服務下載套件時，請使用離線安裝。
當組織採取下列措施時，可能需要離線安裝：

- 封鎖大型下載。
- 限制存取套件登錄庫。
- 要求下載的套件經過核准後才能使用。
- 將開發主機與可連上網際網路的電腦隔離。
:::

在可連上網際網路的電腦上下載套件。
接著將套件傳輸到目標主機進行安裝。

如需執行階段套件，請參閱 [PCIe 主機離線安裝](/getting-started/neat-library/pcie-host/#offline-installation)或 [Neat Library 離線安裝](/getting-started/neat-library/install-or-update/#offline-installation)。

## 選擇套件

| 套件 | 用途 | 供應狀態 |
| --- | --- | --- |
| Neat SDK | 設定 SDK 開發環境。 | SDK 2.1.3.0 或更新版本 |
| Model Compiler | 在 SDK 中或獨立主機上安裝 Model Compiler。 | Model Compiler 2.1.3 或更新版本 |

## 開始之前

- 在可連上網際網路的電腦上安裝 `sima-cli`。請參閱[安裝 SDK](/getting-started/dev-environment/install-the-environment/)。
- 選擇與系統相容的套件版本。請參閱[相容性](/getting-started/compatibility/)。
- 確認目標主機的作業系統和處理器架構。
- 準備 USB 隨身碟或內部檔案共用空間，以供傳輸。

下載用電腦和目標主機可以使用不同的處理器架構。
請選擇適用於**目標主機**的套件。
請將所有套件檔案保存在一起，包括指令碼、中繼資料、檢查碼和相依項目。

## 安裝 Neat SDK {#download-the-sdk-offline-package}

### 1. 下載套件

在可連上網際網路的電腦上執行其中一個命令。
以下範例使用 SDK %%release.sdk%%。

對於 `amd64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install sdk@v%%release.sdk%% -t offline-amd64
</ShellCommand>

對於 `arm64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install sdk@v%%release.sdk%% -t offline-arm64
</ShellCommand>

### 2. 安裝套件

Model Compiler 安裝為選用步驟。
若不需要編譯模型，請略過 Model Compiler 下載，直接依照下方步驟安裝 SDK。
當 SDK 安裝程式詢問是否安裝 Model Compiler 時，請選擇略過。

若要隨 SDK 一起安裝 Model Compiler，請先[下載 Model Compiler 離線套件](#download-model-compiler-package)。
將 Model Compiler 離線套件檔案放在與 SDK 離線套件相同的目錄中。
SDK 安裝程式會自動找到該檔案。
出現提示時，請選擇從本機套件安裝。

1. 將下載的完整目錄複製到目標主機。
2. 在該目錄中開啟終端機。
3. 執行安裝程式：

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

## 安裝 Model Compiler {#download-the-model-compiler-offline-package}

:::tip 何時使用此程序
使用此程序可在既有 SDK 中或直接在主機上單獨安裝 Model Compiler。
若已在前一步驟安裝 SDK 和 Model Compiler，請略過本節。
:::

### 1. 下載套件 {#download-model-compiler-package}

檢查 [Model Compiler 相容性需求](/getting-started/compatibility/#model-compiler)。
在可連上網際網路的電腦上執行其中一個命令。
以下範例使用 Model Compiler %%release.model_compiler%%。

對於 `amd64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v%%release.model_compiler%% -t offline
</ShellCommand>

對於 `arm64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v%%release.model_compiler%% -t offline
</ShellCommand>

### 2. 安裝套件

選擇一個安裝位置。

**在 Neat SDK 中：**

1. 將下載的完整目錄複製到主機工作區，此工作區對應至 SDK 容器中的 `/workspace`。
2. 開啟 SDK shell。
3. 進入套件目錄。對於 `amd64` 套件，請使用：

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
</ShellCommand>

對於 `arm64` 套件，請使用 `/workspace/model-compiler-offline-arm64`。

4. 執行安裝程式：

<ShellCommand prompt="sdk">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

**在獨立主機上：**

1. 將下載的完整目錄複製到目標主機。
2. 在該目錄中開啟終端機。
3. 執行安裝程式：

<ShellCommand prompt="host">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

### 3. 啟用 Model Compiler

1. 重新啟動安裝時使用的 SDK shell 或主機 shell。
2. 啟用 Model Compiler：

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

若要離開 Model Compiler 環境，請執行：

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## 在內部儲存套件 {#hosting-packages-internally}

您可以將核准的套件儲存在內部檔案伺服器上。
請保持每個下載目錄的完整性，並與其他套件版本分開儲存。
執行安裝程式之前，請將完整目錄傳輸到目標主機。
