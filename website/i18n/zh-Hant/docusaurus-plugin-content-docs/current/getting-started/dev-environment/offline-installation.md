---
title: "離線安裝"
description: "下載 SDK 和 Model Compiler 套件，以供在網路受限的環境中使用。"
sidebar_position: 6
---

:::tip 何時使用離線安裝？
當目標機器無法直接從 SiMa.ai 服務下載套件時，請使用離線安裝。這在企業網路中很常見，因為這些網路會限制大型下載、限制登錄檔存取、要求手動批准成品，或將可連線到網際網路的機器與開發主機分開。
:::

在這個工作流程中，請使用已連接網路的電腦來取得安裝套件，然後將下載的目錄透過 USB 隨身碟、內部檔案共享或內部託管的套件位置，移動到目標主機。

## 下載 SDK 離線套件

在能夠存取 SiMa.ai 套件服務的機器上，執行針對目標主機架構的指令。

針對 `amd64` 主機：

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-amd64
</ShellCommand>

針對 `arm64` 主機：

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-arm64
</ShellCommand>

將下載的目錄複製到目標主機。從該目錄中執行：

<ShellCommand prompt="offline-host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK 2.1.2.3 或更高版本的離線套件受到支援。
:::

## 下載 Model Compiler 離線套件

下載與目標環境和 SDK 相容性要求相符的 Model Compiler 套件。如需相容性詳細資訊，請參閱 [相容性](/getting-started/compatibility/#model-compiler)。

針對在 `amd64` 主機上執行的 Model Compiler 2.1.2：

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/amd64 -t offline -d ./model-compiler-offline-amd64
</ShellCommand>

針對在 `arm64` 主機上執行的 Model Compiler 2.1.2：

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/arm64 -t offline -d ./model-compiler-offline-arm64
</ShellCommand>

:::note
針對 Model Compiler 2.1.2 或更新版本，我們支援使用 Model Compiler 的離線套件。
:::

若要在 Neat SDK 內部安裝 Model Compiler，請將下載的目錄複製到主機工作區資料夾中，該資料夾對應於 SDK 容器的 `/workspace` 資料夾。然後，開啟 SDK 終端機，並從對應的 `/workspace` 路徑執行安裝程式。

<ShellCommand prompt="username@neat-sdk-latest">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

如果您下載了 ARM64 套件，請改用 `arm64` 目錄名稱。

對於獨立主機安裝，請將下載的目錄直接複製到目標主機，然後從該目錄執行相同的安裝程式。

安裝完成後，重新載入您的 Shell 環境或重新啟動 SDK Shell。然後，使用以下指令啟用 Model Compiler：

<ShellCommand prompt="offline-host">
activate-model-compiler
</ShellCommand>

若要離開 Model Compiler 環境，請執行：

<ShellCommand prompt="offline-host">
deactivate-model-compiler
</ShellCommand>

## 內部託管方案

如果您的組織內部有已核准的成品複本，在將其發佈到內部位置時，請保持下載的套件目錄的完整性。預期中，中繼資料檔案、安裝指令碼、校驗總和以及套件資源應保持在一起。

使用者接著可以下載內部套件目錄，並在目標主機上執行相同的安裝指令碼。
