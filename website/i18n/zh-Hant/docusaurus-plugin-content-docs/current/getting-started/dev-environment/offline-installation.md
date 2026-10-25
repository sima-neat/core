---
title: 離線安裝
description: 在目標主機無網際網路連線時安裝 SDK、Model Compiler 和 PCIe 套件
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

## 選擇套件

| 套件 | 用途 | 供應狀態 |
| --- | --- | --- |
| Neat SDK | 設定 SDK 開發環境。 | SDK 2.1.3.0 或更新版本 |
| Model Compiler | 在 SDK 中或獨立主機上安裝 Model Compiler。 | Model Compiler 2.1.3 或更新版本 |
| Neat PCIe | 在準備完成的 PCIe 主機及所有已連接的相容卡上安裝 Neat。 | 0.6.0 或更新版本 |
| Neat Library | 在無網際網路連線時安裝 core 套件。 | 0.6.0 或更新版本 |

## 開始之前

- 在可連上網際網路的電腦上安裝 `sima-cli`。請參閱[安裝環境](/getting-started/dev-environment/install-the-environment/)。
- 選擇與系統相容的套件版本。請參閱[相容性](/getting-started/compatibility/)。
- 確認目標主機的作業系統和處理器架構。
- 準備 USB 隨身碟或內部檔案共用空間，以供傳輸。

下載用電腦和目標主機可以使用不同的處理器架構。
請選擇適用於**目標主機**的套件。
請將所有套件檔案保存在一起，包括指令碼、中繼資料、檢查碼和相依項目。

## 安裝 Neat SDK {#download-the-sdk-offline-package}

### 1. 下載套件

在可連上網際網路的電腦上執行其中一個命令。
以下範例使用 SDK 2.1.3.0。

對於 `amd64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

對於 `arm64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

### 2. 安裝套件

1. 將下載的完整目錄複製到目標主機。
2. 在該目錄中開啟終端機。
3. 執行安裝程式：

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

## 安裝 Model Compiler {#download-the-model-compiler-offline-package}

### 1. 下載套件

檢查 [Model Compiler 相容性需求](/getting-started/compatibility/#model-compiler)。
在可連上網際網路的電腦上執行其中一個命令。
以下範例使用 Model Compiler 2.1.3。

對於 `amd64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

對於 `arm64` 目標主機：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
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

## 安裝 PCIe 適用的 Neat

PCIe 套件會在主機及所有已連接的相容卡上安裝 Neat。
目標系統不需要 Docker 或網際網路連線即可進行此安裝。
主機必須與每張卡建立可正常運作的 PCIe 網路連線。

### 1. 準備主機和卡

執行 Neat 安裝程式之前，請完成下列工作：

- 安裝相容的 Platform 主機驅動程式。
- 在每張卡上安裝相符的 Platform 3.0 韌體。
- 在每張卡的 PCIe 介面上設定探索功能。
- 為每張卡指派唯一的 PCIe IPv4 位址。
- 為每張卡設定 SSH 存取權限和可信任的 SSH 主機金鑰。
- 授予 SSH 帳戶在每張卡上使用 `sudo` 的權限。

Neat 安裝程式不會安裝 Platform 驅動程式或變更 Platform 韌體。
請使用 Platform 版本隨附的系統設定說明。

### 2. 下載套件

選擇適用於目標主機的 0.6.0 或更新版本。
套件支援使用 `amd64` 或 `arm64` 處理器的 Ubuntu 22.04 和 24.04 主機。

此範例在可連上網際網路的電腦上下載 0.6.0 版本。
對於 Ubuntu 24.04 `amd64` 目標主機，請執行：

<ShellCommand prompt="host">
sima-cli neat install 'core/pciehost/ubuntu24/amd64@v0.6.0' -t offline -d ./pcie-offline
</ShellCommand>

對於 Ubuntu 22.04，請將 `ubuntu24` 替換為 `ubuntu22`。
對於 `arm64` 主機，請將 `amd64` 替換為 `arm64`。
此命令會下載並解壓縮套件，不會安裝 Neat。

### 3. 檢查系統

1. 將完整的 `pcie-offline` 目錄複製到目標主機。請使用不含空格的路徑。
2. 開啟解壓縮後包含 `install.py` 的 `neat-offline` 目錄。
3. 閱讀隨附的 `INSTALL.md`，以瞭解必要的系統工具和 SSH 設定。
4. 執行檢查：

<ShellCommand prompt="host">
sudo python3 -B install.py --preflight-only
</ShellCommand>

安裝程式會自動探索所有已連接的卡。請勿提供卡的 IP 位址。
檢查會建立暫存檔案，但不會安裝套件。
安裝之前，請修正所有回報的錯誤。

### 4. 安裝並檢查結果

在同一個目錄中執行此命令：

<ShellCommand prompt="host">
sudo python3 -B install.py
</ShellCommand>

安裝程式會先檢查所有卡，再安裝套件。
安裝程式會先安裝並驗證主機，再依序處理每張卡。
確認最終摘要顯示主機及每張卡均已成功安裝並通過驗證。
接著執行應用程式驗收測試。

如果安裝失敗，請閱讀所顯示路徑中的診斷記錄。
如果安裝期間與卡的連線失敗，請先確認該卡的安裝程序已停止，再重試。
安裝失敗後，安裝程式不會自動還原先前的安裝狀態。

## 安裝 Neat Library

Neat Library 離線套件自 0.6.0 版本起提供。
對於使用 Platform 3.0 或更新版本的 Modalix DevKit，請使用此套件。

### 1. 下載套件

在可連上網際網路的電腦上執行此命令：

<ShellCommand prompt="host">
sima-cli neat install core@v0.6.0 -t offline --download-only
</ShellCommand>

### 2. 安裝套件

1. 將下載的完整目錄複製到目標主機。
2. 在該目錄中開啟終端機。
3. 請使用套件隨附的安裝說明。

### 套件選項與安裝程式行為

在可連線的電腦上下載，然後將封存檔複製到已佈建的 Modalix DevKit。預設組合包含執行階段套件及教學課程與測試的額外套件。使用 `-t minimal` 可省略額外套件。目標安裝程式會檢查 OS、架構與 Platform 相容性。

```bash
sima-cli neat install core/offline@v0.6.0 --download-only
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
```

解壓縮 `neat-core-offline-installer.tar.gz`，然後執行 `cd neat-core-offline && ./install.sh`。需要教學課程或測試時，請將另一個額外套件封存檔解壓縮到您選擇的目錄。

安裝程式僅接受 Neat eLxr SDK 或已佈建的 Modalix DevKit。它會將 `/etc/sdk-release` 或 `/etc/buildinfo` 與封存檔內的套件資訊清單比對，不需要下載的發佈中繼資料。其他環境或不相容的 Platform 版本會在安裝前遭到拒絕。在 SDK 中，Neat Library 會複製到 `/opt/toolchain/aarch64/modalix`。若已設定 `DEVKIT_SYNC_DEVKIT_IP`，同一個離線套件會透過 SSH 傳送至已配對的 DevKit。開始安裝前，兩個環境都必須通過預先檢查。在 DevKit 上則會於本機安裝。

標準 SDK 容器內的離線安裝不需要 `unshare`、特權模式或額外容器權限。安裝程式會將套件解壓縮到 sysroot，並僅啟用本機 APT 的 `file`、`copy` 與 `store` 方法。HTTP、HTTPS 及其他網路傳輸皆停用；缺少相依套件時會失敗，不會改用線上來源。原生 DevKit 安裝仍使用隔離的網路命名空間。

原生安裝程式會依據離線 APT 交易計畫，預留將安裝或重新安裝的 Debian 套件與所有 Python wheel 解壓後大小的三倍，另加 128 MiB 供中繼資料與 Python 環境使用。APT 不會變更的相依套件不計入此安裝空間預算。已放置的封存檔不會重複計算，包括位於 NVMe 或 tmpfs 的套件。空間不足時會以 MiB 顯示根檔案系統的可用空間、所需空間與尚需釋放的空間。

配對 DevKit 的傳輸使用 `/tmp` 下的私有目錄；標準 DevKit 的此目錄位於 RAM。作業結束後會刪除暫存目錄，包括已回報的安裝失敗與未完成的傳輸。若安裝期間 SSH 中斷或逾時，遠端安裝程式可能仍在執行，因此會保留目錄並顯示其路徑。

請以一般 SDK 或 DevKit 使用者執行 `./install.sh`。啟動腳本會自動要求 `sudo` 權限，並僅明確保留配對、DevKit 驗證、同步偏好及 SSH 代理設定。部署至配對裝置時仍使用啟動腳本的 SDK 使用者之 SSH 身分。

獨立 Neat Library 與 PCIe 安裝使用相同的即時終端顯示程式。獨立安裝的各步驟會保留編號階段與經過時間，每秒更新最多 10 行最近的命令輸出，並依終端大小裁切。重新導向輸出時，會以純文字每 15 秒顯示執行狀態。詳細報告會保留在私有的 `/tmp/neat-core-offline-*.log` 檔案中，失敗時也不刪除。使用 `NO_COLOR=1 ./install.sh` 可停用色彩，使用 `./install.sh --verbose` 可顯示完整日誌。重複的相容性檢查用於保護套件交易，與安裝作業是不同的步驟。

## 在內部儲存套件 {#hosting-packages-internally}

您可以將核准的套件儲存在內部檔案伺服器上。
請保持每個下載目錄的完整性，並與其他套件版本分開儲存。
執行安裝程式之前，請將完整目錄傳輸到目標主機。
