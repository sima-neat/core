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

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

針對 `arm64` 主機：

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

將下載的目錄複製到目標主機。從該目錄中執行：

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK 2.1.3.0 或更高版本的離線套件受到支援。
:::

## 下載 Model Compiler 離線套件

下載與目標環境和 SDK 相容性要求相符的 Model Compiler 套件。如需相容性詳細資訊，請參閱 [相容性](/getting-started/compatibility/#model-compiler)。

針對在 `amd64` 主機上執行的 Model Compiler 2.1.3：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

針對在 `arm64` 主機上執行的 Model Compiler 2.1.3：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
</ShellCommand>

:::note
針對 Model Compiler 2.1.3 或更新版本，我們支援使用 Model Compiler 的離線套件。
:::

若要在 Neat SDK 內部安裝 Model Compiler，請將下載的目錄複製到主機工作區資料夾中，該資料夾對應於 SDK 容器的 `/workspace` 資料夾。然後，開啟 SDK 終端機，並從對應的 `/workspace` 路徑執行安裝程式。

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

如果您下載了 ARM64 套件，請改用 `arm64` 目錄名稱。

對於獨立主機安裝，請將下載的目錄直接複製到目標主機，然後從該目錄執行相同的安裝程式。

安裝完成後，重新載入您的 Shell 環境或重新啟動 SDK Shell。然後，使用以下指令啟用 Model Compiler：

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

若要離開 Model Compiler 環境，請執行：

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## 內部託管方案

如果您的組織內部有已核准的成品複本，在將其發佈到內部位置時，請保持下載的套件目錄的完整性。預期中，中繼資料檔案、安裝指令碼、校驗總和以及套件資源應保持在一起。

使用者接著可以下載內部套件目錄，並在目標主機上執行相同的安裝指令碼。

## 離線 Core 套件

在可連線的電腦上下載，然後將封存檔複製到已佈建的 Modalix DevKit。預設組合包含執行階段套件及教學課程與測試的額外套件。使用 `-t minimal` 可省略額外套件。目標安裝程式會檢查 OS、架構與 Platform 相容性。

```bash
sima-cli neat install core/offline@v0.6.0 --download-only
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
```

解壓縮 `neat-core-offline-installer.tar.gz`，然後執行 `cd neat-core-offline && ./install.sh`。需要教學課程或測試時，請將另一個額外套件封存檔解壓縮到您選擇的目錄。

安裝程式僅接受 Neat eLxr SDK 或已佈建的 Modalix DevKit。它會將 `/etc/sdk-release` 或 `/etc/buildinfo` 與封存檔內的套件資訊清單比對，不需要下載的發佈中繼資料。其他環境或不相容的 Platform 版本會在安裝前遭到拒絕。在 SDK 中，Core 會複製到 `/opt/toolchain/aarch64/modalix`。若已設定 `DEVKIT_SYNC_DEVKIT_IP`，同一個離線套件會透過 SSH 傳送至已配對的 DevKit。開始安裝前，兩個環境都必須通過預先檢查。在 DevKit 上則會於本機安裝。

標準 SDK 容器內的離線安裝不需要 `unshare`、特權模式或額外容器權限。安裝程式會將套件解壓縮到 sysroot，並僅啟用本機 APT 的 `file`、`copy` 與 `store` 方法。HTTP、HTTPS 及其他網路傳輸皆停用；缺少相依套件時會失敗，不會改用線上來源。原生 DevKit 安裝仍使用隔離的網路命名空間。

原生安裝程式會依據離線 APT 交易計畫，預留將安裝或重新安裝的 Debian 套件與所有 Python wheel 解壓後大小的三倍，另加 128 MiB 供中繼資料與 Python 環境使用。APT 不會變更的相依套件不計入此安裝空間預算。已放置的封存檔不會重複計算，包括位於 NVMe 或 tmpfs 的套件。空間不足時會以 MiB 顯示根檔案系統的可用空間、所需空間與尚需釋放的空間。

配對 DevKit 的傳輸使用 `/tmp` 下的私有目錄；標準 DevKit 的此目錄位於 RAM。作業結束後會刪除暫存目錄，包括已回報的安裝失敗與未完成的傳輸。若安裝期間 SSH 中斷或逾時，遠端安裝程式可能仍在執行，因此會保留目錄並顯示其路徑。

請以一般 SDK 或 DevKit 使用者執行 `./install.sh`。啟動腳本會自動要求 `sudo` 權限，並僅明確保留配對、DevKit 驗證、同步偏好及 SSH 代理設定。部署至配對裝置時仍使用啟動腳本的 SDK 使用者之 SSH 身分。

安裝程式會以編號顯示傳輸、相容性檢查、SDK 安裝、配對 DevKit 安裝及清理步驟。終端輸出使用色彩，耗時步驟每 15 秒會顯示仍在執行的訊息。詳細的 APT、套件及相容性報告會儲存在私有的 `/tmp/neat-core-offline-*.log` 檔案中；安裝程式會顯示完整路徑，失敗時也會保留。使用 `NO_COLOR=1 ./install.sh` 可停用色彩。重複的相容性檢查用於保護套件交易，與安裝作業是不同的步驟。
