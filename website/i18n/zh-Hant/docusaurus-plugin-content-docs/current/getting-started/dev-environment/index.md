---
title: "Neat SDK"
description: "設定 Neat SDK，以便快速開發已準備好用於部署的 Neat 應用程式。"
sidebar_position: 1
---

Neat 開發環境（以下簡稱 Neat SDK）是建構大型 Neat 應用程式並在 Modalix DevKit 上驗證它們的建議主機端工作區。它將建置工具、模型工具、硬體連接和已準備好的代理程式原始碼環境整合到一個容器化的工作流程中。

SDK 將三個地點連接起來：

- **主機：** 您安裝並啟動 SDK 容器的位置。
- **SDK 容器：** 您在此處建置應用程式、編譯模型、使用代理工具，以及檢查共用檔案。
- **Modalix DevKit：** 編譯後的模型成品和 Neat 應用程式在此處於硬體上執行。

DevKit Sync 將這些位置連接到一個共享的 `/workspace`，因此，建構輸出、日誌、模型成品和應用程式檔案都可以從主機、SDK 容器和 DevKit 存取，而無需手動複製步驟。這個共享工作區是 SDK 工作流程的中心。

<div class="overview-section-label">從這裡開始</div>

從安裝 SDK 開始。當您需要變更設定、稍後新增 Model Compiler、了解 DevKit Sync 的運作方式，或為受限制的網路準備離線套件時，請參考其他 SDK 主題。

:::tip 僅限 SDK 的理想情境
如果您已安裝 SDK 且尚未配對 DevKit，您只需要執行兩個步驟：
[安裝環境](/getting-started/dev-environment/install-the-environment/)，
然後 [編譯模型](/compile-a-model/)——模型編譯完全在 SDK 中執行。設定 SDK、DevKit Sync、安裝 Model Compiler（它會在設定過程中提供），而 Neat Library 和 PyNeat 頁面則是可選的附加步驟；只有在您需要時或配對 DevKit 後才訪問它們。
:::

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>SDK 主題</h2>
    <p>安裝 SDK，然後在需要變更設定、安裝 Model Compiler 或與配對的 DevKit 搭配使用時，使用選用的設定主題。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/install-the-environment/"><strong>安裝環境</strong><span>安裝並設定與您的 DevKit 軟體版本相符的 SDK 套件。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/devkit-sync/"><strong>DevKit Sync</strong><span>了解工作區共用、配對更新、rsync 備援機制等。 <code>dk</code> 命令執行。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/offline-installation/"><strong>離線安裝</strong><span>下載 SDK 和 Model Compiler 套件，以供在網路受限的環境中使用。</span></a></li>
    </ul>
  </section>
</div>

若要在安裝後變更 SDK 設定，例如工作區位置或 DevKit 配對設定，請參閱 [設定 SDK](/getting-started/dev-environment/configure-sdk/)。

Model Compiler 會在 SDK 設定期間提供。若要稍後安裝，請指定特定版本，或使用獨立的主機，請參閱 [安裝 Model Compiler。](/getting-started/dev-environment/install-model-compiler/)。

對於無法直接下載套件的主機，請參閱[離線安裝](/getting-started/dev-environment/offline-installation/)。

## 包含哪些內容

- **跨編譯環境：**在您的主機上的 Linux 容器中，建立用於 Modalix 的 C++ Neat 應用程式。
- **DevKit Sync：**將 SDK 與 Modalix DevKit 配對，並在兩個位置中公開相同的工作區。
- **模型工具：**在 SDK 中安裝相應的 Model Compiler。如果需要自行編譯或量化 ONNX 或 GenAI 模型，則必須安裝；如果僅使用預編譯的模型套件，則可以選擇性安裝。
- **Insight：**透過瀏覽器檢查工作區檔案、媒體來源、串流傳遞和執行階段行為。
- **已準備好的代理環境：**使用包含的 Codex 和 Claude 技能，以及最新的 Neat 原始碼參考和範例。

## 主機需求

在安裝 SDK 之前，請確認您的主機符合以下最低要求。
為了在每個支援的主機上安裝 SDK，都需要管理員權限（`sudo`），這不僅僅是為了選用的 DevKit 網路功能，因為安裝 `sima-cli`、Docker Engine、SDK 映像檔和 NFS 套件都需要更高的權限。

| 主機作業系統 | CPU | RAM | 可用磁碟空間 | 管理員 / sudo |
|---|---|---|---|---|
| Ubuntu 22.04 / 24.04 (`x86_64` 或 `arm64`) | 最少 4 個核心 | 最少 16 GB | 100 GB | 需要 `sudo` 權限來安裝 SDK（`sima-cli`、Docker、SDK 映像檔）、安裝/設定 NFS，以及設定共享網路/防火牆 |
| Windows 11 (透過 WSL) (`x86_64`) | 最少 4 個核心 | 最少 16 GB | 100 GB | 管理員權限，用於在 WSL 中安裝 SDK（Docker、`sima-cli`）、設定 WSL 網路，以及 NFS 防火牆規則 |
| macOS 15.5+ Apple Silicon (`arm64`) | 最少 4 個核心 | 最少 16 GB | 100 GB | 管理員權限，用於安裝 SDK（Homebrew、Colima、`sima-cli`）、啟用「完整磁碟存取」（`nfsd`），以及「網際網路共享」 |

:::note 生成式 AI 模型編譯需要更多資源。
編譯 GenAI 模型時，使用 LLiMa 的資源需求遠高於基本 SDK：建議使用 128GB 的 RAM，並最好使用 512GB 的磁碟空間，而且核心數量越多越好。
請參閱 [GenAI 設定](/genai-llima/setup/) 以了解完整的系統需求。
:::

## 支援的平台

| 平台 | 架構 | SDK | Model Compiler |
|---|---|---|---|
| 透過 Docker Engine 的 Ubuntu 22.04 和 24.04 | `x86_64` | 是 | 是 |
| 透過 WSL 和 Docker Engine 的 Windows 11 | `x86_64` | 是 | 是 |
| 透過 Docker Engine 的 Ubuntu 22.04 和 24.04 | `arm64` | 是 | Model Compiler 2.1.2 或更高版本 |
| 透過 Colima 的 macOS 15.5 或更高版本 | `arm64` | 是 | Model Compiler 2.1.2 或更高版本；請在 Neat SDK 內安裝。 |

:::note 架構名稱
`arm64` 和 `aarch64` 實際上是相同的 64 位元 Arm 架構——macOS 會將其報告為 `arm64`，而 Linux 會將其報告為 `aarch64`。同樣地，`x86_64` 和 `amd64` 也是相同的架構。在您的主機上（或在 SDK 內部）執行 `uname -m`，以查看您使用的是哪一種。Model Compiler 的安裝指令使用 `arm64` 和 `amd64`——請參閱 [安裝 Model Compiler。](/getting-started/dev-environment/install-model-compiler/)。
:::

:::note 安裝特定版本
標準安裝會採用目前支援的預設設定。在「[安裝環境](/getting-started/dev-environment/install-the-environment/)」頁面上使用的 `release-2.1` 通道，會隨時追蹤最新的 2.1 版本更新。若要指定確切的 SDK、Neat Library 或 Model Compiler 版本，請參閱「[相容性指南](/getting-started/compatibility/)」。
:::

## SDK 中的工具

當您為配對的 DevKit 建置應用程式時，建議您在此處安裝和更新 [Neat Library](/getting-started/neat-library/)。

若要從終端機、VS Code 或具有 Neat Insight 的瀏覽器存取 SDK，請參閱
[安裝環境](/getting-started/dev-environment/install-the-environment/#access-the-sdk)。

在 SDK 設定期間，`sima-cli` 會提示您自動安裝對應的 Model Compiler。如果您自行編譯或量化模型，請安裝它；如果您僅使用預先編譯的模型套件，則可以跳過此步驟。有關編譯器設定和使用方式，請參閱 [編譯模型](/compile-a-model/)。

## 下一步

首先，請執行 [安裝環境](/getting-started/dev-environment/install-the-environment/)。
