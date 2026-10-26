---
title: "Neat 指令列介面"
description: "使用 neat 命令來檢查並更新已安裝的 Neat 環境。"
sidebar_position: 5
---

已安裝的函式庫提供 `neat` 環境指令。您可以從 SDK 或 DevKit 執行此指令，以查看已安裝元件的版本、已安裝的 `sima-cli` 指令集，以及是否有可用的較新成品。

在 SDK 中，狀態輸出還包含來自 `$HOME/.insight-config/neat-port-map.json` 的 Insight 主機與埠的對應關係。

缺少 PyNeat 或 Insight 安裝不會妨礙狀態回報。若版本探查失敗，元件會顯示
`unknown`，且 `neat` 會將提供者的診斷寫入標準錯誤。
JSON 輸出保持現有結構描述，並將無法取得的版本回報為 `null`。

拆分安裝會顯示 `simaai-runner` 與 `simaai-scheduler`。DevKit 從已安裝套件資料庫讀取版本；SDK 從快取 DEB 的控制中繼資料讀取版本。當舊版 `neat-runtime` 與 `neat-gst-plugins` 套件存在，或未偵測到拆分元件時，仍顯示其列。JSON 新增 `components.runner` 與 `components.scheduler`，並保留舊版 `runtime`、`gstPlugins` 鍵；不存在的版本仍為 `null`。

<ShellCommand prompt="sdk|devkit">
neat
</ShellCommand>

舊版安裝的輸出範例：

```text
Neat Environment
  Mode               Neat SDK
  Sysroot            /opt/toolchain/aarch64/modalix
  Update check       online

Components
  Neat core              %%release.core%% channel=release latest=%%release.core%%
  PyNeat                 %%release.core%%
  neat-runtime           %%release.core%%
  neat-gst-plugins       %%release.core%%
  neat-insight           %%release.insight%% channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         %%release.model_compiler%% run activate-model-compiler to activate

Exposed Ports
  Insight Web UI     https://10.0.0.22:9900

  Name               Protocol Host Port (Start) Host Port (End)
  ------------------ -------- ----------------- ---------------
  mainUI             tcp      9900              -
  metadataUDP        udp      9100              9179
  rtsp.tcp           tcp      8554              -
  videoUDP           udp      9000              9079
  videoUI            tcp      8081              -
  webRTC             udp      40000             40199
  webSSH             tcp      8022              -
```

## JSON 輸出

對於自動化和工具整合，請使用 JSON 輸出：

<ShellCommand prompt="sdk|devkit">
neat --json
</ShellCommand>

## 更新已安裝的元件

若要更新偵測到的管道中的 Neat Library 執行階段、`neat-insight`，以及已安裝的 `sima-cli` 劇本，請執行：

<ShellCommand prompt="sdk|devkit">
neat update
</ShellCommand>

若只要更新 Neat Library 執行階段，不檢查或更新 Insight 與操作手冊，請執行：

<ShellCommand prompt="sdk|devkit">
neat update --core-only
</ShellCommand>

## 下一步

繼續進行 [編譯模型](/compile-a-model/)，以準備用於 Modalix 的模型——
此程式碼在 SDK 中執行，且不需要 DevKit。若要在硬體上執行完整的應用程式，請參閱 [您好，Neat！](/develop-apps/hello-neat/minimal/)，它需要配對的 DevKit。

## PCIe 主機與卡片狀態

`sima-pcie-host` 套件也會在 Ubuntu 安裝 `neat`。執行 `neat` 或 `neat --json`，可讀取主機已安裝的套件、PyNeatPCIe，以及每張 SiMa PCIe 卡片的板卡與已安裝軟體資訊。卡片以 PCI 位址和介面列出。PCI 位址代表目前的連接位置，並非永久的板卡序號。

```bash
neat
neat --json
neat --offline
```

卡片查詢使用已設定的主機介面 `10.0.N.1` 與卡片位址 `10.0.N.2`，與 `pcie-setup.sh` 相同。請先設定網路連線與受信任的 SSH 金鑰。此命令不會設定介面、登錄主機金鑰、提示輸入密碼、在卡片上使用 sudo 或修改套件。缺少介面、非標準位址、SSH 失敗及無法讀取軟體資訊的情況會逐卡回報；可連線的卡片仍會顯示版本。

`NEAT_PCIE_SSH_USER` 指定卡片的 SSH 使用者（預設為 `sima`）。`NEAT_PCIE_SSH_KEY` 指定私密金鑰（預設為 `~/.ssh/sima_neat_pcie_ed25519`），也可繼續使用既有 SSH 設定與代理程式驗證。`--offline` 會略過網際網路更新檢查，但仍透過本機 SSH 讀取已連接的卡片。PCIe 主機狀態不檢查成品更新；請使用 `sima-cli neat install core/pciehost/<distro>/<arch>@<ref>` 更新主機套件。`neat update` 不會更新 PCIe 卡片。

JSON 保留 `sima.neat.status.v1`，並新增 `components.pcieHost`、`components.pcieScheduler`、`components.pyneatpcie` 與 `pcie`。`pcie` 包含主機軟體、具有 PCI 位址、介面、位址、狀態、軟體與錯誤欄位的 `cards` 陣列，以及拓樸錯誤。無法讀取遠端軟體時回傳 `null`，不會推測已安裝的版本。

PyNeat 和 PyNeatPCIe 狀態會列出每個偵測到的安裝版本及其虛擬環境路徑。不同環境會顯示為多個環境，而非版本衝突。對於虛擬環境以外的套件，輸出會改為顯示套件目錄。狀態查詢只讀取套件中繼資料，不會匯入原生模組。

JSON 中的每個 Python 項目會新增 `installations` 陣列。每筆記錄包含 `version`、`environment` 和 `path`（套件目錄）。在虛擬環境以外，`environment` 為 `null`。若安裝版本不同，摘要的 `version` 為 `null`；僅有多個安裝不會設定 `error`。指向同一安裝的符號連結路徑只會列出一次。
