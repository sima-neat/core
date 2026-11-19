---
title: "開發容器化應用程式"
description: "在 Palette SDK 中建置容器映像檔，並在 Modalix DevKit 上執行。"
sidebar_position: 7
---

容器會將應用程式與其所需的函式庫一起封裝，因此應用程式可以在 DevKit 上執行，而不必在 DevKit 本身安裝這些函式庫。使用 Palette SDK 時，您會在主機上建置 ARM64 容器映像檔，將其推送到本機容器登錄，然後使用
`dk container` 在 DevKit 上啟動它。

## 何時使用容器

請將容器用於在 DevKit 的 Arm CPU 上執行的軟體，例如：

- 需要 eLxr 未提供之函式庫的服務或工具。
- 為其他 Linux 發行版（例如 Ubuntu）建置的應用程式。例如，針對與 eLxr 可用版本不同之 ROS 2 版本所建置的 ROS 2 應用程式。
- 需要同一函式庫不同版本的多個應用程式。

:::note 存取加速器需要設定
只要具備必要的裝置節點與相容的執行階段函式庫，Neat 應用程式就能從容器使用 MLA。一般 Ubuntu 映像檔不提供這些相依項目。需求請參閱[存取加速器](/reference/devkit-containers/#accelerators)。
:::

## 運作方式

此工作流程使用與 Palette SDK 其他部分相同的三個位置：

| 位置 | 執行內容 |
| --- | --- |
| 主機 | `sima-cli sdk setup --devkit` 會將 DevKit 與 SDK 配對，並在主機上設定本機容器登錄。 |
| SDK 容器 | 您使用 `docker buildx` 建置 ARM64 映像檔，並將其推送到本機登錄。 |
| DevKit | `dk container deploy` 會從本機登錄下載映像檔，並使用 Docker 啟動它。 |

尚未安裝 Docker 時，第一次執行 `dk container` 指令會提示安裝。

## 開始之前

您需要：

- 主機上安裝 Palette SDK %%release.sdk_series%% 或更新版本。
- 平台版本為 3.0 的 Modalix DevKit，且可從主機連線。若要檢查版本，請在 DevKit 上執行 `cat /etc/buildinfo`。
- DevKit 可存取網際網路，以便下載 Docker。
- DevKit 上 `/data` 有可用空間，Docker 會在此存放其映像檔和容器。`/data` 約為 3 GB，因此請盡量縮小映像檔。

## 建置並執行容器應用程式

### 1. 將 DevKit 與本機登錄配對

在主機上，將 DevKit 與 SDK 配對：

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

當設定程序詢問 **Set up a local container registry so the DevKit can download
images built in the SDK?** 時，請回答 `y`。配對會設定安裝 Docker 所需的免密碼 `sudo` 存取權。接著開啟新的 SDK 殼層：

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

如果您先前配對 DevKit 時未設定登錄，請再次執行設定指令並回答 `y`。

### 2. 在 DevKit 上設定 Docker

在 SDK 殼層中執行：

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

當詢問是否要在 DevKit 上設定 Docker 時，請回答 `y`。如果 Docker 已可正常運作，此指令會重複使用現有安裝。安裝變更內容請參閱[在 DevKit 上安裝 Docker](/reference/devkit-containers/#install)。

### 3. 建置並推送映像檔

為應用程式建立包含 `Dockerfile` 的目錄。例如，下列 `Dockerfile` 會在 Ubuntu 24.04 上印出訊息：

```dockerfile
FROM ubuntu:24.04
CMD ["sh", "-c", "echo Hello from the DevKit"]
```

在 SDK 殼層中，從該目錄建置 ARM64 映像檔，並將其推送到本機登錄：

<ShellCommand prompt="sdk">
docker buildx build --platform linux/arm64 -t "$SIMA_CONTAINER_REGISTRY/my-app:latest" --push .
</ShellCommand>

DevKit 只能執行 ARM64 映像檔。建置時請一律使用
`--platform linux/arm64`。

### 4. 在 DevKit 上執行映像檔

在 SDK 殼層中，將映像檔下載到 DevKit 並啟動它：

<ShellCommand prompt="sdk">
dk container deploy my-app:latest --detach --name my-app
</ShellCommand>

映像檔名稱中不需要包含登錄位址，`dk` 會自動加上。映像檔名稱之後的選項為 `docker run` 選項。

此範例會在印出訊息後結束。讀取輸出並移除已停止的容器：

<ShellCommand prompt="sdk">
dk container logs my-app --follow
dk container remove my-app
</ShellCommand>

對於長時間執行的應用程式，請使用 `dk container logs my-app --follow` 監看輸出，並在移除前使用 `dk container stop my-app` 停止容器。

## 後續步驟

- 關於所有 `dk container` 指令、手動安裝 Docker、執行 Ubuntu 容器以及疑難排解，請參閱
  [DevKit 容器](/reference/devkit-containers/)。
- 關於 SDK 與 DevKit 如何共用檔案及執行指令，請參閱
  [DevKit Sync](/getting-started/dev-environment/devkit-sync/)。
