---
title: "DevKit 容器"
description: "在 Modalix DevKit 上安裝 Docker、dk container 指令、Ubuntu 容器以及疑難排解。"
sidebar_position: 11
---

# DevKit 容器

本頁是在平台版本 3.0 的 Modalix DevKit 上執行容器的參考資料。若要了解概觀和端對端工作流程，請從
[開發容器化應用程式](/getting-started/dev-environment/containerized-apps/) 開始。

## 存取加速器 {#accelerators}

Neat 應用程式可以在容器內使用 MLA。容器必須具備：

- 透過 Docker 的 `--device` 選項傳入必要裝置的存取權，例如 `/dev/mla`、`/dev/cvu` 和 `/dev/dma_heap/linux,cma`。
- 相容的 Neat 函式庫、GStreamer 外掛程式與平台使用者空間函式庫，可封裝於映像檔或從 DevKit 掛載。
- 相容的 DevKit 平台，以及支援容器平台偵測的 Neat Core 版本。應用程式必須符合執行階段的 ABI。

核心驅動程式與韌體仍由 DevKit 提供。僅傳入裝置並不會讓任意 Ubuntu 映像檔具備使用加速器的能力。DSP 工作負載需要各自相容的執行階段與裝置設定；本指南未確立任意容器映像檔對 DSP 的支援。

完整的裝置與函式庫設定請參閱官方的 [Docker 平行應用程式範例](https://github.com/sima-neat/apps/tree/develop/examples/docker/parallel-apps)。請遵循所選範例與平台版本的必要條件。

## 在 DevKit 上安裝 Docker {#install}

在自行管理的 DevKit 上，可從 Palette SDK 或手動安裝 Docker。在由管理員管理的 DevKit 上，請要求管理員啟用 Docker，並透過提供的服務使用 `dk container`，而不要自行安裝或取代 Docker。使用以下任一方式全新安裝時會：

- 從 Docker 的 Debian `trixie` 套件庫安裝 ARM64 版的 Docker CE（`docker-ce`、`docker-ce-cli` 和 `containerd.io`）。
- 將 Docker 的資料存放在 `/data/docker`，並將 containerd 的資料存放在
  `/data/containerd`，以免映像檔佔滿根檔案系統。
- 開啟容器網路所需的 IP 轉送。
- 啟動 `docker` 和 `containerd` 服務，並設定為每次開機時啟動。
- 將 DevKit 使用者加入 `docker` 群組，讓該使用者不需 `sudo` 即可執行 `docker`。

兩種方法都需要 DevKit 能存取網際網路。

### 從 Palette SDK 安裝 {#install-sdk}

請先使用 `sima-cli sdk setup --devkit {devkit-ip}` 配對 DevKit。對於自行管理的 DevKit，配對會設定安裝所需的免密碼 `sudo`。由管理員管理的 DevKit 會保留現有存取原則，並使用提供的 Docker 服務。接著在 SDK 殼層中執行：

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

詢問是否在 DevKit 上設定 Docker 時，請回答 `y`。若要略過問題，請執行 `dk container setup --yes`。如果 Docker 已可正常運作，指令會重複使用現有安裝。在由管理員管理的 DevKit 上，指令會透過提供的服務設定登錄存取權，而不授予不受限制的 `sudo` 權限。如果 SDK 管理的安裝中斷，請重新執行指令以繼續。

在自行管理的 DevKit 上，其他 `dk container` 指令也會在尚未安裝 Docker 時提示安裝。

### 在 DevKit 上手動安裝 {#install-manual}

未使用 Palette SDK，且要在自行管理的 DevKit 上全新安裝 Docker 時，請使用此方式。需要管理員存取權。若已有安裝，請保留其設定，改用 SDK 設定流程。以下所有指令均透過 SSH 或序列主控台在 DevKit 上執行。`sudo` 會要求輸入密碼。若某步驟失敗，請停止，並在疑難排解前依步驟 9 還原套件鏡像清單；不要繼續下一步。

#### 1. 暫時移開 eLxr 套件鏡像清單

您會在步驟 9 將其放回。

<ShellCommand prompt="devkit">
sudo mkdir -p /root/apt-disabled
sudo mv /etc/apt/sources.list.d/0000mirror.list /root/apt-disabled/
</ShellCommand>

#### 2. 將 Docker 的資料保存在 /data

<ShellCommand prompt="devkit">
sudo mkdir -p /etc/docker /data/docker
echo '{"data-root": "/data/docker", "firewall-backend": "nftables"}' | sudo tee /etc/docker/daemon.json
</ShellCommand>

#### 3. 開啟 IP 轉送

<ShellCommand prompt="devkit">
printf 'net.ipv4.ip_forward=1\nnet.ipv6.conf.all.forwarding=1\n' | sudo tee /etc/sysctl.d/99-docker-forward.conf
sudo sysctl --system
</ShellCommand>

#### 4. 新增 Docker 的套件庫

<ShellCommand prompt="devkit">
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo 'deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian trixie stable' | sudo tee /etc/apt/sources.list.d/docker.list
</ShellCommand>

#### 5. 安裝 Docker

<ShellCommand prompt="devkit">
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
</ShellCommand>

#### 6. 將 containerd 的資料移至 /data

<ShellCommand prompt="devkit">
sudo systemctl stop docker.socket docker containerd
sudo sh -eu <<'SH'
if mountpoint -q /var/lib/containerd; then
  echo '/var/lib/containerd is already mounted; stop and inspect its storage.' >&2
  exit 1
fi
mkdir -p /var/lib/containerd /data/containerd
if [ -n "$(find /data/containerd -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo '/data/containerd is not empty; stop and inspect it before migrating.' >&2
  exit 1
fi
cp -a /var/lib/containerd/. /data/containerd/
diff -qr /var/lib/containerd /data/containerd
SH
</ShellCommand>

只有在複製與比較成功後才可繼續。原始資料會保留在 `/var/lib/containerd`，直到被繫結掛載隱藏；請勿在設定期間刪除。如果複製失敗，請讓服務保持停止，並在重試前檢查兩個目錄。

#### 7. 每次開機時掛載 /data/containerd

<ShellCommand prompt="devkit">
sudo grep -Fqx '/data/containerd /var/lib/containerd none bind 0 0' /etc/fstab ||
  echo '/data/containerd /var/lib/containerd none bind 0 0' | sudo tee -a /etc/fstab
sudo mount /var/lib/containerd
</ShellCommand>

#### 8. 啟動服務

<ShellCommand prompt="devkit">
sudo systemctl enable containerd docker
sudo systemctl restart containerd docker
</ShellCommand>

#### 9. 放回 eLxr 套件鏡像清單

<ShellCommand prompt="devkit">
sudo mv /root/apt-disabled/0000mirror.list /etc/apt/sources.list.d/
</ShellCommand>

#### 10. 將您的使用者加入 docker 群組

<ShellCommand prompt="devkit">
sudo usermod -aG docker "$USER"
</ShellCommand>

請登出後再重新登入，群組變更才會生效。

## dk container 指令 {#dk-container}

請在 Palette SDK 殼層中執行這些指令。它們會透過 SSH 連線到已配對的 DevKit，並在該處執行 Docker。

| 指令 | 用途 | 需要本機登錄 |
| --- | --- | --- |
| `dk container setup [--yes]` | 在 DevKit 上安裝 Docker，或保留可正常運作的現有安裝。 | 否 |
| `dk container deploy <image> [options] [-- command]` | 將映像檔從本機登錄下載到 DevKit，然後啟動它。 | 是 |
| `dk container run <image> [options] [-- command]` | 啟動已在 DevKit 上的映像檔。 | 是 |
| `dk container pull <image>` | 將映像檔從本機登錄下載到 DevKit。 | 是 |
| `dk container images` | 列出 DevKit 上來自本機登錄的映像檔。 | 是 |
| `dk container list` | 列出 DevKit 上的所有容器，包括執行中和已停止的容器。 | 否 |
| `dk container logs <container> [options]` | 顯示容器的輸出。加上 `--follow` 可持續監看。 | 否 |
| `dk container stop <container>` | 停止容器。 | 否 |
| `dk container remove <container> [options]` | 移除容器。也可以使用 `rm`。 | 否 |
| `dk container help` | 顯示指令摘要。 | 否 |

這些指令處理引數的方式：

- **映像檔名稱。** 請提供不含登錄位址的映像檔名稱，例如 `my-app:latest`。`dk` 會加上 DevKit 用來存取本機登錄的位址。因此 `deploy`、`run` 和 `pull` 只適用於來自本機登錄的映像檔。
- **選項。** 映像檔名稱之後、`--` 之前的引數為 `docker run` 選項，例如 `--detach`、`--name`、`--rm` 或 `--network host`。`--` 之後的引數則是要在容器中執行的指令。
- **架構。** `deploy` 和 `run` 會拒絕非 ARM64 的映像檔。

例如，若要啟動映像檔並覆寫其指令：

<ShellCommand prompt="sdk">
dk container run my-app:latest --rm -- sh -c 'uname -m'
</ShellCommand>

## 執行 Ubuntu 容器 {#ubuntu}

來自 Docker Hub 的 Ubuntu 映像檔可在 DevKit 的 Arm CPU 上執行。若要試用，請在 DevKit 上執行此指令：

<ShellCommand prompt="devkit">
docker run --rm ubuntu:24.04 cat /etc/os-release
</ShellCommand>

它會顯示 `PRETTY_NAME="Ubuntu 24.04.5 LTS"` 或更新的修訂版本。若要使用互動式殼層，請執行 `docker run --rm -it ubuntu:24.04 bash`。在容器內，`apt-get` 會安裝 ARM64 版的 Ubuntu 套件。

容器與 DevKit 共用 Linux 核心，並非獨立安裝的作業系統。移除容器時，您在容器中變更的檔案會遺失。若要保留檔案，請使用 `-v` 將 DevKit 目錄掛載到容器中。

若要在 Ubuntu 上發佈應用程式，請讓其 `Dockerfile` 以
`FROM ubuntu:<release>` 開頭，並依照
[開發容器化應用程式](/getting-started/dev-environment/containerized-apps/) 所示，從 SDK 建置並部署。
`dk container run ubuntu:24.04` 無法運作，因為 `dk` 會在本機登錄中尋找映像檔。

上述 Ubuntu 指令僅示範 CPU 執行。若要執行 Neat 加速器工作負載，請遵循[存取加速器](#accelerators)；僅變更基礎映像檔不會提供必要的裝置與執行階段函式庫。

## 檢查安裝 {#check}

在 DevKit 上：

<ShellCommand prompt="devkit">
docker info --format '{{.DockerRootDir}}'
docker run --rm hello-world
</ShellCommand>

第一個指令會顯示 `/data/docker`。第二個指令會顯示 `Hello from Docker!`。

## /data 的可用空間 {#storage}

Docker 會將映像檔和容器存放在約 3 GB 的 `/data` 中。請在 DevKit 上檢查可用空間：

<ShellCommand prompt="devkit">
df -h /data
</ShellCommand>

若要釋出空間，請移除不再需要的容器和映像檔：

<ShellCommand prompt="devkit">
docker container prune
docker image prune -a
</ShellCommand>

`docker image prune -a` 會移除所有未被任何容器使用的映像檔。

## 疑難排解 {#troubleshooting}

| 訊息 | 解決方法 |
| --- | --- |
| `The local container registry is not configured for this SDK shell.` | 在主機上執行 `sima-cli sdk setup --devkit {devkit-ip}`，對容器登錄問題回答 `y`，然後開啟新的 SDK 殼層。 |
| `Docker installation requires passwordless sudo` | 在自行管理的 DevKit 上，請使用 `sima-cli sdk setup --devkit {devkit-ip}` 重新配對，或[手動](#install-manual)安裝 Docker。在由管理員管理的 DevKit 上，請要求管理員啟用 Docker。 |
| `Image ... is for amd64, not ARM64.` | 使用 `docker buildx build --platform linux/arm64 ... --push` 重新建置映像檔。 |
| `Image ... is not available on the DevKit.` | 先使用 `dk container deploy` 或 `dk container pull` 下載映像檔。 |
| `permission denied while trying to connect to the Docker daemon socket`（在 DevKit 上） | 設定 Docker 後請登出再重新登入，讓您的使用者加入 `docker` 群組。 |
| `exec format error`（建置時） | `Dockerfile` 中的 `RUN` 步驟在 `x86_64` 主機上需要 ARM64 模擬。請在主機上使用 `docker run --privileged --rm tonistiigi/binfmt --install arm64` 安裝，然後再次建置。 |
