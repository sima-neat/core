---
title: "macOS 主機注意事項"
description: "準備一部 macOS 主機，以便使用 Neat SDK 和 DevKit-Sync。"
sidebar_position: 3
---

當您的主機是 macOS，並且您想執行 Neat 開發環境（稱為 Neat SDK），請參考本指南，並搭配使用 DevKit-Sync。

## 先決條件

- macOS 主機。
- 已在主機上安裝 `sima-cli`。
- Modalix 和 DevKit 可以在同一個網路中互相連線。

:::info 網路拓撲結構
在 macOS 上，您可以透過 USB/乙太網路將 DevKit 直接連接到主機，或者將主機和 DevKit 分開放置在現有的網路中。如果它們位於現有的網路中，只要主機和 DevKit 能夠互相連線以進行 SSH 和 NFS 流量傳輸，就不需要進行任何特殊的共享設定。
:::

## 安裝並執行 Colima。

安裝並啟動 Colima，以便在 macOS 上執行 Docker 工作負載。

```bash
brew install colima docker
colima start
docker ps
```

如果 Colima 已經安裝好，請在執行 `sima-cli sdk setup` 之前，確認它正在執行。

## macOS 上的 NFS 權限

DevKit-Sync 在 SDK 設定期間使用主機 NFS 匯出。在 macOS 上，請確保 `nfsd` 具有「完整磁碟存取」權限，否則主機工作區的匯出/掛載可能會失敗。

步驟：

1. 開啟「系統設定」。
2. 前往「隱私權與安全性」>「完整磁碟存取」。
3. 點擊 `+`，然後按下 `Cmd + Shift + G`，並輸入 `/sbin/`。
4. 選取 `nfsd`，並確認已允許其使用。
5. 在授權後重新執行 SDK 設定。

## 透過網路共享，建立 Mac 與 DevKit 之間的直接連線。

當 DevKit 無法直接連接到您的正常網路，且必須透過直接的 USB/乙太網路連接到您的 Mac 才能存取網際網路時，請使用此流程。

1. 在 DevKit 上，將已連接的網路介面設定為 `DHCP`（這通常是預設設定）。
2. 在 macOS 上，請開啟「`System Settings > General > Sharing > Internet Sharing`」。
3. 將「分享連線來源」設定為 `Wi-Fi`。
4. 啟用與連接至 DevKit 的 USB/乙太網路轉接器介面的資料共享功能。
5. 在 Mac 裝置上，請確認 USB/乙太網路轉接器的介面也已設定為使用 `DHCP`。

啟用網際網路共用功能後，Mac 上的作用中 USB/乙太網路介面應會收到一個位址（例如 `en0` 或 `en1`，具體取決於您的配接器和主機設定）。

### 在 DevKit 上進行 DNS 解決方案的調整。

在這種直接連線情境中，即使 DHCP 成功，DevKit 上的 DNS 可能仍然設定錯誤。如果名稱解析失敗，請更新 DevKit 上的 `/etc/resolv.conf`：

```bash
sudo nano /etc/resolv.conf
```

設定：

```text
nameserver 8.8.8.8
nameserver 127.0.0.1
```

## 使用 Colima UDP 轉發功能來排除 Insight 影片播放問題。

如果 Insight 在瀏覽器中開啟，但即時影片沒有顯示，則可能是 UDP 封包無法傳送到 SDK 容器。在 macOS 上使用 Colima 時，如果 Colima 使用 SSH 端口轉發，可能會發生這種情況。Docker 仍然可以顯示預期的 UDP 端口映射，但 Colima 的主機到虛擬機的轉發路徑可能無法將 UDP 流量傳送到容器中。

SDK 通常會發布以下 UDP 範圍：

- 用於影片的 `9000-9079/udp`。
- 用於元資料的 `9100-9179/udp`。
- 針對 WebRTC 的 `40000-40199/udp`。

如果 SDK 容器正在執行，並且存在這些 UDP 映射，請檢查 Colima 端口轉發器。SSH 轉發僅支援 TCP，而 gRPC 轉發則支援 TCP 和 UDP。

重新設定 Colima 以使用 gRPC 轉發：

```bash
colima stop
colima start --edit
```

在編輯器中，變更：

```yaml
portForwarder: ssh
```

致：

```yaml
portForwarder: grpc
```

然後重新啟動 SDK：

```bash
sima-cli sdk stop
sima-cli sdk start
sima-cli sdk neat
```

驗證 Docker 是否仍然會發布 UDP 埠：

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

確認 SDK 容器是否列出 `9000-9079/udp`、`9100-9179/udp`，以及 WebRTC UDP 範圍。

在傳輸視訊時，檢查 Insight 是否能偵測到來自 SDK 內部的傳入封包：

```bash
curl -k 'https://127.0.0.1:9900/api/ingest/stats?all=1&verbose=1'
```

請檢查預期的通道上，`packets_received` 是否有增加。請確認傳送端是否以 Mac 主機的 IP 位址和正確的 UDP 連接埠為目標，而不是 SDK 容器的 IP 位址。通道 0 使用 UDP `9000`，通道 1 使用 UDP `9001`，依此類推。

如果將 Colima 切換為 gRPC 轉發後，UDP 仍然無法傳送到 Insight，請使用 Docker Desktop 或 Linux 主機進行測試。在這種情況下，SDK 和 Docker 的連接埠映射很可能正確，剩下的可疑層級可能是 Colima 的 macOS UDP 轉發路徑。

## 下一步

傳回 [Neat SDK](/getting-started/dev-environment/)，繼續進行安裝/設定。
