---
title: "Ubuntu 主機注意事項"
description: "準備一個 Ubuntu 主機，以便安裝 Neat SDK 和 DevKit-Sync。"
sidebar_position: 1
---

當您的主機是 Ubuntu，並且您想執行 Neat 開發環境（稱為 Neat SDK），請參考本指南，並搭配使用 DevKit-Sync。

## 先決條件

- Ubuntu 22.04 或 24.04 主機。
- Docker Engine 已安裝並正常運作。
- 已在主機上安裝了 `sima-cli`。
- Modalix 和 DevKit 可以在同一個網路中互相連線。

:::info 網路拓撲結構
在 Ubuntu 系統上，您可以選擇以下兩種方式來連接：一是將 DevKit 直接透過 USB/乙太網路連接到主機；二是將主機和 DevKit 分別放置在現有的網路中。如果它們位於現有的網路中，只要主機和 DevKit 能夠互相連線，以便進行 SSH 和 NFS 流量的傳輸，就不需要進行任何特殊的共享設定。
:::

## 直接從 Ubuntu 連接到 DevKit。

當 DevKit 直接透過 USB/乙太網路連接到 Ubuntu 機器，並且需要共用 Ubuntu 機器網路連線時，請使用此流程。

停用共用 DevKit 連接網路介面上的 IPv6。 DevKit-Sync 仰賴可預測的 IPv4 位址，以便進行 SSH 和 NFS 存取，如果讓共用連結上的 IPv6 保持啟用狀態，可能會導致探索和路由選擇變得不可靠。

### NetworkManager 圖形使用者介面

1. 將 Ubuntu 機器連接到網際網路，可以使用 Wi-Fi 或其他上游介面。
2. 透過 USB/乙太網路轉接器，將 DevKit 連接到 Ubuntu 機器。
3. 在 DevKit 上，請將已連接的網路介面設定為 `DHCP`。
4. 在 Ubuntu 系統中，開啟 `Settings > Network`。
5. 開啟已連接至 DevKit 的有線介面的設定。
6. 在「`IPv4`」分頁中，將「`IPv4 Method`」設定為「`Shared to other computers`」。
7. 在「`IPv6`」分頁中，將「`IPv6 Method`」設定為「`Disabled`」。
8. 套用變更後，請斷開有線介面的連線，然後重新連接。

連結建立後，請從 Ubuntu 系統中找到 DevKit 的 IPv4 位址：

```bash
ip neigh
```

在開始設定 SDK 之前，請確認 SSH 連線是否正常：

```bash
ssh sima@<devkit-ip>
```

接著繼續進行 DevKit 配對：

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

### NetworkManager 命令列介面

如果您偏好使用命令列進行設定，請找出與 DevKit 互動的介面：

```bash
nmcli device status
```

建立一個已啟用 IPv4 並停用 IPv6 的共用連線：

```bash
sudo nmcli connection add type ethernet ifname <devkit-interface> con-name devkit-shared ipv4.method shared ipv6.method disabled
sudo nmcli connection up devkit-shared
```

如果該介面已經存在連線設定檔，請修改現有的設定檔：

```bash
sudo nmcli connection modify "<connection-name>" ipv4.method shared ipv6.method disabled
sudo nmcli connection down "<connection-name>"
sudo nmcli connection up "<connection-name>"
```

## 防火牆注意事項

如果已啟用 Ubuntu 防火牆規則，請在執行 DevKit-Sync 設定之前，允許 SSH 和 NFS 流量通過面向 DevKit 的介面或子網路。 至少，DevKit 必須能夠連線到 SSH，以及由 `sima-cli sdk setup --devkit` 建立的主機 NFS 匯出。

## 下一步

傳回 [Neat SDK](/getting-started/dev-environment/)，繼續進行安裝/設定。
