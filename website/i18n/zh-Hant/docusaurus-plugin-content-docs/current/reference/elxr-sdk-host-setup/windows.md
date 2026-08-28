---
title: "Windows 主機備註"
description: "為 Windows 11 主機準備 Neat SDK 和 DevKit-Sync。"
sidebar_position: 2
---

當您的主機為 Windows 11，且您想使用 DevKit-Sync 執行 Neat 開發環境（以下簡稱為 Neat SDK）時，請參考本指南。

## 先決條件

- Windows 11 主機。
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) 已安裝並正常運作。
- Docker Engine 安裝在 WSL 內部。
- `sima-cli` 已安裝在 WSL 內部。

## 從 WSL 開始。

請從您的 WSL Linux 發行版內部執行 Neat SDK 指令，而不是從 PowerShell 或命令提示字元執行。這包括 `sima-cli neat install sdk@release-2.1`。

## WSL 網路模式

設定 `%UserProfile%\\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
```

然後重新啟動 WSL：

```powershell
wsl --shutdown
```

這可讓 WSL 共享主機網路設定，有助於 DevKit-Sync 和 NFS 通訊。

## 建議的拓撲結構：Windows 直接連接至 DevKit。

對於 Windows 主機，建議採用 Windows 機器與 DevKit 之間的直接 USB/乙太網路連線。與將 DevKit 放置在更廣泛的共用網路中相比，這種方式通常更容易設定，而且 Windows 防火牆的變更範圍可以限定在與本機 DevKit 連接的介面上，而不是整個網路。與 Ubuntu 和 macOS 不同，除非您的環境已經驗證了共用網路的防火牆和 WSL 網路規則，否則對於 Windows，請優先選擇這種直接連線設定。

當 DevKit 需要透過直接連線共用 Windows 機器網路連線時，請使用網際網路連線共用 (ICS)。

1. 將 Windows 機器透過 Wi-Fi 或其他上游介面連接到網際網路。
2. 透過 USB/乙太網路轉接器，將 DevKit 連接到 Windows 電腦。
3. 在 DevKit 上，請將已連接的網路介面設定為 `DHCP`。
4. 在 Windows 上，請開啟 `Control Panel > Network and Internet > Network Connections`.
   您也可以按下 `Win + R` 鍵，執行 `ncpa.cpl`，然後按下 Enter 鍵。
5. 在連線至網際網路的網路介面卡上按一下滑鼠右鍵，然後選取「`Properties`」。
6. 開啟「`Sharing`」分頁。
7. 啟用 `Allow other network users to connect through this computer's Internet connection`。
8. 在 `Home networking connection`，請選擇已連接到電腦的 USB/乙太網路介面卡。 DevKit.
9. 套用變更後，如果 DevKit 未收到訊號，請重新連接 DevKit 側的連接器。
   IPv4 位址。

啟用 ICS 後，Windows 通常會將一個位址指派給已共享的網路介面卡，該位址為 `192.168.137.0/24`。
從 WSL 或 DevKit 主控台中找到 DevKit 的 IPv4 位址，然後確認從 WSL 存取 SSH 的功能。

```bash
ssh sima@<devkit-ip>
```

然後繼續從 WSL 進行 DevKit 配對：

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

:::note Insight 在 Windows 上的直接連結
透過 Windows 的直接網路共用功能，Windows 防火牆和 WSL 的埠轉發行為可能會阻止網路上的其他機器存取 Neat Insight 網路介面。在此設定中，請直接在 Windows Neat SDK 主機上開啟 Insight，例如在 `https://localhost:9900`。
:::

## NFS 防火牆規則（PowerShell）

允許 Windows 防火牆中的 NFS 相關流量。以系統管理員身分執行 PowerShell，並使用 `New-NetFirewallRule` 新增必要的 NFS 連接埠/通訊協定的規則。

範例模式：

```powershell
New-NetFirewallRule -DisplayName "Allow NFS TCP 2049" -Direction Inbound -Protocol TCP -LocalPort 2049 -Action Allow
New-NetFirewallRule -DisplayName "Allow NFS UDP 2049" -Direction Inbound -Protocol UDP -LocalPort 2049 -Action Allow
```

新增您的 NFS 伺服器/使用者端設定所需的任何額外通訊埠。

## 下一步

傳回 [Neat SDK](/getting-started/dev-environment/)，繼續進行安裝/設定。
