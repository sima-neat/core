---
title: "Windows ホストに関する注意事項"
description: "Windows 11のホストを、Neat SDKおよびDevKit-Sync用に準備します。"
sidebar_position: 2
---

ホストマシンが Windows 11 で、DevKit-Sync を使用して Neat 開発環境（Neat SDK と呼ばれる）を実行したい場合は、このガイドを参照してください。

## 前提条件

- Windows 11 ホスト。
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) がインストールされ、正常に動作しています。
- WSL内にインストールされたDocker Engine。
- `sima-cli` が WSL 内にインストールされました。

## WSLから始めましょう。

WSL Linuxディストリビューション内でNeat SDKコマンドを実行してください。PowerShellやコマンドプロンプトからは実行しないでください。これには、`sima-cli neat install sdk@release-2.1`も含まれます。

## WSL ネットワークモード

`%UserProfile%\\.wslconfig` を設定します。

```ini
[wsl2]
networkingMode=mirrored
```

次に、WSLを再起動してください。

```powershell
wsl --shutdown
```

これにより、WSLはホストのネットワーク構成を共有できるようになり、DevKit-SyncやNFS通信に役立ちます。

## 推奨される接続構成：WindowsとDevKitを直接接続

Windowsホストの場合、WindowsマシンとDevKit間の直接的なUSB/イーサネット接続が推奨される構成です。DevKitをより広範な共有ネットワークに配置するよりも、通常はセットアップが簡単であり、Windowsファイアウォールの変更範囲を、ネットワーク全体ではなく、ローカルのDevKitに接続されたインターフェースに限定できます。UbuntuやmacOSとは異なり、共有ネットワークに対してすでに検証済みのファイアウォールとWSLのネットワークルールがある環境でない限り、Windowsの場合はこの直接接続構成を優先してください。

DevKitが、直接接続を介してWindowsマシンのネットワーク接続を共有する必要がある場合は、インターネット接続共有（ICS）を使用します。

1. WindowsマシンをWi-Fiまたは別のネットワークインターフェースを介してインターネットに接続します。
2. USB/イーサネットアダプターを介して、DevKitをWindowsマシンに接続します。
3. DevKitでは、接続されているネットワークインターフェースの設定を`DHCP`のままにしてください。
4. Windowsでは、`Control Panel > Network and Internet > Network Connections` を開きます。
   「Windows」キーと「R」キーを同時に押して（`Win + R`）、`ncpa.cpl`と入力して「Enter」キーを押すことでも実行できます。
5. インターネットに接続されているアダプターを右クリックし、次に「`Properties`」を選択します。
6. 「`Sharing`」タブを開きます。
7. `Allow other network users to connect through this computer's Internet connection`を有効にします。
8. `Home networking connection`で、DevKitに接続されているUSB/イーサネットアダプターを選択してください。
9. 変更を適用し、DevKitが信号を受信しない場合は、DevKitに接続されているアダプターを再接続してください。
   IPv4アドレス。

ICSを有効にすると、Windowsは通常、共有アダプターに`192.168.137.0/24`の範囲内のアドレスを割り当てます。
WSLまたはDevKitコンソールからDevKitのIPv4アドレスを見つけ、WSLからSSHアクセスできることを確認します。

```bash
ssh sima@<devkit-ip>
```

次に、WSLからDevKitのペアリングを行います。

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

:::note Windows での直接リンクによる Insight へのアクセス
Windowsのダイレクトネットワーク共有を使用すると、WindowsファイアウォールとWSLのポートフォワーディングの動作により、ネットワーク上の他のマシンからNeat InsightのWebインターフェースにアクセスできなくなる場合があります。この設定では、たとえば`https://localhost:9900`で、WindowsのNeat SDKホスト上でInsightを直接開きます。
:::

## NFS ファイアウォールルール（PowerShell）

WindowsファイアウォールでNFS関連のトラフィックを許可します。管理者としてPowerShellを実行し、必要なNFSポート/プロトコルに対して、`New-NetFirewallRule`を使用してルールを追加します。

例：

```powershell
New-NetFirewallRule -DisplayName "Allow NFS TCP 2049" -Direction Inbound -Protocol TCP -LocalPort 2049 -Action Allow
New-NetFirewallRule -DisplayName "Allow NFS UDP 2049" -Direction Inbound -Protocol UDP -LocalPort 2049 -Action Allow
```

NFSサーバー/クライアントの設定に必要なポートを追加してください。

## 次のステップ

[Neat SDK](/getting-started/dev-environment/)に戻り、インストール/設定を続行してください。
