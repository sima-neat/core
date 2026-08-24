---
title: "Ubuntu ホストに関する注意事項"
description: "Ubuntuのホストを、Neat SDKとDevKit-Sync用に準備します。"
sidebar_position: 1
---

ホストマシンが Ubuntu で、Neat を実行したい場合は、このガイドを参照してください。開発環境（Neat SDK と呼ばれる）と DevKit-Sync を使用します。

## 前提条件

- Ubuntu 22.04または24.04のホスト。
- Docker Engine がインストールされ、正常に動作しています。
- ホストに`sima-cli`がインストールされました。
- ModalixとDevKitは、同じネットワーク上でアクセス可能です。

:::info ネットワークトポロジー
Ubuntuでは、DevKitをUSB/イーサネット経由でホストに直接接続するか、ホストとDevKitを既存のネットワーク上に個別に配置することができます。既存のネットワーク上に配置する場合、ホストとDevKitがSSHおよびNFSトラフィックで相互に通信できる限り、特別な共有設定は必要ありません。
:::

## UbuntuからDevKitへの直接接続

DevKitをUSB/イーサネット経由でUbuntuマシンに直接接続し、Ubuntuマシンのネットワーク接続を共有する必要がある場合は、以下の手順を使用してください。

共有するDevKit側のネットワークインターフェースでIPv6を無効にします。DevKit-Syncは、SSHおよびNFSのために予測可能なIPv4アドレスに依存しており、共有リンクでIPv6を有効にしたままにすると、検出およびルーティングの選択が不安定になる可能性があります。

### NetworkManager GUI

1. UbuntuマシンをWi-Fiまたは別のアップストリームインターフェースを介してインターネットに接続します。
2. USB/イーサネットアダプターを介して、DevKitをUbuntuマシンに接続します。
3. DevKitでは、接続されているネットワークインターフェースの設定を`DHCP`のままにしてください。
4. Ubuntuでは、`Settings > Network`を開きます。
5. DevKitに接続されている有線インターフェースの設定を開きます。
6. 「`IPv4`」タブで、`IPv4 Method` を `Shared to other computers` に設定します。
7. 「`IPv6`」タブで、`IPv6 Method` を `Disabled` に設定します。
8. 変更を適用し、有線インターフェースを切断して再度接続してください。

リンクが確立されたら、UbuntuからDevKitのIPv4アドレスを調べてください。

```bash
ip neigh
```

SDK のセットアップを開始する前に、SSH アクセスが正常に機能することを確認してください。

```bash
ssh sima@<devkit-ip>
```

次に、DevKitのペアリングを行います。

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

### NetworkManager コマンドラインインターフェース

コマンドラインでの設定をご希望の場合は、DevKitと連携するインターフェースを特定してください。

```bash
nmcli device status
```

IPv6を無効にして、共有のIPv4接続を作成します。

```bash
sudo nmcli connection add type ethernet ifname <devkit-interface> con-name devkit-shared ipv4.method shared ipv6.method disabled
sudo nmcli connection up devkit-shared
```

そのインターフェースに接続プロファイルがすでに存在する場合は、代わりにそれを変更してください。

```bash
sudo nmcli connection modify "<connection-name>" ipv4.method shared ipv6.method disabled
sudo nmcli connection down "<connection-name>"
sudo nmcli connection up "<connection-name>"
```

## ファイアウォールに関する注意事項

Ubuntuのファイアウォールルールが有効になっている場合は、DevKitに接続するインターフェースまたはサブネットで、SSHおよびNFSトラフィックを許可してから、DevKit-Syncのセットアップを実行してください。最低限、DevKitは、SSHにアクセスでき、`sima-cli sdk setup --devkit`によって作成されたホストNFSエクスポートにアクセスできる必要があります。

## 次のステップ

[Neat SDK](/getting-started/dev-environment/)に戻り、インストール/設定を続行してください。
