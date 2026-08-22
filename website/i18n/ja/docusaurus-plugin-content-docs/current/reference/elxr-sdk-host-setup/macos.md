---
title: "macOS ホストに関する注意事項"
description: "macOSのホストを、Neat SDKおよびDevKit-Sync用に準備します。"
sidebar_position: 3
---

ホストマシンが macOS で、Neat を実行したい場合は、このガイドを参照してください。開発環境（Neat SDK と呼ばれます）と DevKit-Sync を使用します。

## 前提条件

- macOS ホスト。
- ホストに`sima-cli`がインストールされました。
- ModalixとDevKitは、同じネットワーク上でアクセス可能です。

:::info ネットワークトポロジー
macOSでは、DevKitをUSB/イーサネット経由でホストに直接接続するか、ホストとDevKitを既存のネットワーク上に個別に配置することができます。既存のネットワーク上に配置する場合、ホストとDevKitがSSHおよびNFSトラフィックで相互に通信できる限り、特別な共有設定は必要ありません。
:::

## Colimaをインストールして実行します。

Colimaをインストールして起動し、macOS上でDockerワークロードを実行できるようにします。

```bash
brew install colima docker
colima start
docker ps
```

Colima がすでにインストールされている場合は、`sima-cli sdk setup` を使用する前に、Colima が実行されていることを確認してください。

## macOSにおけるNFSのパーミッション設定

DevKit-Sync は、SDK のセットアップ中にホストの NFS エクスポートを使用します。macOS では、`nfsd` にフルディスクアクセス権が付与されていることを確認してください。そうしないと、ホストのワークスペースのエクスポートまたはマウントが失敗する可能性があります。

手順：

1. システム設定を開きます。
2. 「プライバシーとセキュリティ」>「フルディスクアクセス」に進みます。
3. `+`をクリックし、次に`Cmd + Shift + G`を押して、`/sbin/`と入力します。
4. `nfsd` を選択し、許可されていることを確認してください。
5. 権限が付与された後に、SDK のセットアップを再度実行してください。

## Mac と DevKit を直接接続するためのインターネット共有

この手順は、DevKit が通常のネットワークに直接接続できず、Mac との直接的な USB/イーサネット接続を通じてインターネットにアクセスする必要がある場合に適用します。

1. DevKitで、接続されているネットワークインターフェースを`DHCP`に設定します（通常はこれがデフォルト設定です）。
2. macOSでは、`System Settings > General > Sharing > Internet Sharing`を開きます。
3. 「**接続を共有するデバイス**」を`Wi-Fi`に設定します。
4. DevKit に接続されている USB/イーサネットドングルインターフェースへの共有を有効にします。
5. Mac の方では、USB/イーサネットドングルインターフェースも `DHCP` に設定されていることを確認してください。

インターネット共有が有効になると、Mac上のアクティブなUSB/Ethernetインターフェースはアドレスを受け取るはずです（例：`en0`または`en1`。これは、アダプターとホストの構成によって異なります）。

### DevKitにおけるDNSの回避策

このダイレクトリンクのシナリオでは、DHCPが正常に完了した後でも、DevKitのDNS設定が誤ったままになっている場合があります。名前解決に失敗した場合は、DevKit上の`/etc/resolv.conf`を更新してください。

```bash
sudo nano /etc/resolv.conf
```

セット：

```text
nameserver 8.8.8.8
nameserver 127.0.0.1
```

## Colima UDP転送を使用したInsightビデオのトラブルシューティング

Insight がブラウザで開くものの、ライブビデオが表示されない場合、UDPパケットがSDKコンテナに届いていない可能性があります。macOSでColimaを使用している場合、これはColimaがSSHポートフォワーディングを使用しているときに発生する可能性があります。Dockerは、期待されるUDPポートマッピングを表示する可能性がありますが、ColimaのホストからVMへのフォワーディングパスが、UDPトラフィックをコンテナに正しく転送しない場合があります。

SDKは通常、以下のUDP範囲を公開します。

- ビデオ用の`9000-9079/udp`
- メタデータ用の`9100-9179/udp`
- WebRTC用の`40000-40199/udp`

SDKコンテナが実行中で、それらのUDPマッピングが存在する場合、Colimaのポートフォワーダーを確認してください。SSHフォワーディングはTCPのみをサポートしますが、gRPCフォワーディングはTCPとUDPの両方をサポートします。

Colimaを再構成して、gRPCフォワーディングを使用するように設定します。

```bash
colima stop
colima start --edit
```

エディターで、以下を変更します。

```yaml
portForwarder: ssh
```

宛先：

```yaml
portForwarder: grpc
```

次に、SDKを再起動してください。

```bash
sima-cli sdk stop
sima-cli sdk start
sima-cli sdk neat
```

Docker が UDP ポートを正しく公開していることを確認してください。

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

SDKコンテナに、`9000-9079/udp`、`9100-9179/udp`、およびWebRTCのUDP範囲がリストされていることを確認してください。

ビデオを送信している間に、InsightがSDK内部からの受信パケットを認識しているかどうかを確認してください。

```bash
curl -k 'https://127.0.0.1:9900/api/ingest/stats?all=1&verbose=1'
```

指定されたチャンネルで、`packets_received`の値が増加していることを確認してください。送信元が、SDKコンテナのIPではなく、MacホストのIPと正しいUDPポートをターゲットにしていることを確認します。チャンネル0はUDP `9000`、チャンネル1はUDP `9001`を使用します。以下同様です。

ColimaをgRPCフォワーディングに切り替えても、UDPがInsightに届かない場合は、Docker DesktopまたはLinuxホストでテストしてください。その場合、SDKとDockerのポートマッピングはおそらく正しく、残りの疑わしい箇所はColimaのmacOS UDPフォワーディングパスである可能性があります。

## 次のステップ

[Neat SDK](/getting-started/dev-environment/)に戻り、インストール/設定を続行してください。
