---
title: "DevKit コンテナ"
description: "Modalix DevKit への Docker のインストール、dk container コマンド、Ubuntu コンテナ、およびトラブルシューティング"
sidebar_position: 11
---

# DevKit コンテナ

このページは、プラットフォームバージョン 3.0 の Modalix DevKit でコンテナを実行するためのリファレンスです。概要とエンドツーエンドのワークフローについては、まず [コンテナ化されたアプリケーションの開発](/getting-started/dev-environment/containerized-apps/) を参照してください。

:::warning コンテナからは MLA と DSP を使用できません
Ubuntu コンテナを含め、コンテナから MLA や DSP を使用することはサポートされていません。コンテナ内のアプリケーションは、DevKit の Arm CPU 上でのみ動作します。MLA や DSP を使用する Neat アプリケーションは、DevKit 上で直接実行してください。
:::

## DevKit への Docker のインストール {#install}

DevKit には Docker が同梱されていません。Palette SDK からインストールするか、DevKit 上で手動でインストールします。どちらの方法でも、次の処理を行います。

- Docker の Debian `trixie` リポジトリから、ARM64 向けの Docker CE (`docker-ce`、`docker-ce-cli`、および `containerd.io`) をインストールします。
- Docker のデータを `/data/docker` に、containerd のデータを `/data/containerd` に保存し、イメージでルートファイルシステムがいっぱいにならないようにします。
- コンテナのネットワークに必要な IP フォワーディングを有効にします。
- `docker` サービスと `containerd` サービスを起動し、起動時に毎回開始されるようにします。
- DevKit ユーザーを `docker` グループに追加し、ユーザーが `sudo` なしで `docker` を実行できるようにします。

どちらの方法でも、DevKit のインターネット接続が必要です。

### Palette SDK から {#install-sdk}

まず、`sima-cli sdk setup --devkit {devkit-ip}` で DevKit をペアリングします。ペアリングにより、DevKit ユーザーにはパスワードなしの `sudo` が付与されます。これはインストールに必要です。その後、SDK シェルで次を実行します。

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

DevKit に Docker を設定するかどうか尋ねられたら、`y` と答えます。この質問を省略するには、`dk container setup --yes` を実行します。DevKit 上で Docker がすでに動作している場合、コマンドはそれをそのまま使用します。インストールが中断された場合は、コマンドを再度実行してください。続きから再開されます。

他の `dk container` コマンドも、Docker がない場合はインストールを提案します。

### DevKit 上で手動で {#install-manual}

Palette SDK を使用しない場合は、この方法を使用します。以下のコマンドはすべて、SSH またはシリアルコンソールを介して DevKit 上で実行します。`sudo` はパスワードを要求します。

#### 1. eLxr パッケージミラーリストを一時的に移動する

手順 9 で元に戻します。

<ShellCommand prompt="devkit">
sudo mkdir -p /root/apt-disabled
sudo mv /etc/apt/sources.list.d/0000mirror.list /root/apt-disabled/
</ShellCommand>

#### 2. Docker のデータを /data に保存する

<ShellCommand prompt="devkit">
sudo mkdir -p /etc/docker /data/docker
echo '{"data-root": "/data/docker", "firewall-backend": "nftables"}' | sudo tee /etc/docker/daemon.json
</ShellCommand>

#### 3. IP フォワーディングを有効にする

<ShellCommand prompt="devkit">
printf 'net.ipv4.ip_forward=1\nnet.ipv6.conf.all.forwarding=1\n' | sudo tee /etc/sysctl.d/99-docker-forward.conf
sudo sysctl --system
</ShellCommand>

#### 4. Docker のパッケージリポジトリを追加する

<ShellCommand prompt="devkit">
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo 'deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian trixie stable' | sudo tee /etc/apt/sources.list.d/docker.list
</ShellCommand>

#### 5. Docker をインストールする

<ShellCommand prompt="devkit">
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
</ShellCommand>

#### 6. containerd のデータを /data に移動する

<ShellCommand prompt="devkit">
sudo systemctl stop docker.socket docker containerd
sudo mkdir -p /data/containerd
sudo sh -c 'mv /var/lib/containerd/* /data/containerd/ 2>/dev/null; true'
</ShellCommand>

#### 7. 起動時に毎回 /data/containerd をマウントする

<ShellCommand prompt="devkit">
echo '/data/containerd /var/lib/containerd none bind 0 0' | sudo tee -a /etc/fstab
sudo mount /var/lib/containerd
</ShellCommand>

#### 8. サービスを起動する

<ShellCommand prompt="devkit">
sudo systemctl enable containerd docker
sudo systemctl restart containerd docker
</ShellCommand>

#### 9. eLxr パッケージミラーリストを元に戻す

<ShellCommand prompt="devkit">
sudo mv /root/apt-disabled/0000mirror.list /etc/apt/sources.list.d/
</ShellCommand>

#### 10. ユーザーを docker グループに追加する

<ShellCommand prompt="devkit">
sudo usermod -aG docker "$USER"
</ShellCommand>

グループの変更を有効にするには、いったんログアウトしてから再度ログインしてください。

## dk container コマンド {#dk-container}

これらのコマンドは Palette SDK シェルで実行します。コマンドは SSH を介してペアリング済みの DevKit に接続し、そこで Docker を実行します。

| コマンド | 機能 | ローカルレジストリが必要 |
| --- | --- | --- |
| `dk container setup [--yes]` | DevKit に Docker をインストールするか、動作中のインストールをそのまま使用します。 | いいえ |
| `dk container deploy <image> [options] [-- command]` | ローカルレジストリから DevKit にイメージをダウンロードし、起動します。 | はい |
| `dk container run <image> [options] [-- command]` | DevKit 上にすでにあるイメージを起動します。 | はい |
| `dk container pull <image>` | ローカルレジストリから DevKit にイメージをダウンロードします。 | はい |
| `dk container images` | DevKit 上にある、ローカルレジストリから取得したイメージを一覧表示します。 | はい |
| `dk container list` | 実行中か停止中かにかかわらず、DevKit 上のすべてのコンテナを一覧表示します。 | いいえ |
| `dk container logs <container> [options]` | コンテナの出力を表示します。監視を続けるには `--follow` を追加します。 | いいえ |
| `dk container stop <container>` | コンテナを停止します。 | いいえ |
| `dk container remove <container> [options]` | コンテナを削除します。`rm` も使用できます。 | いいえ |
| `dk container help` | コマンドの概要を表示します。 | いいえ |

コマンドによる引数の扱い:

- **イメージ名。** イメージ名は、レジストリのアドレスを含めずに指定します (例: `my-app:latest`)。`dk` が、DevKit から見たローカルレジストリのアドレスを追加します。そのため、`deploy`、`run`、および `pull` は、ローカルレジストリのイメージでのみ動作します。
- **オプション。** イメージ名の後から `--` までの引数は、`--detach`、`--name`、`--rm`、`--network host` などの `docker run` オプションです。`--` の後の引数は、コンテナ内で実行するコマンドです。
- **アーキテクチャ。** `deploy` と `run` は、ARM64 ではないイメージを拒否します。

たとえば、イメージを起動してそのコマンドを上書きするには、次のようにします。

<ShellCommand prompt="sdk">
dk container run my-app:latest --rm -- sh -c 'uname -m'
</ShellCommand>

## Ubuntu コンテナの実行 {#ubuntu}

Docker Hub の Ubuntu イメージは、DevKit の Arm CPU 上で動作します。試すには、DevKit 上で次のコマンドを実行します。

<ShellCommand prompt="devkit">
docker run --rm ubuntu:24.04 cat /etc/os-release
</ShellCommand>

`PRETTY_NAME="Ubuntu 24.04.5 LTS"` またはそれ以降のポイントリリースが表示されます。対話型シェルを使用するには、`docker run --rm -it ubuntu:24.04 bash` を実行します。コンテナ内では、`apt-get` で ARM64 向けの Ubuntu パッケージをインストールできます。

コンテナは DevKit の Linux カーネルを共有します。独立したオペレーティングシステムのインストールではありません。コンテナ内で変更したファイルは、コンテナを削除すると失われます。保持したいファイルについては、`-v` を使用して DevKit のディレクトリをコンテナにマウントしてください。

Ubuntu 上でアプリケーションを配布するには、その `Dockerfile` を `FROM ubuntu:<release>` で始め、[コンテナ化されたアプリケーションの開発](/getting-started/dev-environment/containerized-apps/) に示すとおり、SDK からビルドしてデプロイします。`dk container run ubuntu:24.04` は動作しません。`dk` はローカルレジストリ内でイメージを探すためです。

:::warning Ubuntu では MLA と DSP を使用できません
Ubuntu コンテナ内のアプリケーションは、MLA や DSP を使用できません。Ubuntu コンテナは、Arm CPU 上でのみ動作するソフトウェアに使用してください。
:::

## インストールの確認 {#check}

DevKit 上で次を実行します。

<ShellCommand prompt="devkit">
docker info --format '{{.DockerRootDir}}'
docker run --rm hello-world
</ShellCommand>

1つ目のコマンドは `/data/docker` を表示します。2つ目は `Hello from Docker!` を表示します。

## /data の空き容量 {#storage}

Docker はイメージとコンテナを `/data` に保存します。容量は約 3 GB です。DevKit 上で空き容量を確認します。

<ShellCommand prompt="devkit">
df -h /data
</ShellCommand>

空き容量を増やすには、不要になったコンテナとイメージを削除します。

<ShellCommand prompt="devkit">
docker container prune
docker image prune -a
</ShellCommand>

`docker image prune -a` は、どのコンテナも使用していないイメージをすべて削除します。

## トラブルシューティング {#troubleshooting}

| メッセージ | 対処方法 |
| --- | --- |
| `The local container registry is not configured for this SDK shell.` | ホスト上で `sima-cli sdk setup --devkit {devkit-ip}` を実行し、コンテナレジストリに関する質問に `y` と答えてから、新しい SDK シェルを開きます。 |
| `Docker installation requires passwordless sudo` | `sima-cli sdk setup --devkit {devkit-ip}` で DevKit を再度ペアリングするか、Docker を[手動で](#install-manual)インストールします。 |
| `Image ... is for amd64, not ARM64.` | `docker buildx build --platform linux/arm64 ... --push` でイメージを再ビルドします。 |
| `Image ... is not available on the DevKit.` | まず `dk container deploy` または `dk container pull` を使用してダウンロードします。 |
| DevKit 上での `permission denied while trying to connect to the Docker daemon socket` | Docker のセットアップ後にいったんログアウトしてから再度ログインし、ユーザーが `docker` グループに含まれるようにします。 |
| ビルド中の `exec format error` | `Dockerfile` 内の `RUN` ステップには、`x86_64` ホスト上での ARM64 エミュレーションが必要です。ホスト上で `docker run --privileged --rm tonistiigi/binfmt --install arm64` を実行してエミュレーションをインストールしてから、再度ビルドします。 |
