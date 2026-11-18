---
title: "DevKit コンテナ"
description: "Modalix DevKit への Docker のインストール、dk container コマンド、Ubuntu コンテナ、およびトラブルシューティング"
sidebar_position: 11
---

# DevKit コンテナ

このページは、プラットフォームバージョン 3.0 の Modalix DevKit でコンテナを実行するためのリファレンスです。概要とエンドツーエンドのワークフローについては、まず [コンテナ化されたアプリケーションの開発](/getting-started/dev-environment/containerized-apps/) を参照してください。

## アクセラレータへのアクセス {#accelerators}

Neat アプリケーションはコンテナ内で MLA を使用できます。コンテナには次のものが必要です。

- Docker の `--device` オプションで渡す `/dev/mla`、`/dev/cvu`、`/dev/dma_heap/linux,cma` などの必要なデバイスへのアクセス。
- イメージに同梱するか DevKit からマウントした、互換性のある Neat ライブラリ、GStreamer プラグイン、およびプラットフォームのユーザー空間ライブラリ。
- 互換性のある DevKit プラットフォームと、コンテナ内のプラットフォーム検出をサポートする Neat Core バージョン。アプリケーションはランタイムの ABI と一致する必要があります。

最新の DevKit では、`--user UID:GID` を指定して DevKit ユーザーの UID と GID でワークロードを実行し、マッピングしたデバイスへのアクセス権を確保します。Docker へのアクセスを設定した後は、MLA デバイスへのアクセスにアプリケーションの `sudo` や `--privileged` は不要です。現在の公式ランチャーでは、セキュリティ制限を解除するオプションも使用していません。Docker のインストールやホスト設定に必要な管理者権限と、アプリケーションのデバイス権限は別です。

カーネルドライバとファームウェアは引き続き DevKit が提供します。デバイスを渡すだけでは、任意の Ubuntu イメージでアクセラレータを使用できるようにはなりません。DSP ワークロードには独自の互換ランタイムとデバイス設定が必要です。このガイドは、任意のコンテナイメージでの DSP サポートを保証するものではありません。

デバイスとライブラリの完全な設定は、公式の[Docker での並列アプリケーションの例](https://github.com/sima-neat/apps/tree/develop/examples/docker/parallel-apps)を参照してください。選択した例とプラットフォームバージョンの前提条件に従ってください。

## DevKit への Docker のインストール {#install}

Palette SDK またはスタンドアロンインストーラーで Docker をインストールします。以下のいずれかの方法で新規インストールすると、次の処理が行われます。

- Docker の Debian `trixie` リポジトリから、ARM64 向けの Docker CE (`docker-ce`、`docker-ce-cli`、および `containerd.io`) をインストールします。
- Docker のデータを `/data/docker` に、containerd のデータを `/data/containerd` に保存し、イメージでルートファイルシステムがいっぱいにならないようにします。
- コンテナのネットワークに必要な IP フォワーディングを有効にします。
- `docker` サービスと `containerd` サービスを起動し、起動時に毎回開始されるようにします。
- DevKit ユーザーを `docker` グループに追加し、ユーザーが `sudo` なしで `docker` を実行できるようにします。

どちらの方法でも、DevKit のインターネット接続が必要です。

### Palette SDK から {#install-sdk}

まず、`sima-cli sdk setup --devkit {devkit-ip}` で DevKit をペアリングします。ペアリングにより、インストールに必要なパスワードなしの `sudo` アクセスが設定されます。その後、SDK シェルで次を実行します。

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

DevKit に Docker を設定するかどうか尋ねられたら、`y` と答えます。質問を省略するには、`dk container setup --yes` を実行します。Docker がすでに動作している場合は再利用します。インストールが中断された場合は、コマンドを再実行すると再開されます。

他の `dk container` コマンドも、Docker がない場合にインストールを提案します。

### DevKit 上での手動インストール {#install-manual}

Palette SDK を利用できない場合は、SDK のスタンドアロンインストーラーを使用します。ARM64 Modalix DevKit では、インターネット接続と sudo を使用できるユーザーアカウントが必要です。SSH またはシリアルコンソールから、SDK リポジトリのインストーラーをダウンロードし、通常のユーザーとして実行します。

<ShellCommand prompt="devkit">
curl --fail --location --output install-devkit-docker.sh \
  https://raw.githubusercontent.com/sima-neat/sdk/main/scripts/install-devkit-docker.sh
bash install-devkit-docker.sh install
</ShellCommand>

スタンドアロンの `install` モードでは、必要に応じて sudo パスワードを要求します。ストレージ設定や eLxr パッケージミラーの復元を含め、`dk container setup` と同じ Docker セットアップと中断時の復旧処理を使用します。既存の Docker 設定は保持されます。中断された場合は、同じコマンドを再実行してください。

ログアウトして再ログインし、Docker グループへの所属を有効にしてから、[インストールを確認](#check)してください。

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

上記の Ubuntu コマンドは CPU での実行のみを示します。Neat のアクセラレータワークロードでは、[アクセラレータへのアクセス](#accelerators)に従ってください。ベースイメージを変更するだけでは、必要なデバイスとランタイムライブラリは提供されません。

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
| `Docker installation requires passwordless sudo` | `sima-cli sdk setup --devkit {devkit-ip}` で再度ペアリングするか、[スタンドアロンインストーラー](#install-manual)を使用してください。 |
| `Image ... is for amd64, not ARM64.` | `docker buildx build --platform linux/arm64 ... --push` でイメージを再ビルドします。 |
| `Image ... is not available on the DevKit.` | まず `dk container deploy` または `dk container pull` を使用してダウンロードします。 |
| DevKit 上での `permission denied while trying to connect to the Docker daemon socket` | Docker のセットアップ後にいったんログアウトしてから再度ログインし、ユーザーが `docker` グループに含まれるようにします。 |
| ビルド中の `exec format error` | `Dockerfile` 内の `RUN` ステップには、`x86_64` ホスト上での ARM64 エミュレーションが必要です。ホスト上で `docker run --privileged --rm tonistiigi/binfmt --install arm64` を実行してエミュレーションをインストールしてから、再度ビルドします。 |
