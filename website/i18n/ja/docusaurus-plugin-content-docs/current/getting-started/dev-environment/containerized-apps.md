---
title: "コンテナ化されたアプリケーションの開発"
description: "Palette SDK でコンテナイメージをビルドし、Modalix DevKit 上で実行します。"
sidebar_position: 7
---

コンテナは、アプリケーションとそれに必要なライブラリを1つにまとめます。そのため、DevKit 自体にそれらのライブラリをインストールしなくても、アプリケーションを DevKit 上で実行できます。Palette SDK では、ホスト上で ARM64 コンテナイメージをビルドし、ローカルのコンテナレジストリにプッシュしてから、`dk container` で DevKit 上で起動します。

## コンテナを使用する場面

DevKit の Arm CPU 上で動作するソフトウェアには、コンテナを使用します。例:

- eLxr が提供していないライブラリを必要とするサービスやツール。
- Ubuntu など、別の Linux ディストリビューション向けにビルドされたアプリケーション。たとえば、eLxr で利用できるものとは異なる ROS 2 リリース向けにビルドされた ROS 2 アプリケーション。
- 同じライブラリの異なるバージョンを必要とする複数のアプリケーション。

:::warning コンテナからは MLA と DSP を使用できません
Ubuntu コンテナを含め、コンテナから MLA や DSP を使用することはサポートされていません。コンテナ内のアプリケーションは、DevKit の Arm CPU 上でのみ動作します。MLA や DSP を使用する Neat アプリケーションは、[Neat Library](/getting-started/neat-library/) に示すとおり、DevKit 上で直接実行してください。
:::

## 仕組み

このワークフローでは、Palette SDK の他の部分と同じ3つの場所を使用します。

| 場所 | 処理内容 |
| --- | --- |
| ホスト | `sima-cli sdk setup --devkit` が DevKit を SDK とペアリングし、ホスト上にローカルのコンテナレジストリを設定します。 |
| SDK コンテナ | `docker buildx` で ARM64 イメージをビルドし、ローカルレジストリにプッシュします。 |
| DevKit | `dk container deploy` がローカルレジストリからイメージをダウンロードし、Docker で起動します。 |

DevKit には Docker が同梱されていません。最初に `dk container` コマンドを実行すると、Docker のインストールが提案されます。

## 始める前に

次のものが必要です。

- ホスト上の Palette SDK %%release.sdk_series%% 以降。
- ホストから到達可能な、プラットフォームバージョン 3.0 の Modalix DevKit。バージョンを確認するには、DevKit 上で `cat /etc/buildinfo` を実行します。
- Docker をダウンロードするための、DevKit のインターネット接続。
- Docker がイメージとコンテナを保存する、DevKit 上の `/data` の空き容量。`/data` は約 3 GB なので、イメージは小さく保ってください。

## コンテナアプリケーションのビルドと実行

### 1. DevKit をローカルレジストリとペアリングする

ホスト上で、DevKit を SDK とペアリングします。

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

セットアップで **Set up a local container registry so the DevKit can download
images built in the SDK?** と尋ねられたら、`y` と答えます。ペアリングにより、DevKit ユーザーにはパスワードなしの `sudo` も付与されます。これは Docker のインストールに必要です。その後、新しい SDK シェルを開きます。

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

以前にレジストリなしで DevKit をペアリングした場合は、セットアップコマンドを再度実行し、`y` と答えてください。

### 2. DevKit に Docker を設定する

SDK シェルで、次を実行します。

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

DevKit に Docker を設定するかどうか尋ねられたら、`y` と答えます。DevKit 上で Docker がすでに動作している場合、コマンドはそれをそのまま使用します。完了すると、`Docker is
ready on the DevKit` と表示されます。インストールによる変更内容については、[DevKit への Docker のインストール](/reference/devkit-containers/#install) を参照してください。

### 3. イメージをビルドしてプッシュする

アプリケーション用の `Dockerfile` を含むディレクトリを作成します。たとえば、Ubuntu 24.04 上でシェルスクリプトを実行する `Dockerfile` は次のとおりです。

```dockerfile
FROM ubuntu:24.04
COPY hello.sh /usr/local/bin/hello.sh
CMD ["sh", "/usr/local/bin/hello.sh"]
```

SDK シェルで、そのディレクトリから ARM64 向けにイメージをビルドし、ローカルレジストリにプッシュします。

<ShellCommand prompt="sdk">
docker buildx build --platform linux/arm64 -t "$SIMA_CONTAINER_REGISTRY/my-app:latest" --push .
</ShellCommand>

DevKit は ARM64 イメージのみを実行します。必ず `--platform linux/arm64` を指定してビルドしてください。

### 4. DevKit 上でイメージを実行する

SDK シェルで、イメージを DevKit にダウンロードして起動します。

<ShellCommand prompt="sdk">
dk container deploy my-app:latest --detach --name my-app
</ShellCommand>

イメージ名にレジストリのアドレスを含める必要はありません。`dk` が追加します。イメージ名の後のオプションは `docker run` のオプションです。

コンテナの出力を確認してから、コンテナを停止して削除します。

<ShellCommand prompt="sdk">
dk container logs my-app --follow
dk container stop my-app
dk container remove my-app
</ShellCommand>

## 次のステップ

- すべての `dk container` コマンド、Docker の手動インストール、Ubuntu コンテナの実行、およびトラブルシューティングについては、[DevKit コンテナ](/reference/devkit-containers/) を参照してください。
- SDK と DevKit がファイルを共有し、コマンドを実行する仕組みについては、[DevKit Sync](/getting-started/dev-environment/devkit-sync/) を参照してください。
