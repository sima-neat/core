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

:::note アクセラレータへのアクセスには設定が必要です
必要なデバイスノードと互換性のあるランタイムライブラリが利用できれば、Neat アプリケーションはコンテナから MLA を使用できます。通常の Ubuntu イメージには、これらの依存関係は含まれていません。要件は[アクセラレータへのアクセス](/reference/devkit-containers/#accelerators)を参照してください。
:::

## 仕組み

このワークフローでは、Palette SDK の他の部分と同じ3つの場所を使用します。

| 場所 | 処理内容 |
| --- | --- |
| ホスト | `sima-cli sdk setup --devkit` が DevKit を SDK とペアリングし、ホスト上にローカルのコンテナレジストリを設定します。 |
| SDK コンテナ | `docker buildx` で ARM64 イメージをビルドし、ローカルレジストリにプッシュします。 |
| DevKit | `dk container deploy` がローカルレジストリからイメージをダウンロードし、Docker で起動します。 |

Docker がない場合、最初の `dk container` コマンドがインストールを提案します。

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
images built in the SDK?** と尋ねられたら、`y` と答えます。ペアリングにより、Docker のインストールに必要なパスワードなしの `sudo` アクセスが設定されます。その後、新しい SDK シェルを開きます。

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

以前にレジストリなしで DevKit をペアリングした場合は、セットアップコマンドを再度実行し、`y` と答えてください。

### 2. DevKit に Docker を設定する

SDK シェルで、次を実行します。

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

DevKit に Docker を設定するかどうか尋ねられたら、`y` と答えます。Docker がすでに動作している場合、コマンドはそれを再利用します。インストールによる変更内容は、[DevKit への Docker のインストール](/reference/devkit-containers/#install)を参照してください。

### 3. イメージをビルドしてプッシュする

アプリケーション用の `Dockerfile` を含むディレクトリを作成します。たとえば、Ubuntu 24.04 上でメッセージを表示する `Dockerfile` は次のとおりです。

```dockerfile
FROM ubuntu:24.04
CMD ["sh", "-c", "echo Hello from the DevKit"]
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

この例はメッセージを表示した後に終了します。出力を確認して、停止したコンテナを削除します。

<ShellCommand prompt="sdk">
dk container logs my-app --follow
dk container remove my-app
</ShellCommand>

長時間実行するアプリケーションでは、`dk container logs my-app --follow` で出力を監視し、削除する前に `dk container stop my-app` で停止します。

## 次のステップ

- すべての `dk container` コマンド、Docker の手動インストール、Ubuntu コンテナの実行、およびトラブルシューティングについては、[DevKit コンテナ](/reference/devkit-containers/) を参照してください。
- SDK と DevKit がファイルを共有し、コマンドを実行する仕組みについては、[DevKit Sync](/getting-started/dev-environment/devkit-sync/) を参照してください。
