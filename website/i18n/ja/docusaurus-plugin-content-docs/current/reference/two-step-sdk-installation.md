---
title: "SDK の 2 段階インストール"
description: "SDK 2.0.0、2.1.2、および 2.1.2.1 向けの従来の2段階インストール手順"
sidebar_position: 20
---

このドキュメントは、別途イメージのダウンロードとセットアップコマンドが必要なSDKリリースにのみ適用されます。SDK 2.0.0、SDK 2.1.2.0、およびSDK 2.1.2.1が該当します。それ以降の新しいSDKリリースでは、[環境をインストールします。](/getting-started/dev-environment/install-the-environment/)に記載されている簡略化されたパッケージインストール手順を使用します。

従来のSDKのインストールは、次の2つのステップで構成されます。

1. SDKコンテナイメージをプルします。
2. SDK のセットアップを実行して、ワークスペースを作成し、必要に応じて DevKit とペアリングします。

## SDKイメージをプルします。

ホストから、使用しているSDKのバージョンに対応するイメージインストールコマンドを実行してください。

SDK 2.0.0の場合：

<ShellCommand prompt="host">
sima-cli install ghcr:sima-neat/sdk:v2.0.0
</ShellCommand>

SDK 2.1.2 または 2.1.2.1 を使用する場合は、そのリリースに付属する対応する 2.1 イメージタグを使用してください。

最初のインストールには数分かかる場合があります。これは、SDK コンテナーイメージをダウンロードするためです。

## SDK のセットアップを実行します。

ホストからDevKitにアクセスできる場合は、セットアップ時にペアリングしてください。

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

まだ DevKit に接続できない場合は、ペアリングせずに SDK ワークスペースを設定してください。

<ShellCommand prompt="host">
sima-cli sdk setup
</ShellCommand>

セットアップ中に、`sima-cli` は、インストールされた SDK イメージの選択、ホストの作業ディレクトリの選択、および SDK 拡張機能ディレクトリの構成を要求する場合があります。`--devkit` を使用すると、セットアップ時に DevKit の接続情報も要求され、DevKit 同期が構成されます。

## SDKシェルを開きます。

セットアップが完了したら、SDKシェルを開きます。

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

## 互換性

SDK 2.0.0 は、DevKit ソフトウェア 2.0.0 用に設計されています。SDK 2.1.2 および 2.1.2.1 は、DevKit ソフトウェア 2.1.2 用に設計されています。最新の互換性情報については、[互換性](/getting-started/compatibility/) を参照してください。
