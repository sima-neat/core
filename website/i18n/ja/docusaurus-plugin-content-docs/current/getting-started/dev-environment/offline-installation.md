---
title: "オフラインでのインストール"
description: "ネットワーク環境が制限された環境向けに、SDKとModel Compilerパッケージをダウンロードしてください。"
sidebar_position: 6
---

:::tip オフラインインストールを使用するタイミング
ターゲットマシンがSiMa.aiサービスから直接パッケージをダウンロードできない場合は、オフラインインストールを使用してください。これは、大規模なダウンロードをブロックしたり、レジストリへのアクセスを制限したり、アーティファクトの手動承認を要求したり、インターネットに接続されたマシンと開発ホストを分離したりする企業ネットワークではよくあることです。
:::

このワークフローでは、インターネットに接続されたマシンを使用してインストールパッケージを取得し、次にダウンロードしたディレクトリをUSBドライブ、内部ファイル共有、または内部でホストされているパッケージの場所に移動して、ターゲットのホストに転送します。

## SDKオフラインパッケージをダウンロード

ターゲットとなるホストのアーキテクチャに合わせて、SiMa.aiパッケージサービスにアクセスできるマシン上でコマンドを実行してください。

`amd64` ホストの場合：

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-amd64
</ShellCommand>

`arm64` ホストの場合：

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-arm64
</ShellCommand>

ダウンロードしたディレクトリを目的のホストにコピーします。そのディレクトリから、次のコマンドを実行してください。

<ShellCommand prompt="offline-host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK 2.1.2.3 以降のバージョンでは、SDKオフラインパッケージがサポートされます。
:::

## Model Compiler のオフラインパッケージをダウンロードしてください。

対象環境および SDK の互換性要件に適合する Model Compiler パッケージをダウンロードしてください。互換性に関する詳細については、[互換性](/getting-started/compatibility/#model-compiler) を参照してください。

`amd64` のホスト上で動作する Model Compiler 2.1.2 について：

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/amd64 -t offline -d ./model-compiler-offline-amd64
</ShellCommand>

`arm64` のホスト上で動作する Model Compiler 2.1.2 について：

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/arm64 -t offline -d ./model-compiler-offline-arm64
</ShellCommand>

:::note
Model Compilerのオフラインパッケージは、Model Compiler 2.1.2以降のバージョンでサポートされています。
:::

Neat SDK 内に Model Compiler をインストールするには、ダウンロードしたディレクトリを、SDKコンテナの `/workspace` フォルダにマッピングされているホストのワークスペースフォルダにコピーします。次に、SDKシェルを開き、対応する `/workspace` パスからインストーラーを実行します。

<ShellCommand prompt="username@neat-sdk-latest">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

ARM64パッケージをダウンロードした場合は、代わりに「`arm64`」というディレクトリ名を使用してください。

スタンドアロンのホストにインストールする場合は、ダウンロードしたディレクトリを直接ターゲットのホストにコピーし、そのディレクトリから同じインストーラーを実行します。

インストール後、シェルの環境を再読み込みするか、SDKシェルを再起動してください。その後、次のコマンドでModel Compilerを起動します。

<ShellCommand prompt="offline-host">
activate-model-compiler
</ShellCommand>

Model Compiler 環境を終了するには、次のコマンドを実行します。

<ShellCommand prompt="offline-host">
deactivate-model-compiler
</ShellCommand>

## ホスト環境を社内で構築

組織内で承認されたアーティファクトを内部的に複製している場合は、内部の場所に公開する際に、ダウンロードしたパッケージディレクトリをそのまま維持してください。メタデータファイル、インストールスクリプト、チェックサム、およびパッケージリソースは、まとめて保持されることが想定されます。

ユーザーは、内部パッケージディレクトリをダウンロードし、ターゲットホストで同じインストールスクリプトを実行できます。
