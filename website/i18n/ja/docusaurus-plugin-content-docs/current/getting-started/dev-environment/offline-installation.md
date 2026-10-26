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

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

`arm64` ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

ダウンロードしたディレクトリを目的のホストにコピーします。そのディレクトリから、次のコマンドを実行してください。

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK 2.1.3.0 以降のバージョンでは、SDKオフラインパッケージがサポートされます。
:::

## Model Compiler のオフラインパッケージをダウンロードしてください。

対象環境および SDK の互換性要件に適合する Model Compiler パッケージをダウンロードしてください。互換性に関する詳細については、[互換性](/getting-started/compatibility/#model-compiler) を参照してください。

`amd64` のホスト上で動作する Model Compiler 2.1.3 について：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

`arm64` のホスト上で動作する Model Compiler 2.1.3 について：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
</ShellCommand>

:::note
Model Compilerのオフラインパッケージは、Model Compiler 2.1.3以降のバージョンでサポートされています。
:::

Neat SDK 内に Model Compiler をインストールするには、ダウンロードしたディレクトリを、SDKコンテナの `/workspace` フォルダにマッピングされているホストのワークスペースフォルダにコピーします。次に、SDKシェルを開き、対応する `/workspace` パスからインストーラーを実行します。

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

ARM64パッケージをダウンロードした場合は、代わりに「`arm64`」というディレクトリ名を使用してください。

スタンドアロンのホストにインストールする場合は、ダウンロードしたディレクトリを直接ターゲットのホストにコピーし、そのディレクトリから同じインストーラーを実行します。

インストール後、シェルの環境を再読み込みするか、SDKシェルを再起動してください。その後、次のコマンドでModel Compilerを起動します。

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

Model Compiler 環境を終了するには、次のコマンドを実行します。

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## ホスト環境を社内で構築

組織内で承認されたアーティファクトを内部的に複製している場合は、内部の場所に公開する際に、ダウンロードしたパッケージディレクトリをそのまま維持してください。メタデータファイル、インストールスクリプト、チェックサム、およびパッケージリソースは、まとめて保持されることが想定されます。

ユーザーは、内部パッケージディレクトリをダウンロードし、ターゲットホストで同じインストールスクリプトを実行できます。

## オフライン Core パッケージ

ネットワーク接続のあるコンピューターでダウンロードし、セットアップ済みの Modalix DevKit にアーカイブをコピーします。既定のプロファイルにはランタイムバンドルとチュートリアル・テストの追加パッケージが含まれます。追加パッケージを除外するには `-t minimal` を指定します。対象機器のインストーラーが OS、アーキテクチャ、Platform の互換性を確認します。

```bash
sima-cli neat install core/offline@v0.6.0 --download-only
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
```

`neat-core-offline-installer.tar.gz` を展開し、`cd neat-core-offline && sudo ./install.sh` を実行します。チュートリアルやテストが必要な場合は、別の追加アーカイブを任意のディレクトリに展開してください。

インストーラーは Neat eLxr SDK またはセットアップ済みの Modalix DevKit のみを受け入れます。`/etc/sdk-release` または `/etc/buildinfo` をアーカイブ内のパッケージマニフェストと照合するため、ダウンロードした公開用メタデータは不要です。他の環境や互換性のない Platform バージョンでは、インストール前に停止します。SDK では Core を `/opt/toolchain/aarch64/modalix` にコピーします。`DEVKIT_SYNC_DEVKIT_IP` が設定されていれば、同じオフラインバンドルを SSH 経由でペアリング済みの DevKit に転送します。インストール開始前に両方の環境が事前チェックに合格する必要があります。DevKit 上ではローカルにインストールします。

標準 SDK コンテナーでのオフラインインストールには `unshare`、特権モード、追加のコンテナー権限は不要です。パッケージを sysroot に展開し、ローカル APT メソッドの `file`、`copy`、`store` のみを有効にします。HTTP、HTTPS などのネットワーク転送は無効で、依存パッケージが不足してもオンライン取得には切り替えず失敗します。ネイティブ DevKit のインストールは隔離されたネットワーク名前空間を維持します。
