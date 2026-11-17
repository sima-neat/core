---
title: オフラインでのインストール
description: 対象ホストでインターネットに接続せずに SDK と Model Compiler パッケージをインストールする
sidebar_position: 6
---

:::tip オフラインでのインストールが必要な場合
対象ホストが SiMa.ai サービスからパッケージをダウンロードできない場合は、オフラインでインストールします。
組織が次のような方針を採用している場合に必要になることがあります：

- 大容量のダウンロードをブロックする。
- パッケージレジストリへのアクセスを制限する。
- ダウンロードしたパッケージの使用前に承認を要求する。
- 開発ホストをインターネットに接続されたコンピューターから分離する。
:::

インターネットに接続されたコンピューターでパッケージをダウンロードしてください。
次に、インストールするパッケージを対象ホストに転送してください。

ランタイムパッケージについては、[PCIe ホストのオフラインインストール](/getting-started/neat-library/pcie-host/#offline-installation)または [Neat Library のオフラインインストール](/getting-started/neat-library/install-or-update/#offline-installation)を参照してください。

## パッケージを選択する

| パッケージ | 用途 | 提供状況 |
| --- | --- | --- |
| Palette SDK | SDK 開発環境をセットアップします。 | SDK 2.1.3.0 以降 |
| Model Compiler | SDK 内またはスタンドアロンホストに Model Compiler をインストールします。 | Model Compiler 2.1.3 以降 |

## 開始前の準備

- インターネットに接続されたコンピューターに `sima-cli` をインストールしてください。[SDK のインストール](/getting-started/dev-environment/install-the-environment/)を参照してください。
- システムと互換性のあるパッケージバージョンを選択してください。[互換性](/getting-started/compatibility/)を参照してください。
- 対象ホストのオペレーティングシステムとプロセッサーアーキテクチャを確認してください。
- 転送用の USB ドライブまたは内部ファイル共有を準備してください。

ダウンロード用コンピューターと対象ホストのプロセッサーアーキテクチャは異なっていてもかまいません。
**対象ホスト**用のパッケージを選択してください。
スクリプト、メタデータ、チェックサム、依存関係を含むすべてのパッケージファイルを一緒に保管してください。

## Palette SDK をインストールする {#download-the-sdk-offline-package}

### 1. パッケージをダウンロードする

インターネットに接続されたコンピューターで、いずれかのコマンドを実行してください。
以下の例では SDK %%release.sdk%% を使用します。

`amd64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install sdk@v%%release.sdk%% -t offline-amd64
</ShellCommand>

`arm64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install sdk@v%%release.sdk%% -t offline-arm64
</ShellCommand>

### 2. パッケージをインストールする

Model Compiler のインストールは任意です。
モデルをコンパイルする必要がない場合は、Model Compiler のダウンロードを省略し、以下の SDK インストール手順に進んでください。
SDK インストーラーで Model Compiler をインストールするか確認されたら、スキップを選択してください。

SDK と一緒に Model Compiler をインストールするには、[Model Compiler のオフラインパッケージをダウンロードしてください](#download-model-compiler-package)。
Model Compiler のオフラインパッケージファイルを SDK のオフラインパッケージと同じディレクトリに置いてください。
SDK インストーラーはファイルを自動的に検出します。
確認メッセージが表示されたら、ローカルパッケージからのインストールを選択してください。

1. ダウンロードしたディレクトリ全体を対象ホストにコピーしてください。
2. そのディレクトリでターミナルを開いてください。
3. インストーラーを実行してください：

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

## Model Compiler をインストールする {#download-the-model-compiler-offline-package}

:::tip この手順を使用する場合
この手順は、既存の SDK 内またはホスト上に直接 Model Compiler を別途インストールする場合に使用してください。
前の手順で SDK と Model Compiler の両方をインストールした場合は、このセクションを省略してください。
:::

### 1. パッケージをダウンロードする {#download-model-compiler-package}

[Model Compiler の互換性要件](/getting-started/compatibility/#model-compiler)を確認してください。
インターネットに接続されたコンピューターで、いずれかのコマンドを実行してください。
以下の例では Model Compiler %%release.model_compiler%% を使用します。

`amd64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v%%release.model_compiler%% -t offline
</ShellCommand>

`arm64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v%%release.model_compiler%% -t offline
</ShellCommand>

### 2. パッケージをインストールする

インストール先を 1 つ選択してください。

**Palette SDK 内：**

1. ダウンロードしたディレクトリ全体を、SDK コンテナーの `/workspace` にマッピングされたホストのワークスペースにコピーしてください。
2. SDK シェルを開いてください。
3. パッケージディレクトリに移動してください。`amd64` パッケージの場合は、次を使用してください：

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
</ShellCommand>

`arm64` パッケージの場合は、`/workspace/model-compiler-offline-arm64` を使用してください。

4. インストーラーを実行してください：

<ShellCommand prompt="sdk">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

**スタンドアロンホスト：**

1. ダウンロードしたディレクトリ全体を対象ホストにコピーしてください。
2. そのディレクトリでターミナルを開いてください。
3. インストーラーを実行してください：

<ShellCommand prompt="host">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

### 3. Model Compiler を有効にする

1. インストールに使用した SDK シェルまたはホストシェルを再起動してください。
2. Model Compiler を有効にしてください：

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

Model Compiler 環境を終了するには、次を実行してください：

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## パッケージを内部に保管する {#hosting-packages-internally}

承認済みパッケージを内部ファイルサーバーに保管できます。
ダウンロードした各ディレクトリは、内容をすべて保持し、他のパッケージバージョンと分けて保管してください。
インストーラーを実行する前に、ディレクトリ全体を対象ホストに転送してください。
