---
title: オフラインでのインストール
description: 対象ホストでインターネットに接続せずに SDK、Model Compiler、PCIe パッケージをインストールする
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

## パッケージを選択する

| パッケージ | 用途 | 提供状況 |
| --- | --- | --- |
| Neat SDK | SDK 開発環境をセットアップします。 | SDK 2.1.3.0 以降 |
| Model Compiler | SDK 内またはスタンドアロンホストに Model Compiler をインストールします。 | Model Compiler 2.1.3 以降 |
| Neat PCIe | 準備済みの PCIe ホストと、接続されている互換カードすべてに Neat をインストールします。 | 0.6.0 以降 |
| Neat Library (core) | インターネットに接続せずに core パッケージをインストールします。 | 0.6.0 以降 |

## 開始前の準備

- インターネットに接続されたコンピューターに `sima-cli` をインストールしてください。[環境のインストール](/getting-started/dev-environment/install-the-environment/)を参照してください。
- システムと互換性のあるパッケージバージョンを選択してください。[互換性](/getting-started/compatibility/)を参照してください。
- 対象ホストのオペレーティングシステムとプロセッサーアーキテクチャを確認してください。
- 転送用の USB ドライブまたは内部ファイル共有を準備してください。

ダウンロード用コンピューターと対象ホストのプロセッサーアーキテクチャは異なっていてもかまいません。
**対象ホスト**用のパッケージを選択してください。
スクリプト、メタデータ、チェックサム、依存関係を含むすべてのパッケージファイルを一緒に保管してください。

## Neat SDK をインストールする {#download-the-sdk-offline-package}

### 1. パッケージをダウンロードする

インターネットに接続されたコンピューターで、いずれかのコマンドを実行してください。
以下の例では SDK 2.1.3.0 を使用します。

`amd64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

`arm64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

### 2. パッケージをインストールする

1. ダウンロードしたディレクトリ全体を対象ホストにコピーしてください。
2. そのディレクトリでターミナルを開いてください。
3. インストーラーを実行してください：

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

## Model Compiler をインストールする {#download-the-model-compiler-offline-package}

### 1. パッケージをダウンロードする

[Model Compiler の互換性要件](/getting-started/compatibility/#model-compiler)を確認してください。
インターネットに接続されたコンピューターで、いずれかのコマンドを実行してください。
以下の例では Model Compiler 2.1.3 を使用します。

`amd64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

`arm64` の対象ホストの場合：

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
</ShellCommand>

### 2. パッケージをインストールする

インストール先を 1 つ選択してください。

**Neat SDK 内：**

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

## PCIe 用の Neat をインストールする

PCIe パッケージは、ホストと接続されている互換カードすべてに Neat をインストールします。
このインストールでは、対象システムに Docker やインターネット接続は必要ありません。
ホストから各カードへの PCIe ネットワーク接続が動作している必要があります。

### 1. ホストとカードを準備する

Neat インストーラーを実行する前に、次の作業を完了してください：

- 互換性のある Platform ホストドライバーをインストールしてください。
- 各カードに対応する Platform 3.0 ファームウェアをインストールしてください。
- 各カードの PCIe インターフェースで検出を設定してください。
- 各カードに一意の PCIe IPv4 アドレスを割り当ててください。
- 各カードの SSH アクセスと信頼できる SSH ホストキーを設定してください。
- SSH アカウントに、各カードで `sudo` を使用する権限を付与してください。

Neat インストーラーは、Platform ドライバーのインストールや Platform ファームウェアの変更を行いません。
Platform リリースに付属するシステムセットアップ手順を使用してください。

### 2. パッケージをダウンロードする

対象ホスト用のバージョン 0.6.0 以降を選択してください。
パッケージは、`amd64` または `arm64` プロセッサーを搭載した Ubuntu 22.04 および 24.04 ホストに対応しています。

この例では、インターネットに接続されたコンピューターでバージョン 0.6.0 をダウンロードします。
Ubuntu 24.04 の `amd64` 対象ホストの場合は、次を実行してください：

<ShellCommand prompt="host">
sima-cli neat install 'core/pciehost/ubuntu24/amd64@v0.6.0' -t offline -d ./pcie-offline
</ShellCommand>

Ubuntu 22.04 の場合は、`ubuntu24` を `ubuntu22` に置き換えてください。
`arm64` ホストの場合は、`amd64` を `arm64` に置き換えてください。
このコマンドはパッケージをダウンロードして展開します。Neat はインストールしません。

### 3. システムを確認する

1. `pcie-offline` ディレクトリ全体を対象ホストにコピーしてください。スペースを含まないパスを使用してください。
2. `install.py` を含む、展開済みの `neat-offline` ディレクトリを開いてください。
3. 必要なシステムツールと SSH 設定について、付属の `INSTALL.md` を読んでください。
4. チェックを実行してください：

<ShellCommand prompt="host">
sudo python3 -B install.py --preflight-only
</ShellCommand>

インストーラーは接続されているすべてのカードを自動的に検出します。カードの IP アドレスは指定しないでください。
チェックでは一時ファイルを作成しますが、パッケージはインストールしません。
インストール前に、報告されたエラーを修正してください。

### 4. インストールして結果を確認する

同じディレクトリから次のコマンドを実行してください：

<ShellCommand prompt="host">
sudo python3 -B install.py
</ShellCommand>

インストーラーは、パッケージをインストールする前にすべてのカードを確認します。
最初にホストをインストールして検証し、次に各カードを順番に処理します。
最終サマリーで、ホストとすべてのカードのインストールおよび検証が成功したことを確認してください。
次に、アプリケーションの受け入れテストを実行してください。

インストールに失敗した場合は、表示されたパスの診断ログを読んでください。
インストール中にカードへの接続が失敗した場合は、再試行前にカードのインストールプロセスが停止したことを確認してください。
失敗しても、インストーラーは以前のインストール状態を自動的には復元しません。

## Neat Library (core) をインストールする

Neat Library (core) オフラインパッケージは、バージョン 0.6.0 以降で利用できます。
Platform 3.0 以降を使用する Modalix DevKit には、このパッケージを使用してください。

### 1. パッケージをダウンロードする

インターネットに接続されたコンピューターで、次のコマンドを実行してください：

<ShellCommand prompt="host">
sima-cli neat install core@v0.6.0 -t offline
</ShellCommand>

### 2. パッケージをインストールする

1. ダウンロードしたディレクトリ全体を対象ホストにコピーしてください。
2. そのディレクトリでターミナルを開いてください。
3. パッケージに付属するインストール手順を使用してください。

## パッケージを内部に保管する {#hosting-packages-internally}

承認済みパッケージを内部ファイルサーバーに保管できます。
ダウンロードした各ディレクトリは、内容をすべて保持し、他のパッケージバージョンと分けて保管してください。
インストーラーを実行する前に、ディレクトリ全体を対象ホストに転送してください。
