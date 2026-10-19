---
title: "開発環境のインストール"
description: "Neat SDKコンテナをインストールしてセットアップする"
sidebar_position: 2
---

最新のNeat開発環境（以下、Neat SDK）をインストールします。Neat SDKはホストマシン上でコンテナとして動作します。

## 前提条件

- インストールに必要な管理者権限または`sudo`権限を含め、ホストが[ホスト要件](/getting-started/dev-environment/#host-requirements)を満たしていることを確認します。
- [sima-cliインストールガイド](/tools/sima-cli/)に従い、ホストマシンに`sima-cli`をインストールします。
- プラットフォームごとのホスト設定を完了します。
  - [Ubuntuホストに関する注意事項](/reference/elxr-sdk-host-setup/ubuntu)
  - [Windowsホストに関する注意事項](/reference/elxr-sdk-host-setup/windows)
  - [macOSホストに関する注意事項](/reference/elxr-sdk-host-setup/macos)

後でDevKit Syncを使用するには、次の項目も必要です。

- Neat SDKと互換性のあるソフトウェアを実行するDevKit。詳しくは[互換性ガイド](/getting-started/compatibility/)を参照してください。
- NFSトラフィックが許可された同一ネットワーク上のホストマシンとDevKit。
- DevKitのIPアドレス。

## インストール

現在のNeat SDK 2.1リリースチャネルをインストールします。

<ShellCommand prompt="host">
sima-cli neat install sdk@release-2.1
</ShellCommand>

初回インストールではNeat SDKコンテナイメージをダウンロードするため、数分かかることがあります。ダウンロード後、インストーラーがSDKのセットアップを開始し、Modalix DevKitとペアリングするか、対応するModel CompilerをSDK内にインストールするかを確認します。

DevKitとのペアリングを選んだ場合は、表示されたプロンプトにDevKitのIPアドレスを入力します。セットアップ処理はSDKワークスペースを構成し、SDKコンテナを起動してDevKit Syncを設定します。ペアリングを省略してもSDKワークスペースは作成され、後からペアリングできます。

`release-2.1`パッケージは、2.1系の最新Neat SDKパッチリリースを追跡します。現在のリリースはNeat SDK 2.1.3.0で、DevKitソフトウェア2.1.3と互換性があります。

セットアップ中、`sima-cli`は対応するModel CompilerをSDK内にインストールするか確認します。モデルを自分でコンパイルまたは量子化する場合は承認してください。別のバージョンを選択する必要はありません。コンパイル済みモデルパッケージだけを実行する場合は省略できます。後からインストールする場合、特定のパッチを固定する場合、またはスタンドアロンホストを使用する場合は、[Model Compilerのインストール](/getting-started/dev-environment/install-model-compiler/)と[互換性ガイド](/getting-started/compatibility/)を参照してください。

:::note 以前のSDKリリースでは従来の2段階インストールを使用します
イメージの取得とセットアップに個別のコマンドが必要な以前のSDKリリースについては、[2段階SDKインストール](/reference/two-step-sdk-installation/)を参照してください。
:::

インストール後にSDK設定を変更するには、[SDKの構成](/getting-started/dev-environment/configure-sdk/)を参照してください。制限されたネットワーク環境では、[オフラインインストール](/getting-started/dev-environment/offline-installation/)を参照してください。

## SDKへのアクセス

セットアップが成功すると、ターミナル、Chromeブラウザ、またはVS CodeからSDKにアクセスできます。

### SDKシェルを使用する

次のコマンドでNeat SDKシェルを開きます。

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

### Chromeブラウザを使用する

Neat InsightはSDK内から提供され、ブラウザで開くことができます。SDKシェル内で次を実行します。

<ShellCommand prompt="sdk">
neat
</ShellCommand>

コマンド出力の**Web Access**セクションには、Insightとブラウザ版VS CodeのローカルURLおよびリモートURLが表示されます。ローカルアクセスとはSDKを実行しているマシンと同じマシンのブラウザで開くことで、リモートアクセスとは別のマシンから開くことです。ブラウザとSDKが同じホストにある場合は、ホストのネットワークIPが変わっても利用できるローカル`127.0.0.1` URLを推奨します。リモートアクセスには`NFS_SERVER_HOST_IP`から生成されたURLを使用します。VS CodeのURLには設定済みアクセストークンが含まれ、構成済みワークスペースが開きます。詳しくは[Insight](/tools/insight/)を参照してください。

### VS Codeを使用する

SDK Code UIを通じてブラウザからVS Codeを利用できます。SDKのインストール完了時に、`sima-cli`が次のような`codeUI` URLを表示します。

<ShellCommand prompt="host">
codeUI      | https://192.168.76.4:10000/?tkn=gA5CS...&folder=/workspace
</ShellCommand>

ブラウザでURLを開き、SDKワークスペース内で作業します。SDKには、ブラウザ版Code UI用のCodexとClaude Codeの拡張機能がプリインストールされています。

ブラウザ版ではなくネイティブVS Codeも利用できます。[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)を使用して、VS CodeをSDKコンテナへ接続してください。

コンパイル済みモデルなどのアセットをSDKが取得できるよう、SDKコンテナ内で`sima-cli login`を一度実行します。

## アップグレード

現在のSDKパッケージを再インストールまたはアップグレードするには、ホストから上記のインストールコマンドを再実行します。

<ShellCommand prompt="host">
sima-cli neat install sdk@release-2.1
</ShellCommand>

既存のNeat SDKコンテナ内にあるNeatライブラリを更新するには、コンテナシェルからNeat CLIを実行します。

<ShellCommand prompt="sdk">
neat update
</ShellCommand>

このコマンドは、現在のNeat SDKにインストールされているNeatライブラリのコンポーネントを更新します。コンテナレベルの変更が必要な場合に、コンテナイメージ全体のアップグレードを置き換えるものではありません。

後でNeat SDKコンテナを削除または再作成した場合は、新しいコンテナ内でもう一度`neat update`を実行してください。

## アンインストール

インストール済みSDKコンテナを削除するには、次を実行します。

<ShellCommand prompt="host">
sima-cli sdk remove
</ShellCommand>

## 次のステップ

- **SDK内でモデルをコンパイルしますか？** DevKitを必要とせずSDK内だけで実行できる[モデルのコンパイル](/compile-a-model/)に進みます。
- **DevKitをペアリングしますか？** SDKワークスペースを共有し、ハードウェアで`dk`コマンドを実行するために[DevKit Sync](/getting-started/dev-environment/devkit-sync/)を設定します。
