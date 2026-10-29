---
title: "SDK のインストール"
description: "Neat SDKコンテナをインストールしてセットアップする"
sidebar_position: 2
---

最新のNeat SDKをインストールします。Neat SDKはホストマシン上でコンテナとして動作します。

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

現在のNeat SDK %%release.sdk_series%%リリースチャネルをインストールします。

<ShellCommand prompt="host">
sima-cli neat install sdk@%%release.sdk_channel%%
</ShellCommand>

初回インストールではNeat SDKコンテナイメージをダウンロードするため、数分かかることがあります。ダウンロード後、インストーラーがSDKのセットアップを開始し、Modalix DevKitとペアリングするか、対応するModel CompilerをSDK内にインストールするかを確認します。

DevKitとのペアリングを選んだ場合は、表示されたプロンプトにDevKitのIPアドレスを入力します。セットアップ処理はSDKワークスペースを構成し、SDKコンテナを起動してDevKit Syncを設定します。ペアリングを省略してもSDKワークスペースは作成され、後からペアリングできます。

`%%release.sdk_channel%%`パッケージは、%%release.sdk_series%%系の最新Neat SDKパッチリリースを追跡します。現在のリリースはNeat SDK %%release.sdk%%で、DevKitソフトウェア%%release.platform%%と互換性があります。

セットアップ中、`sima-cli`は対応するModel CompilerをSDK内にインストールするか確認します。モデルを自分でコンパイルまたは量子化する場合は承認してください。別のバージョンを選択する必要はありません。コンパイル済みモデルパッケージだけを実行する場合は省略できます。後からインストールする場合、特定のパッチを固定する場合、またはスタンドアロンホストを使用する場合は、[Model Compilerのインストール](/getting-started/dev-environment/install-model-compiler/)と[互換性ガイド](/getting-started/compatibility/)を参照してください。

### 特定の SDK バージョンをインストールする {#install-a-specific-sdk-version}

SDK 2.1.3.0 以降では、公開済みの `v<version>` タグを使用して、以前のパッチリリースを含む特定のバージョンをインストールできます。たとえば、ホストで次のコマンドを実行します。

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0
</ShellCommand>

`v` プレフィックスは必須です。特定のバージョンタグに `release-<full-version>` を使用しないでください。`release-<series>` チャネルは、そのシリーズの最新の公開リリースを追跡します。同じ SDK パッケージを選択する場合、どちらも同じインストールおよび SDK セットアップ手順を実行します。

公開済みのタグは [SDK リリース](https://github.com/sima-neat/sdk/releases) で確認し、[互換性ガイド](/getting-started/compatibility/) に従って DevKit ソフトウェアと互換性のあるバージョンを選択してください。

:::note 従来の2段階インストール
SDK 2.0.0、2.1.2.0、2.1.2.1 では、イメージの取得とセットアップに個別のコマンドが必要です。これらのリリースに限り、[2段階SDKインストール](/reference/two-step-sdk-installation/)を参照してください。
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

`neat` の出力の **Web Access** セクションには、Insight とブラウザ版 VS Code のローカルおよびリモート URL が表示されます。SDK ホストでブラウザを使用する場合は、ローカルの `127.0.0.1` URL を開いてください。別のコンピューターからは、`NFS_SERVER_HOST_IP` に基づくリモート URL を使用します。VS Code の URL は設定済みワークスペースを開きます。インストールされているサーバーに対して `neat` が表示する認証情報を使用してください。[SDK の Web アクセス](/reference/environment-variables/)と [Insight](/tools/insight/)を参照してください。

### VS Codeを使用する

SDK シェル内で `neat` を実行し、SDK ホストのブラウザで **VS Code (local)** の URL を開いてください。別のコンピューターからは **VS Code (remote)** を使用します。code-server を含むイメージでは、`neat` が **VS Code password** も表示します。

```text
VS Code (local)    https://127.0.0.1:10000/?folder=/workspace
VS Code password   <password shown by neat>
```

code-server のサインイン画面で、表示されたパスワードを入力してください。パスワードは URL には含まれません。

従来の OpenVSCode イメージでは、代わりにトークン付き URL を使用します。`neat` が表示する URL を開いてください。SDK には、ブラウザ版 Code UI 用の Codex と Claude Code の拡張機能がプリインストールされています。

ブラウザ版ではなくネイティブVS Codeも利用できます。[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)を使用して、VS CodeをSDKコンテナへ接続してください。

コンパイル済みモデルなどのアセットをSDKが取得できるよう、SDKコンテナ内で`sima-cli login`を一度実行します。

## アップグレード

現在のSDKパッケージを再インストールまたはアップグレードするには、ホストから上記のインストールコマンドを再実行します。

<ShellCommand prompt="host">
sima-cli neat install sdk@%%release.sdk_channel%%
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
