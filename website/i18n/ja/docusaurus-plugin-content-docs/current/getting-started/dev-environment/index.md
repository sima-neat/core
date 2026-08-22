---
title: "Neat SDK"
description: "迅速かつ効率的なエージェント対応の Neat アプリケーション開発のために、Neat SDK をセットアップしてください。"
sidebar_position: 1
---

Neat 開発環境（Neat SDK と呼ばれる）は、大規模な Neat アプリケーションを構築し、Modalix DevKit 上で検証するための、推奨されるホスト側のワークスペースです。これには、ビルドツール、モデルツール、ハードウェア接続、およびエージェントが使用できるソースコードが含まれており、これらすべてがコンテナ化されたワークフローに統合されています。

SDKは、以下の3つの場所を接続します。

- **ホストマシン:** SDKコンテナをインストールして起動する場所。
- **SDKコンテナ:** アプリケーションをビルドし、モデルをコンパイルし、エージェントツールを使用し、共有ファイルを検査する場所。
- **Modalix DevKit:** コンパイルされたモデルのアーティファクトとNeatアプリケーションがハードウェア上で実行される場所。

DevKit Sync は、それらの場所を共有の `/workspace` で接続するため、ビルド出力、ログ、モデルのアーティファクト、およびアプリケーションファイルが、ホスト、SDKコンテナ、および DevKit から、手動でのコピー操作なしに参照できるようになります。この共有ワークスペースは、SDKワークフローの中心です。

<div class="overview-section-label">ここから始めましょう。</div>

まず、SDK のインストールから始めます。設定を変更したり、後で Model Compiler を追加したり、DevKit Sync の動作を理解したり、または制限されたネットワークで使用するためのオフラインパッケージを準備する必要がある場合は、他の SDK のトピックを参照してください。

:::tip SDKのみを使用した、問題なく動作する状態
SDKをインストールし、DevKitとペアリングしていない場合は、次の2つのステップのみを実行します。

[環境をインストールします。](/getting-started/dev-environment/install-the-environment/)、
次に[モデルを構築する](/compile-a-model/)を実行します。モデルのコンパイルは、SDK内で完全に実行されます。SDKを構成し、DevKit Syncを実行し、Model Compilerをインストールします（これはセットアップ中に提供されます）。また、Neat LibraryとPyNeatのページはオプションであり、必要に応じて、またはDevKitとペアリングした後にアクセスしてください。
:::

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>SDK に関する話題</h2>
    <p>SDKをインストールし、必要に応じて設定を変更したり、Model Compilerをインストールしたり、ペアになったDevKitを使用したりする場合は、オプションの構成項目を使用してください。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/install-the-environment/"><strong>環境をインストールします。</strong><span>お使いの DevKit ソフトウェアのバージョンに合った SDK パッケージをインストールして設定してください。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/devkit-sync/"><strong>DevKit Sync</strong><span>ワークスペースの共有、ペアリングの更新、rsyncによる代替処理について理解する。 <code>dk</code> コマンドの実行。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/offline-installation/"><strong>オフラインでのインストール</strong><span>ネットワーク環境が制限された環境向けに、SDKとModel Compilerパッケージをダウンロードしてください。</span></a></li>
    </ul>
  </section>
</div>

インストール後に、ワークスペースの場所や DevKit とのペアリングなど、SDK の設定を変更するには、[SDK を設定します。](/getting-started/dev-environment/configure-sdk/) を参照してください。

Model Compiler は、SDK のセットアップ中に提供されます。後でインストールするには、特定のバージョンを固定するか、スタンドアロンのホストを使用してください。詳細については、[Model Compiler をインストールしてください。](/getting-started/dev-environment/install-model-compiler/) を参照してください。

パッケージを直接ダウンロードできないホストについては、[オフラインでのインストール](/getting-started/dev-environment/offline-installation/) を参照してください。

## 含まれるもの

- **クロスコンパイル環境:** ホスト上の Linux コンテナーから、C++ Neat アプリケーションを Modalix 向けにビルドします。
- **DevKit Sync:** SDK と Modalix DevKit をペアにし、両方の場所で同じワークスペースを共有できるようにします。
- **モデルツール:** SDK に対応する Model Compiler をインストールします。ONNX または GenAI モデルを自分でコンパイルまたは量子化するには必須であり、事前にコンパイルされたモデルパッケージのみを使用する場合はオプションです。
- **Insight:** ブラウザからワークスペースファイル、メディアソース、ストリーム配信、およびランタイムの動作を検査します。
- **エージェント対応コンテキスト:** 現在の Neat ソースリファレンスとサンプルとともに、バンドルされた Codex および Claude スキルを使用します。

## ホストの要件

SDK をインストールする前に、ホストマシンが以下の最小要件を満たしていることを確認してください。
SDK をサポートされているすべてのホストにインストールするには、管理者 (`sudo`) 権限が必要です。これは、オプションの DevKit ネットワークのためだけでなく、`sima-cli`、Docker Engine、SDK イメージ、および NFS パッケージのインストールには、すべてより高い権限が必要となるためです。

| ホストOS | CPU | RAM | 空きディスク容量 | 管理者権限 / sudo |
|---|---|---|---|---|
| Ubuntu 22.04 / 24.04 (`x86_64` または `arm64`) | 4コア以上 | 16GB以上 | 100GB | SDKのインストール（`sima-cli`、Docker、SDKイメージ）、NFSのインストール/設定、および共有ネットワーク/ファイアウォールの設定に必要。`sudo`を使用。 |
| Windows 11 (WSL経由) (`x86_64`) | 4コア以上 | 16GB以上 | 100GB | WSLでのSDKインストール（Docker、`sima-cli`）、WSLのネットワーク設定、およびNFSファイアウォールルールに必要。管理者権限が必要。 |
| macOS 15.5+ Apple Silicon (`arm64`) | 4コア以上 | 16GB以上 | 100GB | SDKのインストール（Homebrew、Colima、`sima-cli`）、フルディスクアクセス（`nfsd`）、およびインターネット共有に必要。管理者権限が必要。 |

:::note 生成AIモデルのコンパイルには、さらに多くのリソースが必要です。
LLiMa を使用して GenAI モデルをコンパイルする場合、基本的な SDK よりもはるかに多くのリソースが必要になります。推奨される RAM は 128 GB、理想的なディスク容量は 512 GB であり、より多くのコア数を使用すると効果的です。完全な要件については、[GenAI の設定](/genai-llima/setup/) を参照してください。
:::

## 対応プラットフォーム

| プラットフォーム | アーキテクチャ | SDK | Model Compiler |
|---|---|---|---|
| Docker Engine を介した Ubuntu 22.04 および 24.04 | `x86_64` | はい | はい |
| WSL および Docker Engine を介した Windows 11 | `x86_64` | はい | はい |
| Docker Engine を介した Ubuntu 22.04 および 24.04 | `arm64` | はい | Model Compiler 2.1.2 以降 |
| Colima を介した macOS 15.5 以降 | `arm64` | はい | Model Compiler 2.1.2 以降。Neat SDK の中にインストールしてください。 |

:::note アーキテクチャ名
`arm64`と`aarch64`は、同じ64ビットArmアーキテクチャです。macOSでは`arm64`として、Linuxでは`aarch64`として報告されます。同様に、`x86_64`と`amd64`も同じアーキテクチャです。ホスト（またはSDK内）で`uname -m`を実行して、どちらのアーキテクチャを使用しているかを確認してください。Model Compilerのインストールコマンドでは、`arm64`と`amd64`を使用します。詳細は、[Model Compiler をインストールしてください。](/getting-started/dev-environment/install-model-compiler/)を参照してください。
:::

:::note 特定のバージョンをインストールする
標準インストールでは、現在サポートされているデフォルト設定が適用されます。`release-2.1`チャンネルは、[環境をインストールします。](/getting-started/dev-environment/install-the-environment/)ページで常に最新の2.1パッチリリースを追跡します。特定のSDK、Neat Library、またはModel Compilerバージョンを固定するには、[互換性ガイド](/getting-started/compatibility/)を参照してください。
:::

## SDKに含まれるツール

SDK は、ペアになった DevKit 用のアプリケーションを開発する際に、[Neat Library](/getting-started/neat-library/) をインストールおよび更新するのに最適な場所です。

ターミナル、VS Code、または Neat Insight を備えたブラウザから SDK にアクセスするには、[環境をインストールします。](/getting-started/dev-environment/install-the-environment/#access-the-sdk) を参照してください。

SDK のセットアップ中に、`sima-cli` が、対応する Model Compiler を自動的にインストールするように促します。モデルを自分でコンパイルまたは量子化する場合は、インストールしてください。事前にコンパイルされたモデルパッケージのみを使用する場合は、インストールをスキップできます。コンパイラのセットアップと使用方法については、[モデルを構築する](/compile-a-model/) を参照してください。

## 次のステップ

まず、[環境をインストールします。](/getting-started/dev-environment/install-the-environment/) から始めましょう。
