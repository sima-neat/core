---
title: "DevKit Sync"
description: "Modalix DevKit と SDK ワークスペースを共有し、ハードウェア上でコマンドを実行します。"
sidebar_position: 3
---

:::tip 参考としてのみ使用
SDK のインストール中に、DevKit Sync の設定はオプションであり、インストーラーがすでにその設定を促します。このページを、DevKit Sync の動作方法、または後で SDK のペアリングを別の DevKit に変更したい場合の参考として使用してください。
:::

DevKit Sync は、Neat 開発環境（Neat SDK とも呼ばれる）と、同じネットワーク上の Modalix DevKit を接続します。ホスト、Neat SDK コンテナ、および DevKit にまたがる、共有ワークスペースを1つ提供し、ハードウェア上で SDK で作成したコマンドを実行するための `dk` ヘルパーを提供します。

![ホスト、コンテナ、DevKit 間のワークスペース対応関係](@site/../docs/images/elxr-sdk-workspaces.svg)

同じワークスペースが、Neat SDKコンテナとDevKitに`/workspace`としてマウントされるため、ビルドされたアーティファクト、ログ、トレース、およびモデルファイルは、それぞれの環境から確認できます。

## DevKit Sync の設定

インストール時にDevKitのペアリングをスキップした場合、または後でペアリングを変更する必要がある場合は、ホストから次のセットアップコマンドを実行してください。

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

セットアップ中：

- 複数の SDK イメージが存在する場合、インストールされている `sdk:v2.1-latest` イメージを選択します。
- 別のパスが必要でない限り、デフォルトの `/workspace` DevKit マウントパスを使用します。
- ホストで NFS サーバーを設定する際に、ホストの管理者パスワードを入力します。
- プロンプトが表示されたら、DevKit ユーザーの認証情報を入力します。デフォルトのユーザーは `sima` で、デフォルトのパスワードは `edgeai` です。
- SDK 2.1.2.2 以降では、ホストと DevKit の間で NFS を構成できない場合に、セットアップを SSH 経由の rsync に切り替えることができます。

セットアップが成功すると、次のような出力が表示されるはずです。

```text
============================================================
  DevKit Connected
============================================================
  DevKit target : sima@192.168.91.221:22
  Mounted path  : /workspace
  Host export   : 192.168.74.48:/Users/joey/workspace

  You can now run DevKit binaries from this SDK shell:
    dk /workspace/<path-to-arm64-binary> [args...]
============================================================
```

SDKシェル内から、ペアリングされたDevKitと、現在アクティブなワークスペースの同期方法を確認するために、`dk status`を使用します。

<ShellCommand prompt="username@neat-sdk-latest">
dk status
</ShellCommand>

## SDKからペアリングを更新します

Neat SDK がすでにインストールされており、別の DevKit とペアリングする必要がある場合、または DevKit の IP アドレスが変更された後にペアリングを更新する必要がある場合は、この手順を使用してください。

Neat SDK コンテナ内で、次のコマンドを実行します。

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh {devkit-ip}
</ShellCommand>

使用する DevKit の IP アドレスで、`{devkit-ip}` を置き換えてください。

例：

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh 192.168.91.221
</ShellCommand>

## DevKit Sync を使用せずに SDK を設定する

DevKit が Neat SDK のホストからアクセスできない場合でも、ペアリングを行わずに SDK ワークスペースを構成できます。

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

Neat SDKコンテナ内でバイナリをビルドすることは引き続き可能ですが、テストのためにそれらをDevKitに手動で転送する必要があります。DevKitが、互換性のあるNeat Libraryバージョンを実行していることを確認してください。

## DevKit Sync を使用したファイル共有

DevKit Sync は、3つの環境を接続します。

1. ホスト
2. Neat SDK コンテナ
3. DevKit

`sima-cli sdk setup --devkit {devkit-ip}` は、NFSを設定し、3つの環境すべてで同じワークスペースが利用できるようにします。

- ホストのワークスペースフォルダーは、ホストの NFS を介してエクスポートされます。
- このフォルダーは、Neat SDK コンテナーに `/workspace` としてマウントされます。
- 同様のコンテンツが、NFS を介して DevKit 上の `/workspace` に表示されます。
- マウントされたフォルダーの名前はデフォルトで `/workspace` に設定されており、セットアップ中に変更できます。

この構成により、ビルドされたアーティファクトに対する直接的なワークフローが提供されます。

- Neat SDK で生成されたアーティファクトは、個別のデプロイ手順なしに、DevKit 上ですぐに確認できます。
- エージェントは、DevKit 上でアプリケーションが実行されている間に生成されたログ、出力、トレース、およびその他の中間ファイルにアクセスできます。
- 開発者とエージェントは、同じワークスペースのコンテキストから同じファイルを確認できます。

Insight を使用すると、Web ブラウザからワークスペースを表示できます。一部の SiMa.ai に固有のモデルアーカイブ（例：`*.tar.gz` 形式のモデルアーティファクト）は、より簡単に検査できるように自動的に最適化されます。

## Rsync のフォールバック

SDK 2.1.2.2 以降では、NFS の設定が正常に完了しなかった場合に、DevKit Sync は、SSH を介した rsync を代替手段として使用できます。これは、DevKit への SSH 接続は可能だが、ホストの NFS エクスポートを DevKit からマウントできないネットワークやホストで役立ちます。

rsyncによるフォールバックが有効になっている場合：

- ホストとSDKコンテナは、引き続きローカルの`/workspace`ディレクトリを使用します。
- DevKitは、通常、同期されたリモートワークスペースである`/workspace-rsync`を使用します。
- `dk status`は、`Sync method : rsync`を報告し、ローカルとリモートのワークスペースのパスを表示します。
- `dk <file> [args...]`は、コマンドをリモートで実行する前に、SDKワークスペースからDevKitのrsyncワークスペースへのパスをマッピングします。
- `dk`がファイルを実行する前に、そのファイルを含む最上位のワークスペースフォルダーを自動的に同期します。たとえば、`dk apps/demo.py`は、DevKit側のコピーを実行する前に、`/workspace/apps`フォルダーを同期します。

現在のペアリングと同期方法を確認してください。

<ShellCommand prompt="username@neat-sdk-latest">
dk status
</ShellCommand>

現在のワークスペースの範囲を手動で同期します。

<ShellCommand prompt="username@neat-sdk-latest">
dk sync
</ShellCommand>

特定のファイルまたはフォルダー範囲を同期します。

<ShellCommand prompt="username@neat-sdk-latest">
dk sync /workspace/apps
</ShellCommand>

ワークスペース全体を同期します。

<ShellCommand prompt="username@neat-sdk-latest">
dk sync --all
</ShellCommand>

rsyncによるフォールバックが有効になっている場合、1つの`dk`コマンドに必要なファイルを、同じ最上位のワークスペースフォルダーにまとめて保存します。コマンドが`/workspace/apps`から起動された場合、`/workspace/models`を参照する引数は、自動同期の範囲外となるため、`dk sync
/workspace/models`を使用して別途同期するか、必要なファイルが同じ最上位フォルダーに保存されるようにプロジェクトを構成する必要があります。

## DevKit で dk を実行

SDKには、ARM64実行ファイルをSDKシェル内からペアリングされたDevKit上で実行するための、`dk`ヘルパー（別名：`devkit-run`）が含まれています。

`dk` を実行すると、SDK はペアになっている DevKit 上でコマンドを実行し、
コンテナーからのファイル引数が DevKit 上で正しく解決されるようにパスを変換します。

<ShellCommand prompt="username@neat-sdk-latest">
dk <file> [args...]
</ShellCommand>

SDKワークスペースでC++アプリケーションをコンパイルした後、生成されたARM64実行ファイルをDevKit上で実行します。

<ShellCommand prompt="username@neat-sdk-latest">
dk build/sima_neat_hello
</ShellCommand>

SDK ワークスペースに Python スクリプトを作成またはコピーした後、ペアになった DevKit で実行します。

<ShellCommand prompt="username@neat-sdk-latest">
dk hello_neat.py
</ShellCommand>

Pythonスクリプトの場合、`dk` は、ペアになった DevKit 上でスクリプトを実行し、DevKit の PyNeat ランタイム環境を使用します。SDKは、統合されたワークスペースおよびオーケストレーション環境として引き続き役立ちますが、Pythonのみを使用するワークフローでは、C++のクロスコンパイルツールチェーンは必要ありません。

:::note `dk` はどこから来たのか
`dk` は、SDKコンテナー内の `~/devkit-sync.rc` で定義されたシェル関数です。
シェルは、`~/.bashrc` を介してこの関数を読み込むため、インタラクティブなセッションで使用できます。
:::

## 次のステップ

ライブラリまたはランタイム自体をインストールまたは更新するには、[Neat Library](/getting-started/neat-library/) を使用してください。
