---
title: "SDK を設定します。"
description: "インストール後、Neat SDK の作業スペース、DevKit のペアリング、または設定オプションを変更します。"
sidebar_position: 4
---

:::tip SDKの設定を変更する場合のみ、ここから操作を開始してください。
SDK のインストールコマンドは、すでに SDK イメージをダウンロードし、設定を行います。ワークスペースの場所、DevKit のペアリング、SDK 拡張機能、その他の設定オプションなど、SDK の設定を変更する必要がある場合は、このページを参照してください。
:::

既存の Neat SDKコンテナを構成する必要がある場合は、`sima-cli sdk setup` を直接実行してください。

## DevKit を使用したペアリング設定

ホストからアクセスできる状態の DevKit を使用し、DevKit のペアリングを追加または更新する場合は、このコマンドを使用してください。

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

以下のものを準備しておいてください。

- DevKit の IP アドレス
- NFS のインストールまたは設定が必要な場合、ホスト管理者のパスワード
- プロンプトが表示されたときに使用する DevKit のユーザー認証情報。デフォルトのユーザーは `sima` で、デフォルトのパスワードは `edgeai` です。

`--devkit` を使用すると、セットアップによって DevKit Sync が有効になります。これにより、ホストのワークスペースが NFS を介してエクスポートされ、デフォルトでは DevKit 上の `/workspace` としてマウントされます。

## DevKit を使用せずに設定する

SDKの設定を更新したいが、まだ DevKit にアクセスできない場合は、このコマンドを使用してください。

<ShellCommand prompt="host">
sima-cli sdk setup
</ShellCommand>

セットアップの際、`sima-cli` は、以下の操作を求める場合があります。

- ホストのワークスペースディレクトリを選択します。特別な理由がない限り、デフォルトのままにしてください。
- SDK拡張機能のディレクトリを選択します。
- Model Compiler をインストールするかどうかを選択します。

Model Compiler は、モデルを自分でコンパイルまたは量子化するために必要であり、あらかじめコンパイルされたモデルパッケージのみを使用する場合はオプションです。モデルをコンパイルまたは量子化する予定がある場合は、ここでインストールし、プロンプトが表示された場合は `sima-cli login` を完了してください。

## 次のステップ

ワークスペースの共有、ペアリングの更新、rsyncによる代替処理、および`dk`コマンドの詳細については、[DevKit Sync](/getting-started/dev-environment/devkit-sync/)をご覧ください。
