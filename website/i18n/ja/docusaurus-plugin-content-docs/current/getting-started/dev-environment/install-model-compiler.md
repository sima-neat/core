---
title: "Model Compiler をインストールしてください。"
description: "Model Compiler を、Neat SDK にインストールするか、対応するスタンドアロンのホストにインストールしてください。"
sidebar_position: 5
---

:::tip Model Compiler を個別にインストールする場合のみ、ここから開始してください。
Model Compiler のインストールは、SDK のインストール時に任意で行います。このページは、そのプロンプトをスキップした場合、またはより新しい互換性のある Model Compiler をインストールしたい場合、あるいはサポートされているホスト上で SDK 以外の場所に Model Compiler をインストールする必要がある場合にのみ使用してください。
:::

Model Compiler は、ONNXモデルを量子化およびコンパイルし、SiMa.ai のMLA上で実行できるようにします。GenAIモデルを含むモデルを自分でコンパイルまたは量子化する場合は、**必須**であり、事前にコンパイルされたモデルパッケージのみを使用する場合は**オプション**です。

SDK のインストール/設定中に、`sima-cli` は、対応するモデルコンパイラを Neat SDK 内の拡張機能としてインストールするように促します。後でインストールすることもできます。インストール先は、Neat SDK コンテナ内、またはサポートされている Ubuntu ホストにスタンドアロンでインストールする方法のいずれかです。サポートされているバージョン組み合わせとスタンドアロンホストの要件については、[互換性](/getting-started/compatibility/#model-compiler) を参照してください。

## SDK 内にインストール

SDKのセットアップ中にModel Compilerをスキップした場合、後でNeat SDK内からインストールしてください。お使いのNeat SDKコンテナのアーキテクチャに合ったコマンドを実行します。確認するには、SDKシェル内で`uname -m`を実行します。`aarch64`の場合は`arm64`コマンドを使用し、`x86_64`の場合は`amd64`コマンドを使用します。

`amd64` Neat SDK コンテナの場合：

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli neat install model-compiler/amd64@v2.1.3
</ShellCommand>

`arm64` Neat SDK コンテナの場合：

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli neat install model-compiler/arm64@v2.1.3
</ShellCommand>

インストール後、Neat SDK のシェルからコンパイラ環境を起動してください。

<ShellCommand prompt="username@neat-sdk-latest">
activate-model-compiler
</ShellCommand>

デフォルトに戻すには Neat SDK シェル、実行：

<ShellCommand prompt="username@neat-sdk-latest">
deactivate-model-compiler
</ShellCommand>

## 単独のホストにインストール

スタンドアロンでのインストールは、[互換性](/getting-started/compatibility/#model-compiler) に記載されているホスト環境でのみサポートされます。サポートされているホスト環境から、対応する `sima-cli neat install` コマンドを実行してください。ホストのアーキテクチャを確認するには、`uname -m` を実行します。`x86_64` は `amd64` コマンドを使用し、`aarch64` は `arm64` コマンドを使用します。

`amd64` のホスト上で動作する Model Compiler 2.1.3 について：

<ShellCommand prompt="user-host-machine">
sima-cli neat install model-compiler/amd64@v2.1.3
</ShellCommand>

`arm64` のホスト上で動作する Model Compiler 2.1.3 について：

<ShellCommand prompt="user-host-machine">
sima-cli neat install model-compiler/arm64@v2.1.3
</ShellCommand>

`amd64` のホスト上で動作する Model Compiler 2.0.0 について：

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.0.0 tools/model-compiler/amd64
</ShellCommand>

## 次のステップ

引き続き、[モデルを構築する](/compile-a-model/) を実行してください。
