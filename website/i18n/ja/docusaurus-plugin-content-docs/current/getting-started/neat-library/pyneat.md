---
title: "PyNeat をインストールしてください。"
description: "カスタムの Python 仮想環境に、PyNeat のホイールパッケージをインストールします。"
sidebar_position: 4
---

:::note DevKit のみ — SDK のインストール時にはスキップします。
これらの手順は、PyNeat を DevKit (またはランタイムとして機能するホスト) にセットアップします。Neat SDK の中で作業している場合は、このページをスキップしてください。
:::

:::tip PyNeat は、Neat Library とともにすでにインストールされています。
PyNeat は、Neat Library と一緒にパッケージ化されており、Neat Library をインストールすると、自動的にインストールされます。

デフォルトでは、`~/pyneat` に仮想環境としてインストールされます。PyNeat を、DevKit 上の別の venv 環境や conda 環境など、カスタムの仮想環境にインストールしたい場合を除き、このページはスキップできます。
:::

以下の手順を DevKit で実行してください。この手順では、ランタイム `.deb` パッケージのインストールや更新は行われません。したがって、対応する Neat Library ランタイムがすでにインストールされている環境で実行してください。

## ホイールをダウンロード

<ShellCommand prompt="devkit">
sima-cli neat install core -t pyneat
</ShellCommand>

特定の Neat Library リリースのホイールをダウンロードするには、バージョン番号を含めてください。

特定のバージョンをインストールするには：

<ShellCommand prompt="devkit">
sima-cli neat install core@v0.4.0 -t pyneat
</ShellCommand>

## Python環境の作成

環境に合わせた仮想環境を作成し、`python3` を使用してアクティブにします。

<ShellCommand prompt="devkit">
python3 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## ホイールをインストールします

<ShellCommand prompt="devkit">
pip install ./pyneat-*.whl
</ShellCommand>

サポートされているNeat Library、SDK、およびDevKitのソフトウェアの組み合わせについては、[互換性ガイド](/getting-started/compatibility/)を参照してください。

## 次のステップ

インストールされた環境を確認するために、[Neat コマンドラインインターフェース](/getting-started/neat-library/neat-cli/) を実行し続けてください。
