---
title: "チュートリアルの設定"
description: "実行環境を選択し、チュートリアルをダウンロードし、モデルアーカイブを準備してください。"
sidebar_position: 2
slug: /tutorials/before-you-run
---

# チュートリアルのセットアップ

チュートリアルを開始する前に、このセットアップを一度完了させてください。チュートリアルのカテゴリに合った環境を選択してください。Neat LibraryとPCIeバンドルは互換性がないことに注意してください。

## 1. 環境の選択

| チュートリアルのカテゴリ | 実行環境 | Python環境 |
|---|---|---|
| モデルと推論、グラフとパイプライン、カメラとストリーミング、GenAI | Modalix DevKit、またはチュートリアルで指定された環境 | `~/pyneat` |
| PCIeコプロセッシング | Modalix PCIeカードに接続されたホスト | `~/pyneatpcie` |

PCIeチュートリアルは、SDKコンテナ内またはカード上で直接実行するのではなく、ホスト上で実行されます。

## 2. Neat Libraryチュートリアルのセットアップ

[Neat Library がインストールされました。](/getting-started/neat-library/install-or-update/)がインストールされていることを確認し、チュートリアルのバンドルを配置したいディレクトリから次のコマンドを実行します。

<ShellCommand prompt="sdk|devkit">
sima-cli neat install core -t extras
cd sima-neat-*-Linux-extras
</ShellCommand>

DevKit 上で直接実行される Python のチュートリアルでは、PyNeat を有効にして、インポートが正常に機能することを確認してください。

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 -c "import pyneat; print('pyneat ready')"
</ShellCommand>

## 3. PCIeチュートリアルの設定

まず、[PCIeホストパッケージをインストールし、検証します。](/getting-started/neat-library/pcie-host/) を実行します。
次に、ホストで実行されているUbuntuバージョン用のチュートリアルバンドルをダウンロードします。
バンドルを配置したいディレクトリからコマンドを実行します。

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04：**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

PCIe を確認します。 PyNeat:

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. モデルアーカイブを準備する

チュートリアルで指定されたモデルをダウンロードするには、Model Zoo を使用します。例：

<ShellCommand prompt="sdk|devkit|pcie-host">
sima-cli modelzoo get resnet_50
sima-cli modelzoo get yolo_v8s
</ShellCommand>

Neat Libraryのチュートリアルでは、`--model`を受け入れるため、ダウンロードしたアーカイブを直接渡すことができます。PCIeチュートリアルでは、PCIeエクストラディレクトリのルートにある固定ファイル名を使用します。

| PCIeチュートリアル | 必要なモデルファイル |
|---|---|
| PCIe経由で最初のモデルを実行 | `yolo_v8s_mpk.tar.gz` |
| PCIe非同期推論を実行 | `yolo_v8s_mpk.tar.gz` |
| 複数のモデルを実行 | `resnet_50_mpk.tar.gz`と`yolo_v8s_mpk.tar.gz` |

Model Zooの出力名と場所は異なる場合があります。必要に応じて、アーカイブを必要な名前でPCIeエクストラディレクトリのルートにコピーしてください。

<ShellCommand prompt="pcie-host">
cp /absolute/path/to/downloaded-resnet-archive.tar.gz resnet_50_mpk.tar.gz
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

## 5. パスと期待される出力を確認する

抽出した追加ファイルのルートディレクトリから、チュートリアルのコマンドを実行します。そこに、ビルドヘルパー、事前にビルドされたC++プログラム、およびチュートリアルのソースコードが含まれていることを確認してください。

<ShellCommand prompt="sdk|pcie-host">
test -x build.sh
ls lib/*/tutorials/
ls share/*/tutorials/
</ShellCommand>

- 事前に作成されたC++プログラムは、`lib/<package>/tutorials/`にあります。
- C++とPythonのソースコードは、`share/<package>/tutorials/`にあります。
- `./build.sh --list-targets`には、再構築できるC++プログラムのリストが記載されています。
- 正常に完了したC++のチュートリアルは、`[OK]`で終了します。Pythonのチュートリアルは、`top1=...`、`completed=...`、または`detections=...`などの簡潔な結果を出力します。

チュートリアルでファイルが見つからないというエラーが表示された場合は、まず現在のディレクトリとモデルファイル名を確認してください。さらに詳しい情報が必要な場合は、[トラブルシューティング](/reference/troubleshooting/)を参照してください。
