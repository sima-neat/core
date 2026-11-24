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
sima-cli neat install core@v%%release.core%% -t extras
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

Windowsでは、ホストパッケージにチュートリアルがすでに含まれています。[Windowsでアプリケーションを実行する](/getting-started/neat-library/pcie-host/#run-applications-on-windows)の説明に従って設定してください。

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04：**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

PCIe を確認します。 PyNeat:

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. モデルアーカイブを準備する

チュートリアルで使用するモデルを[Model Browser](/tools/model-browser/)からダウンロードし、各ZIPファイルからアーカイブを展開します。

<ShellCommand prompt="sdk|devkit|pcie-host">
sima-cli models download --id resnet_50 --variant w8a8 --output models --force
sima-cli models download --id yolov8s --variant w8a8 --output models --force
unzip -o models/resnet_50/w8a8/*.zip resnet_50_w8a8_mpk.tar.gz
unzip -o models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
</ShellCommand>

ダウンロードしたファイルが ZIP ファイルではなく `<id>_<variant>_mpk.tar.gz` ファイルの場合（Model Browser がモデルを ZIP ファイルとして提供する前に最後にビルドされたモデルの場合）は、展開せずに `models/<id>/<variant>` からそのファイルをコピーしてください。 `unzip` がインストールされていない場合は、代わりに `python3 -m zipfile -e models/yolov8s/w8a8/*.zip .` を実行し、`resnet_50` についても同様に実行してください。ZIP ファイル内のすべてのファイルが現在のディレクトリに展開されます。

Neat Libraryのチュートリアルでは、`--model`を受け入れるため、展開したアーカイブを直接渡すことができます。PCIeチュートリアルでは、PCIeエクストラディレクトリのルートにある固定ファイル名を使用します。

| PCIeチュートリアル | 必要なモデルファイル |
|---|---|
| PCIe経由で最初のモデルを実行 | `yolo_v8s_mpk.tar.gz` |
| PCIe非同期推論を実行 | `yolo_v8s_mpk.tar.gz` |
| 複数のモデルを実行 | `resnet_50_mpk.tar.gz`と`yolo_v8s_mpk.tar.gz` |

PCIeチュートリアルでは、PCIeエクストラディレクトリのルートでコマンドを実行し、アーカイブの名前を変更します。

<ShellCommand prompt="pcie-host">
mv resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

Windowsでは、チュートリアルのコピー先で次のPowerShellコマンドを実行します。

```powershell
sima-cli models download --id resnet_50 --variant w8a8 --output models --force
sima-cli models download --id yolov8s --variant w8a8 --output models --force
Expand-Archive models\resnet_50\w8a8\*.zip -DestinationPath models\resnet_50\w8a8 -Force
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8 -Force
Copy-Item models\resnet_50\w8a8\resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

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

Windowsでは、ホストパッケージは`build.sh`をインストールしません。[Windowsでアプリケーションを実行する](/getting-started/neat-library/pcie-host/#run-applications-on-windows)の説明に従ってチュートリアルを実行してください。

チュートリアルでファイルが見つからないというエラーが表示された場合は、まず現在のディレクトリとモデルファイル名を確認してください。さらに詳しい情報が必要な場合は、[トラブルシューティング](/reference/troubleshooting/)を参照してください。
