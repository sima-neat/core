# 018 ライブ RTSP ストリームを処理する

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 5-10 minutes |
| Model | None |
| Labels | rtsp, h264, h265, streaming, input-group, live-input |

## Concept

ライブ H.264 または H.265 RTSP ストリームを、`RtspDecodedInput` フラグメントを使用して `Graph` に接続します。このフラグメントは RTSP に接続し、対応する RTP デパケタイザーとパーサーを選択し、ストリームをデコードして生のフレームに変換します。

## Walkthrough

これは、入力が*プログラム外部*から発生する最初の章です。以前の章では、テスト画像を生成したり、ディスクからファイルを読み込んだりしていましたが、ここでは、ネットワークストリームからフレームが継続的に到着し、そのフレームをできるだけ速く処理します。この仕組みは、再利用可能な`Graph`フラグメントである`RtspDecodedInput`であり、RTSPから生のフレームへの変換処理全体を1つのインターフェースにまとめます。

この章では、あえて「デコードされたフレームを処理する」ところで終了します。それらのフレームを`Model`に供給する方法については、別の場所で説明します（001は単一のモデル実行用、007はモデルをパイプラインに組み込む用、015はモデルをグラフ内に埋め込む用）。最終的には、RTSP URLに接続し、各デコードされたフレームのテンソルの形状を出力することで、ストリームが正常に動作していることを確認します。

これは*コンシューマー*のみです。ストリームを公開するには、別のRTSPサーバー（例：`mediamtx`）を実行し、`--url`をそのサーバーに設定します。

### RTSPクライアントを構成する{#step-configure-rtsp}

`RtspDecodedInputOptions`は、ソースとデコーダーを構成します。`url`は、`rtsp://...`ソースを選択します。`codec`は、エンコードされた形式を選択します。デフォルトはH.264です。このチュートリアルでは、`avc`、`h265`、および`hevc`もサポートされており、AVCはH.264、HEVCはH.265と同等です。

ソースのフレームレートがわかっている場合は、`source_fps`を設定します。設定しない場合、このチュートリアルではRTSPソースをOpenCVで開き、報告されたFPSを読み取り、検出された値を`RtspDecodedInput`に供給します。このグループ自体はURLをプローブしません。プローブに必要なのはOpenCVのみであり、Pythonバージョンでは必要に応じてインポートされるため、`--source-fps`を供給することで、OpenCVなしで実行できます。プローブするには、`pip install opencv-python`でインストールします。H.265の場合、Neatはこの値を解析されたストリームキャップとデコーダー構成に伝播します。フレームレートは変更されません。H.265ストリームは、HEVC Mainプロファイル、8ビット、4:2:0入力を使用する必要があります。

`tcp = true`を設定すると、RTPがTCP経由で送信されます。TCPは順序を保持し、失われたセグメントを再送信するため、UDPと比較して、目に見える損失を減らすことができますが、失われたデータを回復する際に遅延が増加する可能性があります。

### グラフを構成する{#step-compose-graph}

2つのステージだけで`Graph`を構築します。それは、`RtspDecodedInput`フラグメント（ソース）と、シンプルな`Output`ノード（プルエンドポイント）です。フラグメントを追加するのは、単一の`add(...)`操作です。これは内部的に、接続/デパケット化/デコード要素に展開されるため、構成は意図のレベルに留まります。入力がパイプラインの*内部*から生成されるため、初期サンプルを受け取らない`build(RunOptions{})`のオーバーロードを使用します。ストリームがフレームを生成するため、事前にフレームを`build()`する必要はありません。

### デコードされたフレームをプルする {#step-pull-frames}

実行が開始されたら、ループしてタイムアウト付きで`pull(...)`を行います。各成功したプル操作により、1つのデコードされたフレームのテンソルを含む`Sample`が生成されます。チュートリアルでは、デフォルトのNV12出力を使用します。これは、YおよびUVプレーンのメタデータを持つ論理`[H, W]`テンソルとして表されます。何も（または空のテンソル）を返すプル操作は、`frame=N rtsp_timeout`を出力し、ループを中断します。これは通常、URLが間違っているか、ストリームがデータを送信していないことを意味します。タイムアウトは、停止したストリームがプログラムをハングアップするのを防ぎます。

**C++:** フレームは`tensors_from_sample(*sample, true)`を使用して抽出されます。ループは、`shape`を読み取る前に、空のリストがないか確認します。

**Python:** フレームは、`sample.tensors`の最初の要素から読み取られ、その後、その形状が出力されます。

## Run

この章では、ライブRTSPストリームを消費するため、アクセス可能な
`--url`を指定する必要があります。カメラがない場合は、互換性のあるビデオをRTSP
サーバー経由で公開し、`--url`をそれに向けてください。**Neatのインストールルート**（`share/`と
`lib/`を含むディレクトリ）から、**Python**および**C++（事前にビルドされたもの）**コマンドを実行します。**ソースからビルドする**コマンドは、**リポジトリのルート**から実行します。

自動化されたチュートリアル回帰テストは、両方のコーデックを実行します。これは、`SIMANEAT_TEST_RTSP_H264_URL`または`SIMANEAT_TEST_RTSP_H264_URLS`、および
`SIMANEAT_TEST_RTSP_H265_URL`または`SIMANEAT_TEST_RTSP_H265_URLS`から、最初に利用可能なURLを読み取ります。テストは、各ソースをプローブし、検出されたFPSをRTSPグループに供給します。

**Python:**
```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --source-fps 30 --frames 5
```

H.265の場合：

```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --codec hevc --source-fps 30 --frames 5
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_018_consume_rtsp_stream
./build/tutorials-standalone/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

予想される出力（形状は、ストリームの解像度とデコーダーの形式によって異なります）：

```text
frame=0 shape=[720, 1280]
frame=1 shape=[720, 1280]
frame=2 shape=[720, 1280]
frame=3 shape=[720, 1280]
frame=4 shape=[720, 1280]
```

ストリームにアクセスできない場合、代わりに`frame=0 rtsp_timeout`が表示されます。この章のC++ソースコードを、カスタムの`CMakeLists.txt`（追加のフォルダーは不要）を使用して、独自のプロジェクトに統合する方法については、ランディングページにある[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## ソースファイル
- C++: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.cpp`
- Python: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py`
- Python OpenCV FPSプローブ: `tutorials/018_consume_rtsp_stream/probe_rtsp_fps.py`
