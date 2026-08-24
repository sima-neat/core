# 005 モデルオプションの設定

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | yolo_v8s |
| Labels | model-options, configuration, contracts |

## Concept

`ModelOptions` は、入力データ、モデルのパイプラインの各段階、およびその出力デコードとの間の関係を定義する構造体です。デフォルトの動作から変更したい場合に最初に参照する場所となります。

## Walkthrough

第 001 章では、適切なデフォルト値を持つモデルをロードします。特に YOLOv8 のような検出モデルの場合、入力がどのようなピクセル形式とサイズで到着するか、どのように正規化されるか、そして生のネットワーク出力がどのようにフィルタリングされたボックスに変換されるかを *宣言* する必要があります。`ModelOptions` は、それらすべてを 1 つの構造体にまとめ、構築前に値を設定します。

この章では、YOLOv8 モデルを最初から最後まで設定し、次にランタイムがこれらのオプションから解決した契約を調べます。最後に、入力、前処理、および後処理のパラメータを設定し、解決された `input_specs()`/`output_specs()`/`metadata` を読み取り、設定されたモデルを通じて 1 つの決定的なフレームを実行します。

### 入力と前処理を宣言 {#step-set-input-preproc}

最初のブロックでは、フレームがどのようなもので、ネットワークに備えるためにどのように準備するかを記述します。`format` (`BGR`、ここでは) と `input_max_width`/`height`/`depth` の範囲は、ランタイムが検証し、バッファのサイズを決定する入力契約を設定します。正規化フィールドは、モデルがトレーニングされたときに使用された、チャネルごとの平均値と標準偏差を提供するため、生のピクセルはネットワークが期待する範囲にスケーリングされます。

**C++:** フィールドは `opt.preprocess.*` の下にあります: `kind = InputKind::Image`、`color_convert.input_format = PreprocessColorFormat::BGR`、および `normalize.enable = AutoFlag::On`。`mean`/`stddev` は `std::array<float, 3>` として定義されます。

**Python:** フィールドは `opt.preprocess.*` の下にあります: `kind = pyneat.InputKind.Image`、`color_convert.input_format = pyneat.PreprocessColorFormat.BGR`、および `normalize.enable = pyneat.AutoFlag.On`。`mean`/`stddev` はリストとして定義されます。

### 後処理を宣言 {#step-set-postproc}

2 番目のブロックは、検出器の出力を調整します。`decode_type` は YOLOv8 ボックスデコードパスを選択し、`score_threshold`、`nms_iou_threshold`、および `top_k` は生の検出をフィルタリングします。これにより、信頼度の低いボックスが削除され、重複するボックスがマージされ、残るボックスの数が制限されます。`boxdecode_original_width`/`boxdecode_original_height` は、デコーダーに、正規化された座標をピクセルにマッピングするために必要なソースフレームのジオメトリを提供し、`name_suffix` は生成されたステージ名を安定化させ、他のものと組み合わせたときにパイプラインのグラフが読みやすくなるようにします。

**C++:** `decode_type = BoxDecodeType::YoloV8`。ジオメトリフィールドは `boxdecode_original_width`/`boxdecode_original_height` です。

**Python:** `decode_type = pyneat.BoxDecodeType.YoloV8`。ジオメトリフィールドは `boxdecode_original_width`/`boxdecode_original_height` です。

### 解決されたコントラクトをロードして確認する {#step-load-and-inspect}

これらのオプションを使用して `Model` を構築すると、コントラクトがモデルアーカイブに対して解決されます。次に、それを読み戻します。`input_specs()` と `output_specs()` は、ネゴシエートされたテンソルの制約を報告し、`metadata()` は、アーカイブに組み込まれたキー/値のコントラクトを公開します。ロード後にこれらを検査することで、ランタイムがオプションを受け入れたこと、および使用する具体的な形状が確認され、それらの形状がわかります。

**C++:** 仕様は `TensorConstraint` の値です。具体的な形状を出力します。

**Python:** `input_specs()[0]` と `output_specs()[0]` から形状を出力し、さらに `len(model.metadata())` も出力します。

### 1フレームを実行する {#step-run-inference}

最後に、`640×640` の BGR フレームを合成し、構成されたモデルを通して実行し、コントラクト全体が最初から最後まで実行されることを確認し、出力されたテンソルの数を表示します。

**C++:** フレームは `cv::Mat` です。`run()` は `TensorList` を返し、その `size()` を `outputs=` として出力します。

**Python:** フレームは `Tensor.from_numpy(...)` を介して `Tensor` としてラップされます。`run()` は `TensorList` を返すため、その長さを出力します。

## Run

実行すると、解決された仕様の形状、メタデータのキー数、および出力の合計が表示されます。**Neat のインストールルート**（`share/` と `lib/` を含むディレクトリ）から、**Python** および **C++（事前にビルドされたもの）** のコマンドを実行します。**ソースからビルドする** コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/005_configure_model_options/configure_model_options.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_005_configure_model_options
./build/tutorials-standalone/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

予想される出力（形状とキーの数はモデルアーカイブによって異なります。C++ ビルドは詳細な仕様の行と `outputs=` を出力し、Python ビルドは形状と `output_count=` を出力します）。

```text
input_specs[0]: shape=[640,640,3]
output_specs[0]: shape=[]
metadata_keys=8
outputs=1
[OK] 005_configure_model_options
```

カスタムの `CMakeLists.txt` を使用して、この章の C++ ソースを独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## In Practice

### 詳細出力プリセット

フレームワークのビルド/実行メッセージは、`GraphOptions`、`Model::Options`、および `Model::RouteOptions` で `VerboseOptions` を使用して制御されます。

現在の開発におけるデフォルト設定：`VerboseOptions::debug_all()`。出力を少なくしたい場合は、`production()` または `quiet()` を明示的に呼び出します。

| プリセット | 目的 |
|---|---|
| `VerboseOptions::quiet()` | フレームワークの進行状況と詳細な出力を抑制します。 |
| `VerboseOptions::production()` | 実行段階の進行状況のみを表示します。 |
| `VerboseOptions::debug_plugins()` | 本番環境のユーザーエクスペリエンスを維持しつつ、プラグインと GStreamer に関する情報を表示します。 |
| `VerboseOptions::debug_all()` | すべてのトピックに対して、完全な詳細/詳細な出力を強制的に行います。 |

ランタイムにおけるキュー/スループットの調整については、[スループットとキューの深さの調整](/tutorials/tune-throughput-and-queues) を参照してください。

## ソースファイル
- C++: `tutorials/005_configure_model_options/configure_model_options.cpp`
- Python: `tutorials/005_configure_model_options/configure_model_options.py`
