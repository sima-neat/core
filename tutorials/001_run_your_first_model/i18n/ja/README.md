# 001 最初のモデルを実行する

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Model | resnet_50 |
| Labels | model, inference, foundations |

## Concept

コンパイル済みの ResNet-50 アーカイブをロードし、画像を入力として与え、上位1位のクラスを読み出します。これは、「モデルアーカイブがあります」から「予測値が得られました」までの最短のパスです。

## Walkthrough

これは最初の章です。目標は、可能な限り最小限の完全な推論です。コンパイルされたモデルを取得し、1つの画像を入力として与え、予測されたクラスのインデックスを出力します。グラフも、スレッドも、ストリーミングもありません。すべての Neat プログラムが構築されている3つの呼び出しだけです。

*コンパイルされたモデル* は、デプロイ可能な `.tar.gz` アーカイブであり、MPK推論コントラクトが含まれています。これには、モデルのアーティファクトと、ターゲットデバイスで実行するために Neat が必要とするランタイムメタデータが含まれます。アーカイブを解凍したり、ステージを自分で接続したりする必要はありません。Neat にアーカイブを指し示し、入力を与え、出力を読み取ります。最終的には、3行で推論を実行し、`top1=` クラスのインデックスを出力できます。

### モデルをロードする {#step-load-model}

最初の行は、ディスク上のパスを、実行可能な `Model` に変換します。コンストラクションは、アーカイブをロードし、実行に備えて準備します。

**C++:** `build_options(size)` を2番目の引数として渡して、このモデルが期待する入力コントラクト（RGBカラー、`224×224`、およびResNet-50がトレーニングされたImageNet正規化）を宣言します。ここで宣言することで、ランタイムは、生の画像をモデルが要求するテンソルに変換する方法を認識します。

**Python:** `build_options(size)` を使用して、`pyneat.Model` を構築するときに、同じコントラクトを渡します。

### 入力を準備する {#step-prepare-input}

次に、分類する正確に1つの画像を生成します。`--image` を渡すと、画像が読み込まれ、`224×224` にリサイズされ、入力コントラクトに一致するようにRGBに変換されます。それ以外の場合は、単色のグレーフレームを合成して、完全なロード→実行→読み取りパスが、手元にアセットがなくても、最初から最後まで実行されるようにします。

**C++:** フレームは `cv::Mat` であり、`load_rgb(...)` によって生成されるか、グレーのプレースホルダーとして生成されます。

**Python:** フレームは、`load_image(...)` によって構築され、RGB画像メタデータを持つ `Tensor` としてラップされたNumPy配列です。

### 推論を実行し、結果を読み取る {#step-run-inference}

3番目の行は、実際の処理を実行します。`run()` は、入力と `timeout_ms` を受け取り、モデルを同期的に実行し、出力を返します。`timeout_ms` は、待機する最大壁時計時間です。`2000` ミリ秒の場合、「デバイスが2秒以内に出力を生成しない場合は、エラーを発生させる」という意味であり、無限にハングするわけではありません。（`-1` を渡すと、無期限にブロックされます。実際のコードでは、有限の値を優先してください。）次に、`argmax` を使用して出力を単一のクラスインデックスに減らし、`top1=` を出力します。

**C++:** `run()` は `TensorList` を返します。最初のテンソルのバイトを `map_read()` を使用して読み取ります。

**Python:** テンソル/画像入力を持つ `run()` は `TensorList` を返し、`outputs[0].to_numpy()` がNumPy配列を `argmax` に渡します。

これですべてです。後続の章のすべて（非同期、パイプライン、カスタムグラフ）は、これらの同じ3つの手順（構築、入力、読み取り）に基づいて構築されています。

## Run

実行すると、予測されたクラスのインデックスが標準出力に出力されるはずです。**Neat のインストールルート**（`share/` と `lib/` を含むディレクトリ）から **Python** および **C++（事前に構築されたもの）** コマンドを実行します。**ソースからビルドする** コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/001_run_your_first_model/run_your_first_model.py \
  --model /tmp/resnet_50.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_001_run_your_first_model
./build/tutorials-standalone/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

予想される出力（正確なインデックスは画像によって異なります）：

```text
top1=285
[OK] 001_run_your_first_model
```

この章のC++ソースを、カスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは必要ありません）については、ランディングページにある「[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)」を参照してください。

スループット、バッチ処理、またはライブストリームについては、第2章に進んでください。参照：[モデル](/develop-apps/development-workflow/model)。

## In Practice

チュートリアルとテストがモデルアーカイブ（`.tar.gz`）とサンプルアセットを探す場所、およびそれらをローカルで提供する方法。これは、モデルをベースにしたすべてのチュートリアルの前提条件です。

### `sima-cli` がPATHにあることを確認する

一部のテストは、非対話型のシェルから `sima-cli` を呼び出します。`sima-cli` をインストールした後、これを一度実行してください。

```bash
SIMA_CLI_BIN_DIR="<path-to-sima-cli-bin>"
grep -Fqx "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" ~/.bashrc || echo "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

次に、確認します。

```bash
/bin/sh -c 'command -v sima-cli'
```

### モデルアーカイブの場所と環境変数

抽出/ランタイム配置の制御ノブ：
- `SIMA_MPK_EXTRACT_ROOT=<dir>` は、ベースの抽出ディレクトリを設定します。
- `SIMA_MPK_CLEANUP_EXTRACTED=0` は、プロセス終了後に抽出された `proc_*` モデルデータを保持します。
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0` は、起動時のデッド `proc_*` のクリーンアップを無効にします。

#### ResNet50

検索順序：
1. `SIMA_RESNET50_TAR`（モデルごとのオーバーライド）
2. `SIMA_MODEL_TAR`（モデルアーカイブテスト/例の共有フォールバック）
3. `tmp/resnet_50.tar.gz`
4. ローカルファイルが `tmp/` に移動した場合：`resnet_50.tar.gz`、`resnet-50.tar.gz`

ダウンロード（`sima-cli` が利用可能な場合）：
```bash
sima-cli modelzoo get resnet_50
```

### サンプル画像

チュートリアル/テストで使用されるデフォルトの画像候補：
- `tmp/coco_sample.jpg`（見つからない場合はダウンロードされます）
- `test.jpg`
- `tests/assets/preproc_dynamic/ilena_488.jpg`

テストで使用されるCOCO画像のURLは、次のようにオーバーライドできます。
```bash
SIMA_COCO_URL=<custom_url>
```

### テストがダウンロードする場所

テストと例では、通常、ダウンロードされたアセットをリポジトリのルートにある `tmp/` に配置します。チュートリアルは、必要なアセットが不足している場合、正常にスキップされます。

### アセットのトラブルシューティング

- チュートリアルで `SKIP: missing ...` が出力された場合は、アセットを提供するか、フラグ（例：`--model <path>`、`--image <path>`）を渡します。
- `sima-cli` が利用できない場合は、環境変数を設定して、ローカルのモデルアーカイブを指し示します。

## ソースファイル
- C++：`tutorials/001_run_your_first_model/run_your_first_model.cpp`
- Python：`tutorials/001_run_your_first_model/run_your_first_model.py`
