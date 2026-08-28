# 006 推論前に画像を前処理する

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | resnet_50 |
| Labels | preprocessing, normalization, image |

## Concept

前処理の段階（形式、サイズ、およびチャンネルごとの正規化など）を設定し、生の画像入力が、モデルの学習に使用されたテンソルと正確に一致するようにします。 正確な前処理は、モデルが正常に動作するか、そうでないかの違いを生むことがよくあります。

## Walkthrough

コンパイルされたモデルは、特定の形状と値の範囲で入力されることを想定しています。つまり、固定された色の順序、固定された次元、および学習時に使用された正規化のレシピです。前処理は、生のデコードされた画像を目的のテンソルに変換する段階です。これを間違えると、モデルは実行されますが、信頼性の低い無意味な結果を返すだけです。そのため、デプロイされたモデルが「壊れている」ように見える場合は、まず前処理を確認する必要があります。

この章では、最も頻繁に使用する前処理の制御（色の形式、入力/出力の次元、リサイズ動作、およびチャネルごとの`mean`/`stddev`正規化）を設定し、次に、完全なモデルに単一の決定論的なテンソルを適用する前に、モデルの前処理グラフを調べます。この章の終わりまでに、完全な前処理契約を宣言し、それをモデルに添付し、設定されたルートが存在することを確認します。

### 前処理契約を設定する{#step-configure-preproc}

これらのオプションは、前処理段階で適用される契約を宣言します。`format`（または`color_convert.input_format`）は、入力時の色の順序を固定します。`input_max_*`フィールドは、ランタイムが受け入れる動的な入力を制限します。リサイズ/出力の次元は、推論のために生成されるテンソルのサイズを設定します。そして、`normalize`とチャネルごとの`mean`/`stddev`定数は、値のスケーリングを適用します。正規化定数は、モデルの学習時のレシピと一致する必要があります。一致しない統計は、信頼性の低い出力の最も一般的な原因です。

**C++:** フィールドは`Model::Options::preprocess`の下に存在します。`color_convert.input_format`は`PreprocessColorFormat`列挙型を受け取り、`normalize.enable`は`AutoFlag`であり、`normalize.mean` / `normalize.stddev`は`std::array<float, 3>`です。

**Python:** フィールドは`ModelOptions.preprocess`の下に存在します。`color_convert.input_format`は`PreprocessColorFormat`列挙型を受け取り、`normalize.enable`は`AutoFlag`であり、定数は`normalize.mean` / `normalize.stddev`に割り当てられたリストです。

### モデルを構築する{#step-load-model}

アーカイブパスとオプションから`Model`を構築すると、前処理契約がロードされたモデルにバインドされます。これにより、モデルは前処理定義を保持するため、そこから派生したすべての段階または実行で同じレシピが再利用されます。

### 前処理を個別に検査する{#step-inspect-preproc}

この章では、完全なモデルを実行する前に、前処理フラグメントを検査します。これにより、後続のデバッグを行う前に、ルートが存在することを確認できます。

**C++:** `stages::Preproc(frames, model)` は、前処理ステップを単独で実行し、前処理された `Tensor` を直接返します。`pre.shape.size()`（ランク）と `pre.dtype` を読み取り、コントラクトが有効になっていることを確認します。

**Python:** `model.preprocess()` は、前処理 `Graph` の一部を返します。そのため、構成されたルートを検査するために `describe()` を出力します。その後、`model.run([tensor])` が完全なパスを実行し、出力数を報告します。

## Run

**Neat のインストールルート**（`share/` と `lib/` を含むディレクトリ）から、**Python** および **C++（事前にビルドされたもの）** のコマンドを実行します。**ソースコードからビルドする**コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/006_preprocess_images/preprocess_images.py \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_006_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_006_preprocess_images
./build/tutorials-standalone/tutorial_006_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

期待される出力（C++ ビルドは、前処理されたテンソルのランクと dtype 列挙型を出力します）。

```text
preproc_rank=3
preproc_dtype=1
[OK] 006_preprocess_images
```

（Python ビルドは、`preproc_graph=ready`、グラフの説明、および `output_count=...` を出力します。）この章の C++ ソースコードを、カスタムの `CMakeLists.txt` を使用して、独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## ソースファイル
- C++: `tutorials/006_preprocess_images/preprocess_images.cpp`
- Python: `tutorials/006_preprocess_images/preprocess_images.py`
