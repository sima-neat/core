# 020 Direct API を使用して VLM を実行する

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, vlm, image, cache, multimodal |

## Concept

同じ画像について、画像を毎回エンコードし直さずに、繰り返し質問します。

## Walkthrough

ビジョン-言語モデルは、テキストと画像テンソルを受け入れることができます。1つの質問に対して、画像を `GenerationRequest.images` に直接添付します。繰り返し質問する場合は、画像を1回エンコードし、後続のリクエストでキャッシュされた画像埋め込みを再利用します。

### VLM と画像をロードする {#step-load-inputs}

デプロイされた LLiMa モデルディレクトリから `VisionLanguageModel` をロードし、ディスクから画像をデコードします。

**C++:** OpenCV を使用して画像を読み込みます。Neat は、3チャンネルの `cv::Mat` 入力を BGR として扱い、内部的に RGB に変換します。

**Python:** OpenCV でデコードし、BGR を RGB に変換し、NumPy 配列をリクエストに渡します。

### 直接画像を添付して質問する {#step-direct-image}

最初のリクエストに画像を直接添付します。これは最も簡単な方法であり、多くの場合、一度限りの視覚的な質問には十分です。

### 画像埋め込みをキャッシュする {#step-cache-image}

`encode(...)` を呼び出して、モデルに画像埋め込みをキャッシュします。この呼び出しは、画像が受け入れられ、キャッシュされた場合に `true` を返します。

### 後続の質問をする {#step-follow-up}

キャッシュされた画像を使用する必要がある各リクエストで、`use_cached_images = true` を設定します。同じキャッシュされた画像について、複数の質問をすることができます。このフラグがないリクエストは通常どおりに動作します。テキストのみのリクエストでは画像は使用されず、直接画像リクエストでは独自の `images` が使用され、別の `encode(...)` 呼び出しによってキャッシュされた画像が置き換えられます。

### チャットメッセージに画像を添付する {#step-message-image}

`messages` を使用する場合は、必要なユーザーメッセージに画像を添付します。これにより、画像が関連する正確なテキストの隣に配置されます。

## Run

Modalix DevKit で、Hugging Face から LLiMa CLI を使用して LFM2-VL 1.6B VLM をダウンロードします。

```bash
llima pull LFM2-VL-1.6B-a16w4
```

Modalix で、DevKit ローカルモデルディレクトリとローカル画像を使用して、チュートリアルを実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/020_run_a_vlm/run_a_vlm.py \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_020_run_a_vlm
./build/tutorials-standalone/tutorial_020_run_a_vlm \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image tests/images/people.jpg
```

期待される出力は、直接的な画像リクエストに対する1つの回答、キャッシュされた画像を再利用する複数のフォローアップ回答、およびメッセージレベルの画像リクエストに対する1つの回答です。

## In Practice

ユーザーが同じフレーム、製品画像、図、またはドキュメントページについて複数の質問をする場合、画像キャッシュを使用します。各リクエストで異なる画像を使用する場合は、キャッシュを避けてください。なぜなら、直接的な画像パスの方がシンプルで、プロンプトの状態を明確に保つことができるからです。

一部のモデルファミリーでは、キャッシュされた画像の再利用がサポートされていない場合があります。その場合は、各リクエストで直接的な画像を使用してください。

会話を構築し、1つのメッセージにのみ画像を含める必要がある場合は、`ChatMessage.images`を使用します。よりシンプルな単一プロンプト形式の場合は、最上位レベルの`GenerationRequest.images`を使用します。

## ソースファイル
- C++: `tutorials/020_run_a_vlm/run_a_vlm.cpp`
- Python: `tutorials/020_run_a_vlm/run_a_vlm.py`
