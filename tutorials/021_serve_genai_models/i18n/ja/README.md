# 021 GenAIモデルの提供

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Beginner |
| Estimated Read Time | 15-20 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4, Qwen3-VL-4B-Instruct-GPTQ-a16w4, whisper-small-a16w8 |
| Labels | genai, server, llm, vlm, asr, http |

## Concept

複数のGenAIモデルをNeat GenAIサーバーの背後でホストすることで、UI、サービス、またはリモートクライアントが、1つのプロセスからLLM、VLM、およびASRエンドポイントを呼び出せるようにします。

## Walkthrough

ほとんどのアプリケーションでは、`GenAIServer`とそのOpenAI互換の`POST /v1/chat/completions`エンドポイントから開始します。組み込みのアプリケーションロジックが同じプロセスでモデルの呼び出しを所有する必要がある場合は、直接`model.run(request)`呼び出しを使用します。

完全なエンドポイントとリクエストコントラクトについては、[GenAI Serverリファレンス](/develop-apps/development-workflow/genai-model/genai-server)を参照してください。

### サーバーの設定 {#step-configure-server}

ホストとポートを選択します。デフォルトのホストは`0.0.0.0`で、これはModalixデバイスにアクセスできる他のマシンからの接続を受け入れます。

### モデルディレクトリの登録 {#step-register-models}

デプロイされた各モデルディレクトリを、提供する名前とともに追加します。このチュートリアルでは、`llm`、`vlm`、および`asr`を登録します。クライアントが`model`フィールドに送信するのは、この提供する名前です。

### 提供の開始 {#step-start-serving}

ブロッキングフォアグラウンドプロセスにする場合は`serve()`を呼び出し、アプリケーションがプロセスの残りのライフサイクルを所有する場合は`start()`を呼び出します。

サーバーが開始されたら、`GET /v1/models`を使用して、登録されたモデル名を確認します。

```bash
curl http://<modalix-ip>:9998/v1/models
```

応答には、このチュートリアルで登録された提供する名前、つまり`llm`、`vlm`、および`asr`が含まれている必要があります。

## Run

Modalix DevKitでは、Hugging FaceからLLM、VLM、およびASRモデルをLLiMa CLIを使用してダウンロードします。

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
llima pull whisper-small-a16w8
```

Modalixで、3つのDevKitローカルモデルディレクトリすべてを含むサーバーを開始します。

**Python:**
```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/serve_genai_models.py \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_021_serve_genai_models
./build/tutorials-standalone/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

開発中にサブセットのみを提供したい場合は、`--vlm`または`--asr`を削除します。

サーバーが実行されたら、最初にすべての提供する名前が登録されていることを確認します。

```bash
curl http://<modalix-ip>:9998/v1/models
```

次に、クライアントからエンドポイントを呼び出します。`<modalix-ip>` を、お使いの Modalix デバイスの IP アドレスまたはホスト名に置き換えてください。
以下に示すリクエストクライアントは、Python の `requests` を使用し、レスポンスをストリーミングし、サーバー側の TTFT に加えて、報告された場合のトークンごとの平均、最小、および最大 TPS を出力します。

### LLM へのテキストリクエスト

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_text.py \
  --server-ip <modalix-ip> \
  --model llm \
  "Give me three tips for designing a small REST API."
```

### LLM へのツール呼び出しリクエスト

OpenAI 互換の `POST /v1/chat/completions` エンドポイントと、Ollama 互換の
`POST /api/chat` エンドポイントは、`tools` 配列内の関数定義を受け入れます。各エントリには、`type: "function"`、`function` オブジェクト、および空でない文字列の `function.name` が必要です。関数の説明と JSON スキーマのパラメータは、`function` オブジェクト内に含めることができます。

```bash
curl http://<modalix-ip>:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "stream": false
  }'
```

`tool_choice` を `"auto"` に設定すると、モデルが宣言されたツールを選択するか、`"none"` に設定すると、ツールプロンプトと解析が無効になります。`tool_choice` を省略するか、`null` に設定した場合、`tools` が空でない場合は、`"auto"` と同じように動作します。不正なツール定義、配列でない `tools`、およびサポートされていない `tool_choice` の値または型の場合、HTTP 400 とともに `invalid_request_error` が返されます。同じツール定義とツール選択の検証は、推論が開始される前に、直接の `GenerationRequest` 呼び出しにも適用されます。

### VLM へのテキストと画像のリクエスト

リクエストスクリプトは、画像を Base64 エンコードし、OpenAI 互換の `image_url` コンテンツの一部として送信します。

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_image.py \
  --server-ip <modalix-ip> \
  --model vlm \
  image.jpg \
  "What is the main subject of this image?"
```

### ASR モデルへのオーディオリクエスト

転写クライアントは、デフォルトで自動ソース言語検出を行います。ソース言語がわかっている場合は、`--language` を使用します。

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  speech.wav
```

音声を英語に翻訳するには、`--translate` を追加します。クライアントは、`POST /v1/audio/translations` に同じ multipart リクエストを送信します。

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  --translate \
  speech-in-another-language.wav
```

転写には、`POST /v1/audio/transcriptions` を使用します。両方のルートは `stream=true` をサポートします。提供されたクライアントは、テキストをストリーミングし、検出されたソース言語、`no_speech_prob`、および最終イベントからの `avg_logprob` を出力します。`no_speech_prob` が高いほど、Whisper は入力に音声が含まれていない可能性が高いと判断します。`avg_logprob` は、生成されたトークンの平均ログ確率であり、値が高い（絶対値が小さい）ほど、より確信を持ってデコードされたことを示します。

## In Practice

ネットワーク境界が役立つ場合にサーバーを使用します。同じプロセス内でオーバーヘッドの少ないアプリケーションコードを実行するために、直接`GenAIModel`、`VisionLanguageModel`、および`ASRModel`を呼び出します。

通常のアプリケーションでは、複数のモデル名を登録して、1つの`GenAIServer`プロセスを実行します。DevKitに十分なメモリがある場合、複数のサーバープロセスは異なるポートにバインドできますが、それぞれが独自のモデルインスタンスをロードし、同じMLAハードウェアゲートキーパーを共有するため、ハードウェアのスループットを増やす方法として扱うべきではありません。

`/v1/models`エンドポイントは、最も簡単な動作確認です。登録されたモデル名が返される場合、サーバーにアクセスでき、モデルレジストリが正しく設定されていることを意味します。

## ソースファイル
- C++: `tutorials/021_serve_genai_models/serve_genai_models.cpp`
- Python: `tutorials/021_serve_genai_models/serve_genai_models.py`
- リクエストクライアント：
  - `tutorials/021_serve_genai_models/request_chat_completion_text.py`
  - `tutorials/021_serve_genai_models/request_chat_completion_image.py`
  - `tutorials/021_serve_genai_models/request_audio_transcription.py`
