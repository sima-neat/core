# 022 GenAIをグラフに組み込む

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, graph, composition, streaming, advanced |

## Concept

LLM、VLM、またはASRの処理が、より大規模な処理の1つの段階である場合に、GenAIのグラフフラグメントを使用します。 Neat グラフ。

## Walkthrough

ほとんどの生成AIアプリケーションは、直接的なモデルAPIから始めるべきです。生成AIが他のモデルと連携する必要がある場合、グラフによる構成が役立ちます。 Neat ステージ、名前付き入力、名前付き出力、ルーティング、またはアプリケーションレベルのオーケストレーション。

### GenAI グラフフラグメントを作成 {#step-create-fragment}

タスク固有のモデルハンドルを作成し、グラフフラグメントのオプションを設定し、パブリックモデルを構築します。 `Graph` フラグメント。

このビジョンと言語のフラグメントは、 `prompt`, `image`および `use_cached_image` 入力に加えて `tokens`, `done`, `encoded`および `error` 出力。音声文字起こし機能の一部は、次の機能を提供します。 `audio` そして `audio_path` 入力に加えて `tokens`, `done`および `error` 出力。

`SpeechTranscriberOptions` デフォルトでは、言語を自動的に検出し、
文字起こしを行います。設定してください。 `task` へ `ASRTask::Translate` C++または
`ASRTask.Translate` Pythonで音声を聞き取り、英語に翻訳します。 `done`
bundleは、検出されたソース言語を報告し、利用可能な場合は
`no_speech_prob` そして `avg_logprob`.

### フラグメントをアプリのグラフに追加します {#step-compose-graph}

このフラグメントを、より大規模なアプリケーションのグラフに追加します。このフラグメントは、公開エンドポイントの名前を保持するため、アプリケーションコードは名前を使用してデータをプッシュおよびプルできます。

### グラフの入力のビルドとプッシュ {#step-push-prompt}

グラフを組み込んで `Run`、イメージサンプルをプッシュして `image` 入力後、テキストサンプルをプッシュします。 `prompt` 入力を与え、GenAIステージでトークンを生成させます。

### トークンと完了メタデータを取得 {#step-pull-results}

～から取得 `tokens` ～まで `done` サンプルが到着しました。 `done` サンプルは、生成されたトークンの数や完了理由などのフィールドを含む一連のデータです。

## Run

～において Modalix DevKitLFM2-VL 1.6B VLMをダウンロードしてください。 Hugging Face を使用して LLiMa CLI：

```bash
llima pull LFM2-VL-1.6B-a16w4
```

チュートリアルを次の場所で実行してください。 Modalix ～とともに DevKit-ローカルモデルディレクトリとローカルイメージ：

**Python:**
```bash
python3 share/sima-neat/tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_022_compose_genai_into_graph
./build/tutorials-standalone/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

期待される出力は、グラフの説明と、`tokens`の出力から取得したストリーミング形式の回答を表示します。

## In Practice

GenAIがより大規模なアプリケーションのグラフの一部である場合に、このパターンを使用します。単純なリクエスト/レスポンスアプリケーションコードでは、`GenAIModel`、`VisionLanguageModel`、および`ASRModel`への直接の呼び出しを維持します。

## ソースファイル
- C++: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.cpp`
- Python: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py`
