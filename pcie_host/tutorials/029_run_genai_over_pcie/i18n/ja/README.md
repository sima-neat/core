# 029 PCIe 経由で GenAI モデルを実行する

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | Qwen3-0.6B-Autoround-a16w4, LFM2-VL-450M-Autoround-a16w4, whisper-small-a16w8 |
| Labels | PCIe, GenAI, LLM, VLM, Whisper |

## Concept

ホストアプリケーションから Modalix PCIe カード上でテキスト生成、画像への質問、音声の文字起こしを実行します。モデルファイル、リクエスト、結果は PCIe 経由で転送されます。アプリケーションは GStreamer パイプラインではなく、公開 GenAI API を使用します。

## Walkthrough

各例は、ホスト上の準備済み LLiMa モデルディレクトリを読み込み、上限を設けたリクエストを実行して、モデルを閉じます。まず各例を個別に実行してください。接続設定は共通ですが、リクエストの入力は異なります。

### LLM を読み込む {#step-load-llm}

C++ では `pcie::genai::GenAIModel`、Python では `pyneatpcie.genai.GenAIModel` を構築します。構築時にカードへモデルを読み込むため、`build()` やキュー選択は不要です。`connection.card_id` でカードを選択します。モデルの絶対パスと、現在の作業ディレクトリからの相対パスを使用できます。

### テキストリクエストを送信する {#step-text-request}

`prompt` と `max_new_tokens` を設定し、`run()` を呼び出します。結果には回答テキストと生成メトリクスが含まれます。例では生成トークン数の上限を既定で 128 に設定します。`--max-tokens` で変更できます。

### 回答をストリーミングする {#step-stream-answer}

LLM の例に `--stream` を渡すと、代わりに `TokenSample` を反復処理します。各テキスト断片を到着時に表示してください。最終サンプルには生成完了時のメトリクスが含まれます。各モデルで同時に実行できるリクエストは 1 つだけです。

### モデルを閉じる {#step-close-model}

C++ は推論後に `close()` を明示的に呼び出し、例外で処理が中断した場合はデストラクタでクリーンアップします。Python はコンテキストマネージャーを使い、エラー時も含めてブロックを抜ける際にモデルを閉じます。接続のタイムアウトは起動が 15 分、リクエストが 5 分です。

### RGB 画像を準備する {#step-prepare-image}

VLM の例はホスト上で OpenCV を使って画像を読み込み、BGR を RGB に変換し、形状 `[H, W, 3]` の連続した UInt8 ピクセルを生成します。C++ は画素を所有するベクトルを PCIe テンソルに包み、Python は NumPy 配列を渡します。ピクセルを正規化したり、カード上の画像パスに置き換えたりしないでください。

### 画像について質問する {#step-image-request}

VLM を読み込み、プロンプトとともに RGB 入力を `request.images` に設定します。画像の前処理はモデルが行います。返されたテキストが質問への回答です。PCIe リクエストに画像ファイルのフィールドはありません。

### Whisper で音声を文字起こしする {#step-audio-request}

Whisper ディレクトリを読み込み、`audio_file` にホスト上の音声パスを設定します。`language="auto"` で言語を検出するか、`--language` で入力言語を指定します。既定では文字起こしを行い、`--translate` を指定すると英語への翻訳を選択します。文字起こし、言語、任意の無音声確率を確認してください。ASR リクエストにチャットプロンプトを追加しないでください。

## Run

[PCIe ホストのインストール](/getting-started/neat-library/pcie-host/) と [チュートリアルの準備](/tutorials/before-you-run) に従ってください。PCIe GenAI API を提供する対応したホストとカードのパッケージを使用します。以下のコマンドはすべてホスト上で、展開した PCIe extras のルートから実行してください。

### 準備済みモデルをダウンロードする

これらの SiMa.ai Model Zoo リポジトリにはコンパイル済み LLiMa アセットが含まれています：[Qwen3](https://huggingface.co/simaai/Qwen3-0.6B-Autoround-a16w4)、[LFM2-VL](https://huggingface.co/simaai/LFM2-VL-450M-Autoround-a16w4)、[Whisper](https://huggingface.co/simaai/whisper-small-a16w8)。実行するモデルだけをダウンロードしてください。モデルカードを確認し、インストール済みプラットフォームに対応するアセットを使用してください。

```bash
source ~/pyneatpcie/bin/activate
python -m pip install huggingface_hub opencv-python
mkdir -p models
hf download simaai/Qwen3-0.6B-Autoround-a16w4 \
  --local-dir models/Qwen3-0.6B-Autoround-a16w4
hf download simaai/LFM2-VL-450M-Autoround-a16w4 \
  --local-dir models/LFM2-VL-450M-Autoround-a16w4
hf download simaai/whisper-small-a16w8 \
  --local-dir models/whisper-small-a16w8
```

トークナイザー、設定、コンパイル済みアセットを含む各ディレクトリを完全な状態で保持してください。展開やカードへの手動コピーは不要です。未加工の Hugging Face チェックポイントは準備済み LLiMa ディレクトリではありません。独自モデルについては、このチュートリアルを実行する前に [LLiMa による GenAI](/genai-llima/) に従ってください。

### メディアを準備する

VLM のコマンドは付属の街路シーン画像を使用します。Whisper には、ホスト上で短い音声クリップを録音するか用意してください。FFmpeg がインストールされていれば、既存の録音を 16 kHz モノラル WAV に変換できます：

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le speech.wav
```

無音の入力で文字起こしの品質を判断しないでください。例では `--image` または `--audio` で別の画像や音声パスを指定できます。

**Python:**

```bash
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_llm.py \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_vlm.py \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_whisper.py \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./lib/sima-pcie-host/tutorials/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./lib/sima-pcie-host/tutorials/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_029_run_llm
./build.sh --target tutorial_029_run_vlm
./build.sh --target tutorial_029_run_whisper
./build/tutorials-standalone/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./build/tutorials-standalone/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./build/tutorials-standalone/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

C++ のソースビルドには PCIe 開発パッケージと OpenCV 開発ファイルが必要です。すべての例は既定でカード 0 と、接続の既定 SSH ユーザー sima を使用します。環境が異なる場合は `--card N`、`--user USER`、`--ssh-key PATH`、`--card-host ADDRESS` を追加してください。設定済みの SSH アカウントを使用してください。例はアクセス設定を行いません。

LLM の回答、シーンの説明、または録音した音声に対応する文字起こしが得られます。表現はモデルと入力に依存します。実行成功だけでは品質の確認になりません。

## In Practice

独立したモデルには別のハンドルを使用し、カードのメモリと計算資源の制約を考慮してください。チャットでは `prompt` ではなく `messages` で履歴を渡します。独立した呼び出しは以前のターンを記憶しません。ストリームをキャンセルする場合は、`cancel()` を呼び出して最終サンプルまで消費してからモデルを再利用してください。

API の概要は [PCIe GenAI API](/develop-apps/development-workflow/genai-model/pcie-api) を参照してください。スタンドアロンの GenAI チュートリアルは異なる名前空間とローカルのモデル実行を使用します。そのモデルハンドルをここで代用しないでください。

## ソースファイル

- `run_llm.cpp` / `run_llm.py`
- `run_vlm.cpp` / `run_vlm.py`
- `run_whisper.cpp` / `run_whisper.py`
- `tutorial_args.h`（共通の C++ コマンドライン解析）
- `../assets/street-scene.png`
