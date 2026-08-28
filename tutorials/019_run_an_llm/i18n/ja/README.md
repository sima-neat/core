# 019 Direct API を使用して LLM を実行する

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Intermediate |
| Estimated Read Time | 10 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4 |
| Labels | genai, llm, chat, history, streaming |

## Concept

GenAI モデルのディレクトリをロードし、シンプルなプロンプトを送信し、システムプロンプトを追加してから、同じパターンをチャット履歴とストリーミングに拡張します。

## Walkthrough

従来の `Model` チュートリアルでは、`.tar.gz` MPK アーカイブを使用します。GenAI モデルでは、代わりに LLiMa モデルディレクトリと `neat::genai` API を使用します。最小限のリクエストから始めます。モデルをロードし、`request.prompt` を設定し、実行して、回答を出力します。それが機能したら、会話の状態が必要な場合は、`request.messages` に切り替えます。

### モデルディレクトリをロードする {#step-load-model}

デプロイされた LLiMa モデルディレクトリを `GenAIModel` に指定します。このチュートリアルでは、`GenAIModel` を使用します。これは、ディレクトリが LLM、VLM、または ASR モデルであるかどうかを自動的に検出するためです。

**C++:** モデルのパスから `simaai::neat::genai::GenAIModel` を構築します。

**Python:** モデルのパスから `pyneat.genai.GenAIModel` を構築します。

### 1 つのプロンプトを送信する {#step-send-prompt}

`prompt` とトークン予算を使用して、`GenerationRequest` を構築します。これは、単発の質問、テスト、およびスクリプトにとって最も簡単な方法です。

### システムプロンプトを定義する {#step-system-prompt}

短いシステム指示を使用して、モデルの動作を制御します。`system_prompt` を使用して、単純なプロンプト要求に添付できます。チャット履歴に切り替える場合は、同じ指示をメッセージリストの `system` メッセージとして保持します。

### メッセージに切り替える {#step-store-history}

チャットスタイルの要求の場合、`prompt` ではなく、`messages` を使用します。システムメッセージとユーザーメッセージから開始し、要求を実行し、アシスタントの応答を保存します。モデルは、それ自体で以前の `run()` 呼び出しを記憶しません。アプリケーションがメッセージ履歴を管理します。

### 履歴を使用してフォローアップを行う {#step-follow-up}

別のユーザーメッセージを追加し、更新されたメッセージリストを送信し、回答を読み取ります。モデルは、アプリケーションが保持している完全な会話を認識します。

### 回答をストリーミングする {#step-stream-answer}

UI スタイルの出力の場合、`stream()` を呼び出し、返された `GenerationStream` を反復処理します。各トークンサンプルには、最新のテキストフラグメントが含まれます。

## Run

Modalix DevKit で、Hugging Face から Qwen3 4B などの LLM を LLiMa CLI を使用してダウンロードします。

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

Modalixのチュートリアルを、DevKitのローカルモデルディレクトリを使用して実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/019_run_an_llm/run_an_llm.py \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_019_run_an_llm
./build/tutorials-standalone/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

期待される出力は、単純なプロンプトへの回答、システムプロンプトによる回答、文脈を考慮したフォローアップ、およびストリーミング形式の最終的な応答です。

## In Practice

アプリケーションに必要な量のメッセージ履歴のみ保持します。長い履歴はコンテキストトークンを消費し、最初のトークンまでの時間を増加させます。永続的なチャットアプリケーションの場合、会話をモデルオブジェクトの外に保存し、各ターンで`GenerationRequest.messages`を再構築します。

## ソースファイル
- C++: `tutorials/019_run_an_llm/run_an_llm.cpp`
- Python: `tutorials/019_run_an_llm/run_an_llm.py`
