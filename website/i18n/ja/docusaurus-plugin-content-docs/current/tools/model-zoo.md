---
title: "Model Zoo"
sidebar_position: 5
---

# Model Zoo

Model Zoo は、事前にコンパイルおよび量子化されたモデルをまとめたコレクションであり、SiMa デバイスで実行可能な状態になっています。

次のような場合に利用してください。

- Modalix ハードウェアにおけるモデルの精度とパフォーマンスを評価する
- 既知のモデルに対して、手動によるコンパイルと量子化を避ける
- 検証済みのモデルアーティファクトから開始する
- 特定のハードウェアターゲット向けに構築されたモデルを選択する

Model Zoo は、Neat C++ および PyNeat アプリケーション向けに、コンパイル済みモデルアーティファクトを提供します。ただし、GenAI モデルは含まれません。

利用可能なモデルの一覧：

```bash
sima-cli modelzoo list
```

ダウンロードする前に、モデルをチェックしてください。

```bash
sima-cli modelzoo describe yolov5
```

モデルアーティファクトをダウンロードしてください。

```bash
sima-cli modelzoo get yolov5s
```

モデル名は、リリースによって異なる場合があります。どのモデル識別子を使用すべきか不明な場合は、まず `sima-cli modelzoo list` を使用してください。

コマンドの詳細については、[`sima-cli modelzoo`](/tools/sima-cli/modelzoo/) のリファレンスを参照してください。

## 生成AIモデル

GenAI 向けに、SiMa.ai は [Hugging Face](https://huggingface.co/simaai) で、コンパイル済みの LLM、VLM、および ASR モデルコレクションを提供します。LLiMa CLI を使用してダウンロードしてください。

```bash
llima pull <model_name>
```

例：

```bash
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

GenAI モデルをダウンロードした後、LLiMa ランタイムを使用して DevKit 上で実行します。
セットアップとランタイムのコマンドについては、[LLiMa を活用した生成AI](/genai-llima/) を参照してください。
