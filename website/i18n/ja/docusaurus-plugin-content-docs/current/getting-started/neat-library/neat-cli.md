---
title: "Neat コマンドラインインターフェース"
description: "インストールされた Neat 環境を検査および更新するには、neat コマンドを使用します。"
sidebar_position: 5
---

インストールされたライブラリは、`neat` 環境コマンドを提供します。SDKまたはDevKitから実行すると、インストールされているコンポーネントのバージョン、インストールされている`sima-cli`プレイブック、およびより新しいアーティファクトが利用可能かどうかを確認できます。

SDKでは、ステータス出力に、`$HOME/.insight-config/neat-port-map.json` から取得した Insight のホストとポートのマッピングも含まれます。

<ShellCommand prompt="sdk|devkit">
neat
</ShellCommand>

例：

```text
Neat Environment
  Mode               Neat SDK
  Sysroot            /opt/toolchain/aarch64/modalix
  Update check       online

Components
  Neat core              0.4.0 channel=release latest=0.4.0
  PyNeat                 0.4.0
  neat-runtime           0.4.0
  neat-gst-plugins       0.4.0
  neat-insight           0.0.6 channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         2.1.3 run activate-model-compiler to activate

Exposed Ports
  Insight Web UI     https://10.0.0.22:9900

  Name               Protocol Host Port (Start) Host Port (End)
  ------------------ -------- ----------------- ---------------
  mainUI             tcp      9900              -
  metadataUDP        udp      9100              9179
  rtsp.tcp           tcp      8554              -
  videoUDP           udp      9000              9079
  videoUI            tcp      8081              -
  webRTC             udp      40000             40199
  webSSH             tcp      8022              -
```

## JSON出力

自動化やツールとの連携には、JSON形式の出力を利用してください。

<ShellCommand prompt="sdk|devkit">
neat --json
</ShellCommand>

## インストールされているコンポーネントを更新

検出されたチャンネルから、Neat Libraryのランタイム、`neat-insight`、およびインストール済みの`sima-cli`のプレイブックを更新するには、次を実行します。

<ShellCommand prompt="sdk|devkit">
neat update
</ShellCommand>

## 次のステップ

[モデルを構築する](/compile-a-model/) を実行して、Modalix 用のモデルを準備します。これは SDK 内で実行され、DevKit は必要ありません。ハードウェア上で完全なアプリケーションを実行するには、[こんにちは、Neat！](/develop-apps/hello-neat/minimal/) を参照してください。これには、ペアになった DevKit が必要です。
