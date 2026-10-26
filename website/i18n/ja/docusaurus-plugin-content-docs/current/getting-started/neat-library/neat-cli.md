---
title: "Neat コマンドラインインターフェース"
description: "インストールされた Neat 環境を検査および更新するには、neat コマンドを使用します。"
sidebar_position: 5
---

インストールされたライブラリは、`neat` 環境コマンドを提供します。SDKまたはDevKitから実行すると、インストールされているコンポーネントのバージョン、インストールされている`sima-cli`プレイブック、およびより新しいアーティファクトが利用可能かどうかを確認できます。

SDKでは、ステータス出力に、`$HOME/.insight-config/neat-port-map.json` から取得した Insight のホストとポートのマッピングも含まれます。

PyNeat または Insight がインストールされていなくても、ステータスは報告できます。バージョンの照会に失敗した場合、コンポーネントには `unknown` が表示され、`neat` はプロバイダーの診断を標準エラーに出力します。JSON 出力は既存のスキーマを維持し、取得できないバージョンを `null` として報告します。

分割構成のインストールでは `simaai-runner` と `simaai-scheduler` を表示します。DevKit のバージョンはインストール済みパッケージデータベースから、SDK のバージョンはキャッシュされた DEB の制御メタデータから取得します。従来の `neat-runtime` と `neat-gst-plugins` の行は、それらが存在する場合、または分割コンポーネントが検出されない場合に表示します。JSON には `components.runner` と `components.scheduler` を追加し、既存の `runtime` と `gstPlugins` キーを維持します。存在しないバージョンは引き続き `null` です。

<ShellCommand prompt="sdk|devkit">
neat
</ShellCommand>

従来の構成での出力例:

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

Insight やプレイブックの確認・更新を行わず、Neat Library ランタイムだけを更新するには、次を実行します。

<ShellCommand prompt="sdk|devkit">
neat update --core-only
</ShellCommand>

## 次のステップ

[モデルを構築する](/compile-a-model/) を実行して、Modalix 用のモデルを準備します。これは SDK 内で実行され、DevKit は必要ありません。ハードウェア上で完全なアプリケーションを実行するには、[こんにちは、Neat！](/develop-apps/hello-neat/minimal/) を参照してください。これには、ペアになった DevKit が必要です。

## PCIe ホストとカードの状態

`sima-pcie-host` パッケージは Ubuntu にも `neat` をインストールします。`neat` または `neat --json` で、ホストのインストール済みパッケージ、PyNeatPCIe、各 SiMa PCIe カードのボード情報とソフトウェアを確認できます。カードは PCI アドレスとインターフェースで表示します。PCI アドレスは現在の接続を示し、永続的なボードシリアル番号ではありません。

```bash
neat
neat --json
neat --offline
```

カードの照会には `pcie-setup.sh` と同じ、設定済みのホストインターフェース `10.0.N.1` とカードアドレス `10.0.N.2` を使用します。ネットワーク接続と信頼済み SSH キーを事前に設定してください。このコマンドはインターフェース設定、ホストキー登録、パスワード入力、カード上の sudo、パッケージ変更を行いません。インターフェースの欠落、標準外のアドレス、SSH エラー、ソフトウェア情報の取得失敗はカードごとに表示し、接続可能なカードのバージョンは引き続き表示します。

`NEAT_PCIE_SSH_USER` はカードの SSH ユーザー（既定値 `sima`）を選択します。`NEAT_PCIE_SSH_KEY` は秘密キー（既定値 `~/.ssh/sima_neat_pcie_ed25519`）を選択します。既存の SSH 設定とエージェント認証も利用できます。`--offline` はインターネット経由の更新確認を省略しますが、ローカル SSH によるカードの読み取りは行います。PCIe ホストの状態表示ではアーティファクト更新を確認しません。ホストの更新には `sima-cli neat install core/pciehost/<distro>/<arch>@<ref>` を使用してください。`neat update` は PCIe カードを更新しません。

JSON は `sima.neat.status.v1` を維持し、`components.pcieHost`、`components.pcieScheduler`、`components.pyneatpcie`、`pcie` を追加します。`pcie` にはホストのソフトウェア、PCI アドレス・インターフェース・アドレス・状態・ソフトウェア・エラーを持つ `cards` 配列、トポロジーエラーが含まれます。リモートのソフトウェア情報が取得できない場合は、バージョンを推測せず `null` を返します。
