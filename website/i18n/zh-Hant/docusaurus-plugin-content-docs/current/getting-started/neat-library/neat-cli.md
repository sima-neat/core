---
title: "Neat 指令列介面"
description: "使用 neat 命令來檢查並更新已安裝的 Neat 環境。"
sidebar_position: 5
---

已安裝的函式庫提供 `neat` 環境指令。您可以從 SDK 或 DevKit 執行此指令，以查看已安裝元件的版本、已安裝的 `sima-cli` 指令集，以及是否有可用的較新成品。

在 SDK 中，狀態輸出還包含來自 `$HOME/.insight-config/neat-port-map.json` 的 Insight 主機與埠的對應關係。

<ShellCommand prompt="sdk|devkit">
neat
</ShellCommand>

範例輸出：

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

## JSON 輸出

對於自動化和工具整合，請使用 JSON 輸出：

<ShellCommand prompt="sdk|devkit">
neat --json
</ShellCommand>

## 更新已安裝的元件

若要更新偵測到的管道中的 Neat Library 執行階段、`neat-insight`，以及已安裝的 `sima-cli` 劇本，請執行：

<ShellCommand prompt="sdk|devkit">
neat update
</ShellCommand>

## 下一步

繼續進行 [編譯模型](/compile-a-model/)，以準備用於 Modalix 的模型——
此程式碼在 SDK 中執行，且不需要 DevKit。若要在硬體上執行完整的應用程式，請參閱 [您好，Neat！](/develop-apps/hello-neat/minimal/)，它需要配對的 DevKit。
