---
title: "開発ワークフロー"
description: "SiMa.aiとNeatの開発プロセスの概要を、インストールからデプロイまでを網羅して説明します。"
sidebar_position: 4
---

# 開発ワークフロー

このページは、SiMa.aiとNeatを日常的にどのように使用するかを示す概要図です。各ステップは、詳細なページへのリンクとなっており、必要に応じてそちらを参照できます。

## ループ

典型的な Neat 開発サイクルは次のようになります。

1. **インストール** — ホストまたはデバイスに、`sima-neat` パッケージと、必要に応じて `pyneat` Pythonバインディングをインストールします。
2. **Hello Neat を試してみましょう** — 最小限のサンプルをコンパイルして、ライブラリが正しく設定されていることを確認します。
3. **コンパイル済みのモデルを選択してください** — Neat は、コンパイル済みのモデルアーカイブ（`.tar.gz`、通常は MPK と呼ばれます）を使用します。Model Zoo から選択するか、Model Compiler を使用して独自のモデルをコンパイルしてください。
4. **`Model` / `Graph` / `Run` を作成する** — モデルをロードし、グラフを構成し、そのジョブに最適な最小限のランタイムで実行します。
5. **実行と検証** — 入力を与え、出力を確認し、`GraphReport` / `MeasureReport` を使用して動作を検証します。
6. **チュートリアルを活用して反復学習する** — 単一の推論から、パイプライン、複数入力モデル、複数ストリームのグラフ、そして実運用レベルのエラー処理へとステップアップしましょう。
7. **デプロイ** — ターゲットデバイスにインストールされている Neat ライブラリとアプリケーションをリンクします。

## モデル、グラフ、または実行を選択してください。

最も小さな範囲で、目的を達成できるところから始めましょう。アプリがより多くの機能を持つようになれば、いつでも機能を拡張できます。

| もし何か必要なことがあれば…… | まず最初に | なぜ | 次の停車駅 |
| --- | --- | --- | --- |
| コンパイル済みのモデルを1回実行します。 | `Model.run(...)` | アーティファクトから出力テンソルへの最短経路。 | [最初のモデルを実行する](/tutorials/run-your-first-model) |
| 接続された Modalix PCIeカード上でモデルを実行します。 | `pcie::Model` / `pyneatpcie.Model` | カード上にネイティブアプリケーションを構築することなく、ホスト側で共同処理を行う。 | [PCIe コプロセッシング](/develop-apps/development-workflow/pcie-model) |
| モデル契約書または経路を確認する | `Model` | 仕様、メタデータ、およびルート情報には、モデルがどのような入力を受け入れ、どのような出力を生成するかについて説明されています。 | [モデル](/develop-apps/development-workflow/model) |
| アプリケーションのフローにモデルを追加します。 | `Graph` | 入力と出力に名前を付け、ノードを構成し、トポロジーを明確に保ちます。 | [グラフ](/develop-apps/development-workflow/graph) |
| 時間経過に伴い、グラフを再利用する。 | `graph.build(...)` → `Run` | プッシュ/プル、クローズ/ドレイン制御、測定、およびキューポリシーを提供します。 | [グラフを実行する](/develop-apps/development-workflow/pipeline) |
| 複数のストリームを同時に処理するか、最大の処理能力を追求します。 | `RunOptions` + 測定 | スループットの最適化には、キューポリシー、ドロップカウンター、およびストリームごとのデータが必要です。 | [スループットとキュー深度を調整する](/tutorials/tune-throughput-and-queues) |

もしこれが初めての操作手順であれば、まず[チュートリアル](/tutorials)の事前確認リストから始め、次にグラフ関連の機能を追加する前に、まず1つのモデルを実行してください。

## 主要な概念を簡単に見てみましょう。

開発ワークフローのページでは、これらの各項目について詳細に解説しています。以下に概要を示します。

- [モデル](/develop-apps/development-workflow/model) — コンパイルされたモデルパッケージを読み込み、実行可能なユニットとして公開します。
- [PCIeによる共同処理](/develop-apps/development-workflow/pcie-model) — ホストアプリケーションから、接続された Modalix PCIeカード上でコンパイルされたモデルを実行します。
- [生成AIモデル](/develop-apps/development-workflow/genai-model) — 生成モデルを直接実行するか、HTTP経由で提供します。
- [テンソルとサンプル](/develop-apps/development-workflow/core_types) — テンソルとサンプル。これは、ステージ間でやり取りされるペイロードとメタデータの集合です。
- [実行 / 推論](/develop-apps/development-workflow/overview) — 同期的に実行する（`run`）か、非同期的に実行する（`push` / `pull`）。
- [グラフ](/develop-apps/development-workflow/graph) — モデルのステージ、ノード、入力、および出力を組み合わせて、アプリケーションのグラフを作成します。
- [グラフを実行する](/develop-apps/development-workflow/pipeline) — グラフをライブの`Run`に組み込み、その後、プッシュ、プル、ドレイン、測定、調整を行い、スループットを最適化します。
- [ノード](/develop-apps/development-workflow/node) は、グラフを構成する基本的な要素です。

最初に1つのページだけ学習する場合は、[実行／推論の概要](/develop-apps/development-workflow/overview) から始めましょう。このページでは、`Model`、`Graph`、および `Run` のすべてを最初から最後まで一貫して説明しています。

## 次にどこへ行くか

新規ユーザー向けの段階的な導入手順：

- [Neat SDK](/getting-started/dev-environment/) — Neat SDK をインストールし、DevKit とペアリングし、`dk` を搭載したハードウェアで実行します。
- [構築する](/develop-apps/contribute/build) — ソースコードから Neat を `build.sh` を使用してビルドします（開発者向けワークフロー）。
- [こんにちは、Neat！](/develop-apps/hello-neat/minimal) — インストールされたライブラリにリンクする、最小限の CMake アプリケーション。
- [チュートリアル](/tutorials) —「最初のモデル」から「本番環境のパイプライン」までを段階的に解説する一連のチュートリアル。

より詳細な情報を必要とする場合に参照できる資料：

- [実行 / 推論](/develop-apps/development-workflow/overview) — `Model`、`Graph`、`Run`、ノード、および入出力について、概念ごとに詳細に説明します。
- [C++リファレンス](/reference/cppapi) — インストールされたヘッダーファイルで利用可能な、APIのすべての機能。
- [Pythonリファレンス](/reference/pythonapi) — `pyneat` のバインディングに関するドキュメント。

## あなたが書くことと、Neatが提供すること

Neat は、モデルのロード、検証、パイプラインの構築、スケジューリング、クリーンアップ、および診断といったランタイムを管理します。一方、入力と出力を接続し、結果に応じて動作するアプリケーションコードは、お客様が管理します。境界は、`include/` に含まれるパブリック API であり、これは**安定**していると見なされます。つまり、アプリケーションを書き換えることなく、Neat をアップグレードできます。

もし Neat のコードを 3 行だけ覚えるとしたら、以下の行を覚えておいてください。

```cpp
simaai::neat::Model      model(mpk_path);
simaai::neat::TensorList outputs = model.run(input_tensors, /*timeout_ms=*/2000);
simaai::neat::Mapping    view = outputs[0].map_read();  // inspect the output bytes
```

このドキュメントにある他のすべての要素、つまりグラフ、実行ハンドル、非同期キュー、およびマルチストリームアプリケーションは、その基本的な3行のストーリーを拡張したものです。
