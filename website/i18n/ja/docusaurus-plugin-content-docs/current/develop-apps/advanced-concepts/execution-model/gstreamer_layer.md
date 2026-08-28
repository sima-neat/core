---
title: "GStreamer の下"
description: "このフレームワークが GStreamer をどのように抽象化し、何がそのまま残り、いつ生の GStreamer を使用すべきか。"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/gstreamer_layer
---

# GStreamer の下層

Neat フレームワークのパイプラインは、GStreamer 上で実行されます。アプリケーションから見て表示されるほぼすべてのもの（`Graph`、`Run`、`Node`）は、GStreamer の概念を型付きでラップしたものです。このページでは、その階層構造について説明します。何が隠蔽され、何が隠蔽されないのか、そして、生の GStreamer を使用するのが適切なのはいつなのかを説明します。

## フレームワークが抽象化するもの

| GStreamer の概念 | フレームワークによる抽象化 |
|---|---|
| `gst-launch` テキストフラグメント | `Node::backend_fragment(int node_index)` |
| 要素名（`name=…`） | `Node::element_names()` からの決定的な `n<idx>_<role>` |
| パイプライン文字列（連結されたフラグメント） | `Graph::add()` が構築および連結 |
| Caps のネゴシエーション | `Graph::build()` が `NodeCapsBehavior` を介して Caps を検証 |
| `gst_pipeline_set_state()` | `Graph::run()` / `Run::start()` |
| バスメッセージ | `GraphReport::bus_messages` |
| `appsrc` プッシュ API | `Run::push()`（`InputRole::Push` を持つノードのみ） |
| `appsink` プル API | `Run::pull()` |
| 要素ごとのタイミング | `Run::start_measurement()` からの `MeasureReport` |

アプリケーションは、起動文字列を直接記述したり、要素を直接指定したり、GStreamer C API に直接アクセスしたりすることはありません。すべてはノードを介して行われます。

## どの程度まで情報が伝わるか

フレームワークが隠蔽しない（または隠蔽すべきではない）いくつかの GStreamer の概念：

- **Caps のセマンティクス** — ビデオ/オーディオ Caps が持つフィールド。アプリケーションコードは、[`FormatTag`](/reference/cppapi/files/include-pipeline-formatspec-h) を読み取り、関連する Caps フィールドに対応する `Sample` メタデータを検査できます。
- **バッファフラグ** — 不連続性、EOS、ギャップ。フレームワークは、これらのフラグを `Sample` 上で伝播するため、アプリケーションコードはストリームの境界に反応できます。
- **イベントの順序** — GStreamer は、イベント（Caps、セグメント、EOS）がバッファとともに順序どおりに流れることを保証します。フレームワークは、この順序をプル側で維持します。

構築されたグラフの正確な GStreamer 起動文字列を知りたい場合は、`Graph::describe()` を呼び出してください。これにより、パイプラインをバイト単位で再現する決定的な `gst-launch` 再生ツールが生成されます。

## 生の GStreamer を使用するタイミング

通常、アプリケーションコードでは必要ありません。適切なケースは次のとおりです。

- **カスタム GStreamer プラグイン** — フレームワークがノードとして提供していない GStreamer 要素が必要な場合は、プラグインをラップし、適切な `backend_fragment()` を出力するノードのサブクラスを作成します。詳細な設計の説明（§0.10）の「カスタムノードの構築」を参照してください。
- **診断ツール** — `Graph::describe()` からの `repro_gst_launch` 再生ツールは、GStreamer が使用する起動文字列そのものです。オフラインデバッグのために、これを `gst-launch-1.0` に貼り付けることができます。
- **プラグインの作成** — SiMa 独自の GStreamer プラグイン（`sima*` ファミリー）は、プラグインマニフェスト ABI（[`gst/SimaPluginStaticManifestAbi.h`](/reference/cppapi/files/include-gst-simapluginstaticmanifestabi-h) を参照）にドキュメント化されており、フレームワークによって自動的にロードされます。

## 決定性の保証

フレームワークの要素命名は決定論的です。つまり、同じオプションを持つ同じノードのリストは、常に同じ `gst-launch` 文字列を生成します。これにより、次のことが可能になります。

- `repro_gst_launch` フィールドが実際に再現可能になります。
- テストスナップショットが実行ごとに安定します。
- 要素の識別（例：測定の属性付けのため）が機械にとって扱いやすくなります。

命名規則は `n<node_index>_<role>` で、`role` はノードの作成者が選択する、短く安定した識別子です。この命名スキームに参加する公開ノードラッパーについては、[ノード API グループ](/reference/cppapi/groups/nodes) を参照してください。

## 詳細

- 「GStreamer 抽象化」— デザインの詳細な解説の §0.8 を参照。
- [Node API](/reference/cppapi/groups/nodes) — 決定論的なバックエンドフラグメントを生成する具体的なノードラッパー。
- [`Graph::describe()`](/reference/cppapi/classes/simaai-neat-graph) — 起動文字列を出力します。
- 「SiMa プラグインマニフェスト」— デザインの詳細な解説の §51 および §95 を参照。
