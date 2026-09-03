---
title: "Neat Library リリースノート"
sidebar_position: 3
---

# Neat Library リリースノート

SiMa.ai Neat Library のリリースノート。

## 未リリース

### 互換性のない変更

- Neat LibraryのC++ ABIは現在バージョン4となり、共有ライブラリのSONAMEは`libsima_neat.so.4`です。テンソルは、特徴抽出器のセマンティックメタデータを持ちます。公開されているGenAIのリクエスト/結果の型は、ASRタスク、言語、およびプローブのメタデータを持ちます。`GraphLinkOptions`には、リアルタイムの許容制限が含まれます。C++アプリケーションとプラグインを再構築し、対応するCoreランタイムと開発パッケージをインストールしてください。
- リアルタイムのグラフ合成では、`GraphLinkOptions`、`Graph::connect()`、および`Graph::build()`が使用されます。プレビュー用の`RealtimeGraphLinkOptions`、`connect_realtime()`、`build_fused_realtime_sources()`/ `build_fused_realtime_source()`、および`RealtimeEveryFrameByStream` APIは削除されました。`realtime_every_frame_by_stream`を含む保存されたグラフは、サポートされているポリシーを使用して再作成する必要があります。詳細は[ライブフラグメントを接続する](/develop-apps/development-workflow/graph/#connect-live-fragments)を参照してください。

### ランタイムの変更

- C++およびPythonでは、`SimaDecode`と`RtspDecodedInput`を通じて、ネイティブのH.265/HEVCデコードが利用可能です。`RtspEncodedInput`は、デコードせずに解析されたH.265アクセスユニットを提供します。H.265入力は、HEVC Mainプロファイル、8ビット、4:2:0を使用する必要があります。コーデックセレクターは、`H265`と`HEVC`の両方をサポートします。H.264セレクターも`AVC`をサポートします。`FormatTag` / `pyneat.Format`は、エンコードされたグラフの境界で同じエイリアスを受け入れ、`H264`と`H265`としてシリアライズします。
- `VideoSender`は、エンコードされたH.264またはH.265を、`VideoSenderOptions::Passthrough(codec)` / `pyneat.VideoSenderOptions.passthrough(codec)`を通じて再エンコードせずに、UDP経由のRTPとして転送します。H.265はデフォルトでRTPペイロードタイプ98を使用します。H.264は96を維持します。`H264RtpUdpFromEncoded()`は、`Passthrough(RtspCodec::H264)`に置き換えられ、非推奨となりました。
- 生の `VideoSender` 入力は、システムまたは SiMaAI メモリ内で NV12 であることが確定しており、インストール済みエンコーダーが `input-layout-aware=true` を通知する場合、フォーマット変換を自動的に省略するようになりました。他の生フォーマット、不明なメモリ/レイアウト、および信頼できるフォーマット契約を持たない入力では、従来どおり NV12 に変換します。`H264RtpUdpFromRaw(...)` C++ および Python API は変更されていません。
- RTSP 入力は、`RtspEncodedInputOptions` と `RtspDecodedInputOptions` のコーデックに依存しない単一の `payload_type` フィールドで RTP ペイロードタイプを選択します。`-1` はコーデックのデフォルト（H.264/H.265 は 96、MJPEG は 26）を選択し、`0` はペイロードフィルタリングを無効にし、正の値は指定したペイロードを選択します。`RtspEncodedInputOptions::h264_payload_type` と `mjpeg_payload_type` は非推奨で、解決済みペイロードを変更する場合はランタイムで一度警告します。
- 通常の`build()`は、適格なライブファンインに対して、自動的に融合されたローワーリングを選択するようになりました。直接エンコードされたH.264またはH.265 `VideoSender`ブランチは、デコードされたフレームのCPUコピーなしで、デコード前に融合されます。ソース、デコーダー、および送信者は、コーデックについて合意する必要があります。不一致のペアは、個別のパイプラインセグメントに留まります。ライブプレビューのために、そのエッジを`RealtimeLatestByStream`に設定すると、遅いビデオ受信機が古いアクセスユニットを置き換え、デコーダーブランチにバックプレッシャーをかけません。

- MIPI/libcameraソースが所有するグラフ（CVU/MLAモデルのルーティング前に適応型SiMaAIメモリハンドオフを含む）について、C++およびPythonの`CameraInput`に関するドキュメントとチュートリアルを追加しました。
- `MetadataSender`では、より大きなJSONメッセージを分割することで、UDPペイロードを1200バイト以内に収めるようになりました。このNeat Libraryのバージョンと同時に、またはそれ以前にメタデータチャンクを再構成するバージョンにInsightを更新してください。古いバージョンのInsightは、変更なしで最大1200バイトのJSONペイロードをサポートし続けます。

### グラフの作成と検証

- グラフの構成において、現在は1つのノードオブジェクトを1つの論理的な頂点として扱います。重複した挿入や重複するフラグメントのインポートはアトミックに失敗し、繰り返し行われる `connect()` 呼び出しでは、既存のノードを再利用してファンアウトを行います。
- 実体化されたパイプラインセグメントは、`build()` の際に最終的な GStreamer 名を検証するようになりました。明示的な `validate()` 呼び出しは不要です。重複または欠落した名前は、切り捨てられたパイプラインを生成する代わりに、`misconfig.pipeline_shape` でエラーとなります。
- カスタムフラグメントは、現在はすべての明示的な名前と、名前付きパッド参照を宣言とともに報告します。名前の衝突が発生した場合、自動的に名前が変更されるのではなく、拒否されます。

| リリース | 互換性のある Neat SDK | 備考 |
|---|---|---|
| 0.4.0 | 2.1.3.0 | [Neat Library 0.4.0](https://github.com/sima-neat/core/releases/tag/v0.4.0) |
| 0.3.0 | 2.1.2.3 | [Neat Library 0.3.0](https://github.com/sima-neat/core/releases/tag/v0.3.0) |
| 0.2.2 | 2.1.2.2 | [Neat Library 0.2.2](https://github.com/sima-neat/core/releases/tag/v0.2.2) |
| 0.2.1 | 2.1.2.1 | [Neat Library 0.2.1](https://github.com/sima-neat/core/releases/tag/v0.2.1) |
| 0.2.0 | 2.1.2 | [Neat Library 0.2.0](https://github.com/sima-neat/core/releases/tag/v0.2.0) |
| 0.1.0 | 2.0.0 | [Neat Library 0.1.0](https://github.com/sima-neat/core/releases/tag/v0.1.0) |
