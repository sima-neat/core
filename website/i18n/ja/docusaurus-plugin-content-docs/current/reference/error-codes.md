---
title: "エラーコードカタログ"
description: "安定したフレームワークのエラーコード、その発生条件、および対処方法"
sidebar_position: 7
---

# エラーコードカタログ

Neat は、`NeatError` と `PullError` を通じて型付きの障害を通知します。各障害は、安定した
エラーコード、人間が読めるメッセージ、および利用可能な場合は構造化されたコンテキストを持つ `GraphReport` を提供します。

プログラムによるトリアージにはエラーコードを使用してください。メッセージは開発者に表示してください。公開定数の
完全な一覧は
[`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h) にあります。

## 動作上の破壊的変更と移行

診断の分類体系は、GStreamer の具体的な根本原因を保持するようになりました。公開メソッドのシグネチャは
変更されていませんが、エラー文字列を完全一致で比較するコードは移行が必要になる場合があります。

| 以前の一致 | 現在返される、より具体的なコード | 移行 |
| --- | --- | --- |
| ランタイムの GStreamer ネゴシエーションエラーに対する `misconfig.caps` | `misconfig.media_caps`。形式のみが互換性を持たない場合は `misconfig.media_format` | メディアコードを処理してください。`misconfig.caps` は、caps オーバーライドと隣接ノードの契約に対するフレームワークの検証にのみ使用してください。 |
| すべての `gst_parse_launch` の失敗に対する `build.parse_launch` | `build.plugin_missing`、`build.property_invalid`、または `build.pipeline_syntax` | 具体的なビルドコードを処理してください。`build.parse_launch` は、分類されないパーサーの失敗に対するフォールバックとして残してください。 |
| 伝播したバス障害に対する `runtime.pull` | `misconfig.media_caps`、`io.rtsp_connection_failed`、`resource.output_pool_exhausted` などの根本原因コード | 根本原因コードを処理し、デフォルトの分岐を残してください。`runtime.pull` は、具体的な原因のないローカルの pull 失敗に対するフォールバックとして引き続き使用されます。 |

文字列リテラルを繰り返し記述せず、C++ または Python の定数を使用してください。新しい Neat Library ビルドで
追加されたコードに備えて、常にデフォルトの経路を残してください。

## 公開定数

両方の言語 API で同じ値を利用できます。

| エラーコード | C++ | Python |
| --- | --- | --- |
| `misconfig.pipeline_shape` | `error_codes::kPipelineShape` | `pyneat.ERROR_PIPELINE_SHAPE` |
| `misconfig.caps` | `error_codes::kCaps` | `pyneat.ERROR_CAPS` |
| `misconfig.input_shape` | `error_codes::kInputShape` | `pyneat.ERROR_INPUT_SHAPE` |
| `misconfig.runtime_abi_mismatch` | `error_codes::kRuntimeAbiMismatch` | `pyneat.ERROR_RUNTIME_ABI_MISMATCH` |
| `misconfig.graph_element_name` | `error_codes::kGraphElementName` | `pyneat.ERROR_GRAPH_ELEMENT_NAME` |
| `misconfig.media_caps` | `error_codes::kMediaCaps` | `pyneat.ERROR_MEDIA_CAPS` |
| `misconfig.media_format` | `error_codes::kMediaFormat` | `pyneat.ERROR_MEDIA_FORMAT` |
| `misconfig.input_capacity` | `error_codes::kInputCapacity` | `pyneat.ERROR_INPUT_CAPACITY` |
| `misconfig.tensor_dtype_missing` | `error_codes::kTensorDtypeMissing` | `pyneat.ERROR_TENSOR_DTYPE_MISSING` |
| `misconfig.option_out_of_range` | `error_codes::kOptionOutOfRange` | `pyneat.ERROR_OPTION_OUT_OF_RANGE` |
| `build.parse_launch` | `error_codes::kParseLaunch` | `pyneat.ERROR_PARSE_LAUNCH` |
| `build.pipeline_syntax` | `error_codes::kPipelineSyntax` | `pyneat.ERROR_PIPELINE_SYNTAX` |
| `build.plugin_missing` | `error_codes::kPluginMissing` | `pyneat.ERROR_PLUGIN_MISSING` |
| `build.property_invalid` | `error_codes::kPropertyInvalid` | `pyneat.ERROR_PROPERTY_INVALID` |
| `runtime.pull` | `error_codes::kRuntimePull` | `pyneat.ERROR_RUNTIME_PULL` |
| `runtime.element_failed` | `error_codes::kRuntimeElementFailed` | `pyneat.ERROR_RUNTIME_ELEMENT_FAILED` |
| `runtime.output_timeout` | `error_codes::kOutputTimeout` | `pyneat.ERROR_OUTPUT_TIMEOUT` |
| `runtime.unexpected_eos` | `error_codes::kUnexpectedEos` | `pyneat.ERROR_UNEXPECTED_EOS` |
| `io.parse` | `error_codes::kIoParse` | `pyneat.ERROR_IO_PARSE` |
| `io.open` | `error_codes::kIoOpen` | `pyneat.ERROR_IO_OPEN` |
| `io.file_not_found` | `error_codes::kFileNotFound` | `pyneat.ERROR_FILE_NOT_FOUND` |
| `io.permission_denied` | `error_codes::kPermissionDenied` | `pyneat.ERROR_PERMISSION_DENIED` |
| `io.rtsp_connection_failed` | `error_codes::kRtspConnectionFailed` | `pyneat.ERROR_RTSP_CONNECTION_FAILED` |
| `io.camera_not_found` | `error_codes::kCameraNotFound` | `pyneat.ERROR_CAMERA_NOT_FOUND` |
| `io.model_not_found` | `error_codes::kModelNotFound` | `pyneat.ERROR_MODEL_NOT_FOUND` |
| `io.source_ended` | `error_codes::kSourceEnded` | `pyneat.ERROR_SOURCE_ENDED` |
| `io.response_too_large` | `error_codes::kResponseTooLarge` | `pyneat.ERROR_RESPONSE_TOO_LARGE` |
| `codec.invalid_h264_stream` | `error_codes::kInvalidH264Stream` | `pyneat.ERROR_INVALID_H264_STREAM` |
| `codec.decode_failed` | `error_codes::kDecodeFailed` | `pyneat.ERROR_DECODE_FAILED` |
| `codec.encode_failed` | `error_codes::kEncodeFailed` | `pyneat.ERROR_ENCODE_FAILED` |
| `resource.memory_allocation_failed` | `error_codes::kMemoryAllocationFailed` | `pyneat.ERROR_MEMORY_ALLOCATION_FAILED` |
| `resource.device_memory_exhausted` | `error_codes::kDeviceMemoryExhausted` | `pyneat.ERROR_DEVICE_MEMORY_EXHAUSTED` |
| `resource.output_pool_exhausted` | `error_codes::kOutputPoolExhausted` | `pyneat.ERROR_OUTPUT_POOL_EXHAUSTED` |
| `resource.buffer_too_small` | `error_codes::kBufferTooSmall` | `pyneat.ERROR_BUFFER_TOO_SMALL` |
| `resource.disk_full` | `error_codes::kDiskFull` | `pyneat.ERROR_DISK_FULL` |
| `infra.dispatcher_unavailable` | `error_codes::kDispatcherUnavailable` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE` |
| `infra.accelerator_execution_failed` | `error_codes::kAcceleratorExecutionFailed` | `pyneat.ERROR_ACCELERATOR_EXECUTION_FAILED` |
| `infra.peripheral_daemon_unavailable` | `error_codes::kPeripheralDaemonUnavailable` | `pyneat.ERROR_PERIPHERAL_DAEMON_UNAVAILABLE` |
| `infra.peripheral_daemon_timeout` | `error_codes::kPeripheralDaemonTimeout` | `pyneat.ERROR_PERIPHERAL_DAEMON_TIMEOUT` |
| `DispatcherUnavailable`（レガシー） | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## 設定ミス

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | グラフのトポロジーが無効であるか、入力/出力の境界がありません。 | グラフの接続と、必要な `Input` または `Output` ノードを修正してください。 |
| `misconfig.caps` | フレームワークの検証中に、caps オーバーライドまたは隣接ノードの契約が互換性を持ちません。 | 宣言した形式、寸法、レート、および隣接ノードの契約を一致させてください。 |
| `misconfig.input_shape` | 入力テンソルが、想定される形状またはデータ型と一致しません。 | 想定どおりの入力を渡すか、モデルオプションでモデルの前処理を設定してください。 |
| `misconfig.runtime_abi_mismatch` | Neat とインストール済みのランタイムプラグインが互換性のない ABI を使用しています。 | 互いに対応する Neat Library とランタイムプラグインのビルドをインストールしてください。 |
| `misconfig.graph_element_name` | カスタムフラグメントに、安定したノード名を割り当てられない要素が含まれています。 | カスタム要素に、安定した一意の名前を付けてください。 |
| `misconfig.media_caps` | 接続された GStreamer ステージが、互換性のないメディア caps を要求しています。 | ステージを揃えるか、必要な変換、スケーリング、またはレート変換のノードを挿入してください。 |
| `misconfig.media_format` | 接続されたステージが、互換性のないメディア形式を要求しています。 | 共通の形式を設定するか、明示的な形式変換を追加してください。 |
| `misconfig.input_capacity` | ソース画像が、設定された前処理の入力容量を超えています。 | `input_max_width` と `input_max_height` を大きくするか、モデルステージの前でソースをスケーリングしてください。 |
| `misconfig.tensor_dtype_missing` | テンソル契約にデータ型または形式が指定されていません。 | 上流のテンソル契約で、サポートされているデータ型を宣言してください。 |
| `misconfig.option_out_of_range` | 現在の入力契約に対して、オプションが無効です。 | 診断に示された範囲内の値をオプションに設定してください。 |

## ビルドの失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `build.parse_launch` | GStreamer が、生成されたパイプラインを構築できません。 | カスタムフラグメント、要素のプロパティ、およびプラグインの有無を確認してください。 |
| `build.pipeline_syntax` | カスタムの GStreamer フラグメントの構文が無効です。 | フラグメントを修正し、`gst-launch-1.0` で検証してください。 |
| `build.plugin_missing` | 必要な GStreamer 要素またはコーデックプラグインを利用できません。 | コンポーネントをインストールまたは置き換えてから、`gst-inspect-1.0` で確認してください。 |
| `build.property_invalid` | 要素のプロパティ名または値が無効です。 | `gst-inspect-1.0 <element>` でプロパティを確認してください。 |

## ランタイムの失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `runtime.pull` | pull 操作が失敗し、より具体的なコードがありません。 | 添付されたレポートと、最初に発生した上流のエラーを確認してください。 |
| `runtime.element_failed` | パイプラインのステージが停止し、より具体的な分類がありません。 | 報告されたステージの設定と、その上流からの入力を修正してください。 |
| `runtime.output_timeout` | 設定された待機時間が経過するまでに出力が届きません。 | ソースのフローとバックプレッシャーを確認するか、待機が想定どおりの場合はタイムアウトを調整してください。 |
| `runtime.unexpected_eos` | 必要な出力を生成する前に、パイプラインが EOS に達しました。 | 入力で EOS が早期に発生していないかを確認し、十分な入力が供給されたことを確かめてください。 |

## I/O の失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `io.parse` | Neat が JSON、モデル契約、またはステージ設定を解析できません。 | 設定の構文、スキーマ、および必須フィールドを検証してください。 |
| `io.open` | Neat がファイル、デバイス、またはリモートリソースを開けません。 | パスまたはアドレス、権限、およびリソースの可用性を確認してください。 |
| `io.file_not_found` | 入力ファイルが存在しません。 | パスを修正し、ファイルが DevKit 上に存在することを確認してください。 |
| `io.permission_denied` | 必要なアクセス権でファイルまたはデバイスを開けません。 | 報告されたリソースの所有者または権限を修正してください。 |
| `io.rtsp_connection_failed` | Neat が RTSP ソースに接続できません。 | URL、サーバー、ネットワークの到達性、および認証情報を確認してください。 |
| `io.camera_not_found` | 要求されたカメラを利用できません。 | 利用可能なカメラを選択するか、デフォルトのカメラを使用してください。 |
| `io.model_not_found` | 要求されたモデルアーカイブが存在しません。 | モデルのパスを修正し、アーカイブがインストールされていることを確認してください。 |
| `io.source_ended` | 入力ソースが正常な終端に達しました。 | そのソースの読み取りを停止するか、アプリケーションがさらにデータを想定している場合は追加の入力を渡してください。 |
| `io.response_too_large` | 上限付きのローカルプロトコル応答が、文書化されたサイズ上限を超えました。 | カタログのサイズを減らすか、互いに対応するバージョンのクライアントとサービスをインストールしてください。 |

## パイプライン実体化の失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | パイプラインのトポロジーが無効であるか、GStreamer での構築後に最終的な要素名が重複、曖昧、または欠落しています。 | すべての明示的な要素に、その実体化されたセグメント内で一意の短い名前を付けてください。`name=` 宣言と名前付きパッドの参照を同期させてください。 |
| `build.parse_launch` | 構文、プラグイン、またはプロパティが無効なため、GStreamer が最終的な起動文字列を解析または構築できません。 | `GraphReport::pipeline_string` を調べ、フラグメントを `gst-launch-1.0` で、プラグインを `gst-inspect-1.0` で確認してください。 |

これらのチェックは `Graph::build()` の実行中に自動的に行われます。入力に依存する接続セグメントでは、
最初の入力がセグメントを実体化した時点で、同じコードと `GraphReport` が表面化することがあります。

## コーデックの失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `codec.invalid_h264_stream` | 入力に有効な H.264 フレームが含まれていません。 | 完全な H.264 ストリームを供給し、設定されたコーデックを確認してください。 |
| `codec.decode_failed` | 受け入れたストリームをデコーダーがデコードできません。 | コーデックを確認し、エンコードされた入力が完全で破損していないことを確かめてください。 |
| `codec.encode_failed` | 供給されたフレームをエンコーダーがエンコードできません。 | 入力形式、解像度、およびエンコーダーの設定を確認してください。 |

## リソースの失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `resource.memory_allocation_failed` | デバイス固有の原因なしに、必要なメモリ割り当てが失敗しました。 | ストリーム数、解像度、またはバッファリングを減らし、他のワークロードが使用しているメモリを解放してください。 |
| `resource.device_memory_exhausted` | デバイスの連続 DMA/CMA メモリが枯渇しています。 | 同時ストリーム数、入力解像度、またはバッファの深さを減らしてください。 |
| `resource.output_pool_exhausted` | すべての出力バッファが使用中のままです。 | ゼロコピー出力を速やかに解放するか、所有権を持つコピーを使用してください。 |
| `resource.buffer_too_small` | バッファが、宣言されたフレームまたはテンソルのペイロードより小さくなっています。 | 上流の寸法とストライドを修正するか、必要なバイト数を割り当ててください。 |
| `resource.disk_full` | 書き込み先の空き容量が不足しているため、書き込みが失敗しました。 | 空き容量を確保するか、別の書き込み先を選択してください。 |

## インフラストラクチャの失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat がアクセラレータのランタイムを取得できません。 | DevKit との互換性を確認し、アクセラレータを排他的に所有しているワークロードを停止してください。 |
| `infra.accelerator_execution_failed` | アクセラレータがモデルステージを実行できません。 | パイプラインを再起動し、同時に実行するアクセラレータのワークロードを減らしてください。 |
| `infra.peripheral_daemon_unavailable` | ローカルの SiMa Sentinel API ソケットが存在しないか接続を拒否した、Sentinel のペリフェラル検出が無効または停止している、あるいはインストール済みの Sentinel がペリフェラルカタログを提供できないほど古い状態です。 | `sima-cli neat install sentinel` で Sentinel をインストールまたは更新するか、`simaai-sentinel.service` を起動してから、そのジャーナルを確認してください。 |
| `infra.peripheral_daemon_timeout` | 上限付きのペリフェラルカタログ要求が完了しませんでした。 | `simaai-sentinel.service` とプロバイダーの状態を確認してから、再試行してください。 |

## 内部の失敗

| コード | 発生条件 | 対処方法 |
| --- | --- | --- |
| `internal.plugin_failure` | ユーザーが対処できる分類なしに、Neat プラグインが失敗しました。 | 添付された `GraphReport` を取得し、サポートに失敗を報告してください。 |

`DispatcherUnavailable` は、互換性のために受け入れられるレガシーの表記です。新しいアプリケーションでは
`infra.dispatcher_unavailable` と `error_codes::kDispatcherUnavailable` 定数を使用してください。

## プログラムでエラーを処理する

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build();
  // Push and pull application data.
} catch (const simaai::neat::NeatError& error) {
  if (error.report().error_code == simaai::neat::error_codes::kInputShape) {
    handle_input_contract_error(error.report());
  } else {
    throw;
  }
}
```

`PullError.code` も同じ定数を使用します。`what()` を解析したり、人間が読むためのテキストと照合したりしないでください。

## 関連資料

- [診断とデバッグ](/reference/diagnostics) — 本番向けのメッセージ、デバッグ詳細、および
  `GraphReport` の収集。
- [プラグインのエラー形式](/reference/error_format) — GStreamer プラグインのエラーに関する
  構造化された契約。
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — 型付きの例外。
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — 構造化されたエラーコンテキスト。
