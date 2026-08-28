---
title: "エラーコードカタログ"
description: "安定したフレームワークのエラーコード、それが発生した場合、およびそれに対応する方法"
sidebar_position: 7
---

# エラーコードカタログ

Neatは、`NeatError`と`PullError`を通じて、型エラーなどの問題を報告します。各エラーには、安定したエラーコード、人間が読めるメッセージ、および利用可能な場合は、構造化されたコンテキストを含む`GraphReport`が含まれます。

エラーコードをプログラムによるトリアージに使用します。メッセージを開発者に表示します。すべての公開定数は、[`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h)に定義されています。

## 動作に関する破壊的な変更と移行

診断分類では、特定の GStreamer の根本原因が保持されるようになりました。パブリックメソッドのシグネチャは変更されていませんが、正確なエラー文字列を比較するコードは、変更が必要になる場合があります。

| 前回のマッチ | より詳細なコードが返されるようになりました | 移行 |
| --- | --- | --- |
| `misconfig.caps` ランタイム向けに GStreamer 交渉エラー | `misconfig.media_caps`または `misconfig.media_format` 形式が互換性がない場合に限る | メディアコードを処理します。保持します。 `misconfig.caps` これは、キャップのオーバーライドと隣接するノードのコントラクトに関するフレームワークの検証にのみ使用されます。 |
| 各`gst_parse_launch`の失敗に対して、`build.parse_launch`を処理します。| プラグインが見つからない、プロパティが無効、またはパイプラインの構文エラーなど、`build.plugin_missing`、`build.property_invalid`、または`build.pipeline_syntax`などの具体的なエラーコードを処理します。| 分類されていないパーサーの失敗の場合は、`build.parse_launch`をフォールバックとして使用します。|
| 伝播されたバス障害に対する`runtime.pull` | 根本原因コード（例：`misconfig.media_caps`、`io.rtsp_connection_failed`、または`resource.output_pool_exhausted`）| 根本原因コードを処理し、デフォルトのブランチを維持します。`runtime.pull`は、特定の原因がないローカルのプル障害に対するフォールバックとして残ります。|

文字列リテラルを繰り返し使用する代わりに、C++またはPythonの定数を使用してください。常に、新しいバージョンのNeat Libraryで導入されたコードに対して、デフォルトのパスを保持するようにしてください。

## パブリック定数

両方の言語 API で同じ値が利用できます。

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
| `DispatcherUnavailable`（レガシー）| `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## 設定ミス

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | グラフのトポロジーが無効であるか、入力/出力境界がありません。| グラフの接続を修正し、必要な`Input`または`Output`ノードを追加してください。|
| `misconfig.caps` | フレームワークの検証中に、キャプチャオーバーライドまたは隣接するノードとの契約が競合しています。| 宣言されたフォーマット、寸法、レート、および隣接するノードとの契約を調整してください。|
| `misconfig.input_shape` | 入力テンソルが、想定される形状またはデータ型と一致しません。| 想定される入力を提供するか、モデルオプションを通じてモデルの前処理を設定してください。|
| `misconfig.runtime_abi_mismatch` | Neatとインストールされたランタイムプラグインが互換性のないABIを使用しています。| 互換性のあるNeat Libraryとランタイムプラグインのビルドをインストールしてください。|
| `misconfig.graph_element_name` | カスタムフラグメントに、安定したノード名を割り当てることができない要素が含まれています。| カスタム要素には、安定した一意の名前を付けてください。|
| `misconfig.media_caps` | 接続されている GStreamer ステージは、互換性のないメディア機能を持っています。| ステージを調整するか、必要な変換、スケーリング、またはレート変換ノードを挿入してください。|
| `misconfig.media_format` | 接続されているステージで互換性のないメディア形式が使用されています。| 共通の形式を設定するか、明示的な形式変換を追加してください。|
| `misconfig.input_capacity` | 入力された画像が、設定された前処理の入力容量を超えています。| `input_max_width`と`input_max_height`の値を大きくするか、モデル処理の前に画像のサイズを調整してください。|
| `misconfig.tensor_dtype_missing` | テンソルのデータ型または形式が指定されていません。| 上流のテンソルコントラクトで、サポートされているデータ型を宣言してください。|
| `misconfig.option_out_of_range` | 現在の入力契約に対して、オプションが無効です。| 診断で示された範囲内の値にオプションを設定してください。|

## ビルドの失敗

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `build.parse_launch` | GStreamer が生成されたパイプラインを構築できません。| カスタムフラグメント、要素のプロパティ、およびプラグインの可用性を確認してください。|
| `build.pipeline_syntax` | カスタムの GStreamer パイプラインのフラグメントに、構文エラーがあります。| フラグメントを修正し、`gst-launch-1.0` を使用して検証してください。|
| `build.plugin_missing` | 必要な GStreamer 要素またはコーデックプラグインが利用できません。| コンポーネントをインストールまたは置き換え、その後 `gst-inspect-1.0` を使用して、それが正しく機能することを確認してください。|
| `build.property_invalid` | 要素のプロパティ名または値が無効です。 | 以下のプロパティを確認してください。 `gst-inspect-1.0 <element>`. |

## ランタイム時のエラー

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `runtime.pull` | より具体的なコードがない場合、プル操作は失敗します。 | 添付されているレポートと、最初に発生した上流のエラーを確認してください。 |
| `runtime.element_failed` | より具体的な分類がない場合、パイプラインのステージは停止します。 | 報告されたステージ構成と、その上流からの入力について修正してください。 |
| `runtime.output_timeout` | 設定された待機時間が経過するまでに、出力は生成されません。 | ソースのフローとバックプレッシャーを確認するか、待機時間が予想される場合はタイムアウトを調整してください。 |
| `runtime.unexpected_eos` | パイプラインが、必要な出力を生成する前にEOS（End of Stream）に到達しました。| 入力にEOSが早期に発生していないか確認し、十分な量の入力が供給されていることを確認してください。|

## I/Oエラー

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `io.parse` | Neat JSON、モデルコントラクト、またはステージ構成を解析できません。 | 構成の構文、スキーマ、および必須フィールドを検証します。 |
| `io.open` | Neat は、ファイル、デバイス、またはリモートリソースを開けませんでした。| パスまたはアドレス、アクセス許可、およびリソースの可用性を確認してください。|
| `io.file_not_found` | 入力ファイルが存在しません。| パスを修正し、ファイルがDevKit上に存在することを確認してください。|
| `io.permission_denied` | 必要なアクセス権限でファイルまたはデバイスを開けません。| 対象のリソースに対する適切な所有権またはアクセス許可を確認してください。|
| RTSPソースへの接続に失敗しました。`io.rtsp_connection_failed` | Neat がRTSPソースに接続できません。| URL、サーバー、ネットワーク接続、および認証情報を確認してください。|
| `io.camera_not_found` | 要求されたカメラは利用できません。| 利用可能なカメラを選択するか、デフォルトのカメラを使用してください。|
| `io.model_not_found` | 要求されたモデルアーカイブが見つかりません。| モデルのパスを修正し、モデルアーカイブがインストールされていることを確認してください。|
| `io.source_ended` | 入力ソースが通常終了に達しました。| そのソースの読み込みを停止するか、アプリケーションがより多くのデータを必要とする場合は、追加の入力を提供してください。|

## パイプラインの実行失敗

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | パイプラインの構成が無効であるか、GStreamer の構築後に、最終要素の名前が重複している、曖昧である、または欠落しています。| 各明示的な要素に、それが存在するセグメント内で一意の短い名前を付けてください。`name=` の宣言と名前付きパッドへの参照を同期させてください。|
| `build.parse_launch` | GStreamer は、構文、プラグイン、またはプロパティが無効であるため、最終的な起動文字列を解析または構築できません。| `GraphReport::pipeline_string` を調べてください。`gst-launch-1.0` を使用してフラグメントを、`gst-inspect-1.0` を使用してプラグインを検証してください。|

これらのチェックは、`Graph::build()` の実行中に自動的に行われます。入力に依存する接続されたセグメントの場合、最初の入力によってセグメントが生成されると、同じコードと `GraphReport` が表示されることがあります。

## コーデックのエラー

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `codec.invalid_h264_stream` | 入力に有効なH.264フレームが含まれていません。| 完全なH.264ストリームを供給し、設定されているコーデックを確認してください。|
| `codec.decode_failed` | デコーダーが受信したストリームをデコードできません。| コーデックを確認し、エンコードされた入力が完全で破損していないことを確認してください。|
| `codec.encode_failed` | エンコーダーが入力されたフレームをエンコードできません。| 入力形式、解像度、およびエンコーダー設定を確認してください。|

## リソースの障害

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `resource.memory_allocation_failed` | デバイス固有の理由がないにもかかわらず、必要なメモリの割り当てに失敗しました。| ストリーム数、解像度、またはバッファリングを減らし、他のワークロードで使用されているメモリを解放してください。|
| `resource.device_memory_exhausted` | デバイスの連続したDMA/CMAメモリが不足しています。| 同時ストリーム数、入力解像度、またはバッファの深さを減らしてください。|
| `resource.output_pool_exhausted` | すべての出力バッファが使用中です。| ゼロコピー出力は速やかに解放するか、所有権のあるコピーを使用してください。|
| `resource.buffer_too_small` | バッファーのサイズが、宣言されたフレームまたはテンソルのペイロードよりも小さい。| 上流の次元とストライドを修正するか、必要なバイト数を割り当てる。|
| `resource.disk_full` | 書き込みに失敗しました。これは、宛先に十分な空き容量がないために発生しました。| 空き容量を増やすか、別の宛先を選択してください。|

## インフラの故障

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat がアクセラレータのランタイムを取得できません。| DevKit との互換性を確認し、アクセラレータを専有しているワークロードを停止してください。|
| `infra.accelerator_execution_failed` | アクセラレータはモデルのステージを実行できません。 | パイプラインを再起動し、同時実行されるアクセラレータのワークロードを削減します。 |

## 内部エラー

| コード | 発生時 | 対処法 |
| --- | --- | --- |
| `internal.plugin_failure` | ユーザーが対応できるような分類なしに、Neatプラグインがエラーを起こしました。| 添付の`GraphReport`をキャプチャし、エラーをサポートに報告してください。|

`DispatcherUnavailable` は、互換性のために受け入れられている古い形式のスペルです。新しいアプリケーションでは、`infra.dispatcher_unavailable` と `error_codes::kDispatcherUnavailable` 定数を使用してください。

## プログラムでエラーを処理する。

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

`PullError.code` は、同じ定数を使用します。`what()` を解析したり、人間が読めるテキストと照合したりしないでください。

## 関連資料

- [診断とデバッグ](/reference/diagnostics) — 運用時のメッセージ、デバッグの詳細など。
  `GraphReport` コレクション。
- [プラグインのエラー形式](/reference/error_format) — GStreamerプラグインの構造化されたエラーフォーマット
  エラー。
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — 型付きの例外。
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) - 構造化されたエラーコンテキスト。
