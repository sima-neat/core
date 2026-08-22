---
title: "診断とデバッグ"
description: "GraphReportの診断情報、ランタイムエラーコード、およびグラフメトリクスのアーティファクトを収集します。"
sidebar_position: 9
---

# 診断とデバッグ

## グラフレポート

`GraphReport` は、構造化された診断情報を記録します。
- パイプライン文字列（再現用）
- 標準的な`error_code`（機械によるトリアージ）
- `repro_note`（人間による要約＋ヒント）
- ノードレポートと所有要素名
- バスのメッセージとエラーの詳細
- オプションのフロー/タイミングカウンタ

エラーが発生した場合、`NeatError` は、ログに記録したり、シリアライズしたりできる `GraphReport` を含みます。

## エラー分類

フレームワークのエラーには、安定したコードファミリーが使用されます。

| エラーコード | 意味 | 一般的な解決策 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | ノードの順序/形状に関する制約違反 | パッシュ型パイプラインでは、最初に`Input()`を、プル型パイプラインでは最後に`Output()`を配置するようにしてください。|
| `misconfig.caps` | フレームワークの caps-override または隣接するノードの契約の不一致 | `caps_override` と宣言されたノードの契約を一致させる |
| `misconfig.input_shape` | 入力テンソル/フレーム/サンプルの形状またはデータ型が、モデルの要件と一致しません。| 期待される形状とデータ型を指定するか、モデルの前処理を設定してください。|
| `misconfig.runtime_abi_mismatch` | Neatとランタイムプラグインが互換性のないABIを使用しています | バージョンが一致するNeat Libraryとランタイムをインストールしてください |
| `misconfig.graph_element_name` | カスタム要素に固定のノード名を割り当てることはできません | カスタム要素に、固定で一意の名前を付けてください |
| `misconfig.input_capacity` | 入力画像が、前処理の入力容量を超えています | `input_max_width` / `input_max_height` を大きくするか、モデル処理の前に画像を縮小してください |
| `misconfig.media_caps` | 隣接するGStreamerステージでは、互換性のないメディア機能が要求されています。| フォーマット、解像度、フレームレートを調整するか、変換を挿入してください。|
| `misconfig.media_format` | ステージでサポートされていないメディア形式が使用されています | サポートされている形式を設定するか、形式変換を挿入してください |
| `misconfig.tensor_dtype_missing` | テンソルコントラクトにdtype/フォーマットがありません。| 上流のコントラクトでサポートされているテンソルのdtypeを宣言してください。|
| `misconfig.option_out_of_range` | 現在のテンソルに対して、選択されたステージオプションが無効です。| 診断で示されている範囲内の値を選択してください。|
| `build.parse_launch` | `gst_parse_launch` の失敗は、それ以上の詳細な分類ができません。| 詳細については、添付されているレポートでパーサーのコンテキストを確認してください。|
| `build.pipeline_syntax` | カスタム GStreamer フラグメントの構文が無効です。 | フラグメントを修正し、検証します。 `gst-launch-1.0` |
| `build.plugin_missing` | 必要な GStreamer 要素またはコーデックプラグインがインストールされていません | インストールまたは置き換えを行い、`gst-inspect-1.0` で確認してください |
| `build.property_invalid` | 要素のプロパティが不明または無効です | プロパティ名と値を確認してください。`gst-inspect-1.0` を使用して確認します。|
| `runtime.pull` | より具体的な根本原因が特定されないまま、プル操作が失敗しました。 | 添付のレポートと、最初に発生した上流のエラーを確認してください。 |
| `runtime.element_failed` | より具体的なマッピングがないため、あるステージが失敗しました。 | 報告されたステージとその上流の入力を修正します。 |
| `runtime.output_timeout` | 設定されたタイムアウト時間内にデータが受信されませんでした。 | ソースフローを確認するか、予想されるタイムアウト時間を長くしてください。 |
| `runtime.unexpected_eos` | パイプラインが、必要な出力が生成される前にEOS（End of Stream）に到達しました。| ソースからの早期EOSと、十分な入力が供給されているかを確認してください。|
| `io.parse` | JSONまたはステージ構成の解析/スキーマの処理に失敗しました | 構成の構文と必須フィールドを検証してください |
| `io.open` | グラフの保存/読み込みファイルのオープン/読み込み/書き込みに失敗しました | パスの存在、アクセス権、およびストレージの状態を確認してください |
| `io.file_not_found` | 入力ファイルが存在しません | パスを修正し、ファイルがDevKitに存在することを確認してください |
| `io.permission_denied` | ファイルまたはデバイスが読み取り可能ではありません | 所有者/権限を修正してください |
| `io.rtsp_connection_failed` | RTSPソースに接続できません | URL、接続可能性、サーバー、および認証情報を確認してください |
| `io.camera_not_found` | 要求されたカメラは利用できません | 利用可能なカメラを選択するか、デフォルトのカメラを使用してください |
| `io.model_not_found` | 要求されたモデルアーカイブが存在しません | モデルのパスを修正し、インストールされていることを確認してください |
| `io.source_ended` | 入力ソースが通常終了に達しました | 入力の停止、または追加の入力の提供をお願いします |
| `codec.invalid_h264_stream` | 入力に有効なH.264フレームが含まれていません | 完全なH.264ストリームを供給するか、コーデックを修正してください |
| `codec.decode_failed` | ストリームの受信後、デコーダーが失敗しました | コーデックと入力の整合性を確認してください |
| `codec.encode_failed` | 提供されたフレームのエンコードに失敗しました | 入力形式、解像度、およびエンコーダーの設定を確認してください |
| `resource.memory_allocation_failed` | 必要なメモリの割り当てに失敗しました。| ワークロードのメモリ使用量を減らし、他のアプリケーションまたはパイプラインで使用されているメモリを解放してください。|
| `resource.device_memory_exhausted` | デバイスのDMA/CMA割り当てに失敗しました。| 同時ストリーム数、解像度、またはバッファリングを減らしてください。|
| `resource.output_pool_exhausted` | すべての出力バッファが使用中です | ゼロコピー出力を解放するか、所有権のあるコピーを使用してください |
| `resource.buffer_too_small` | バッファーのサイズが、宣言されたペイロードよりも小さい | 正しいサイズ/ストライドを指定するか、必要なバイト数を割り当てる |
| `resource.disk_full` | ストレージがいっぱいで書き込みに失敗しました | 空き容量を増やすか、別の保存先を選択してください |
| `infra.dispatcher_unavailable` | アクセラレーターのランタイムを取得できません。| 他の処理を停止し、DevKitとの互換性を確認してください。|
| `infra.accelerator_execution_failed` | アクセラレータがモデルのステージを実行できませんでした。| パイプラインを再起動し、同時実行されるアクセラレータの処理量を減らしてください。|
| `DispatcherUnavailable` | `infra.dispatcher_unavailable` の古い表記 | ハンドラーを標準のインフラストラクチャコードに移行する |
| `internal.plugin_failure` | ユーザーが対応できるような分類なしに、プラグインがエラーを起こしました。| エラーレポートを記録し、サポートに連絡してください。|

`PullError.code` は、同じ分類（例外パスだけでなく）を使用します。
C++およびPythonの定数名と、以前の粗いコードに一致するアプリケーションの移行ガイダンスについては、[エラーコードカタログ](/reference/error-codes) を参照してください。

プロダクションメッセージでは、意図的に GStreamer の内部構造を省略します。プラグインのデバッグの詳細度を上げると、生のGErrorドメイン/コード、要素ファクトリ、メッセージ、および構造化されたプラグインの詳細が追加されます。URIユーザー情報、`auth`、`playback-token`、`hdnts`、`stream-key`、および `tkn` を含む、認識された認証情報とURLの秘密パラメータは、どちらかの形式で保存する前に削除されます。レポートに表示されるパイプライン文字列、ノードフラグメント、再現コマンド、およびシリアライズされたJSONは、内部で保持されている実行可能なパイプラインを変更せずに削除されます。

## プログラムによる処理

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build(input);
  simaai::neat::Sample out;
  simaai::neat::PullError perr;
  const auto st = run.pull(500, out, &perr);
  if (st == simaai::neat::PullStatus::Error) {
    if (perr.code == simaai::neat::error_codes::kMediaCaps) {
      // Fix the incompatible upstream/downstream media contract.
    } else {
      // Handle another specific code, including future codes, or report it.
    }
  }
} catch (const simaai::neat::NeatError& e) {
  if (e.report().error_code == simaai::neat::error_codes::kPluginMissing) {
    // Install or replace the missing GStreamer component.
  }
}
```

## デバッグ用設定（環境）

重要な環境変数（詳細は[アーキテクチャ](/develop-apps/contribute/architecture)を参照）：
- `SIMA_GST_DOT_DIR`: 失敗したテストのグラフをDOT形式で出力します。
- `SIMA_GST_BOUNDARY_PROBES`：境界フローカウンタ
- `SIMA_GST_ELEMENT_TIMINGS`：要素ごとのタイミング
- `SIMA_GST_FLOW_DEBUG`：要素ごとのフローカウンタ
- `SIMA_GST_ENFORCE_NAMES`: 名付け規則を強制する

`NeatError::what()` と `GraphReport.repro_note` に、編集された生の GStreamer コンテキストを追加するには、エラーが発生したコマンドの両方の変数を設定します。

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` は Neat Library の設定ではありません。通常の使用時には、詳細な出力を無効にしておいてください。これは、短い診断実行のために設計されており、認識された認証フィールドは削除されますが、デプロイ固有のパスやメディアのアドレスが含まれる場合があります。

## デバッグワークフロー

1) まず、`GraphReport.error_code` を記録し、エラーの種類ごとに分類して集計します。
2) 具体的な状況と組み込みのヒントを把握するために、`GraphReport.repro_note` を記録してください。
3) パイプラインの説明テキスト：`Graph::describe_backend()` または `last_pipeline()`。
4) 構造化された診断情報を取得します：`MeasureReport::to_text()` または `NeatError::report()`。
5) `GraphReport.bus` を調べて、最初の端末で発生した `ERROR` の原因と詳細を確認してください。
6) ランタイムが停止またはタイムアウトした場合、フローの停止箇所を特定するために、境界または要素プローブを有効にします。

推奨されるサポートバンドル：
- `error_code`
- `repro_note`
- 完全な`pipeline_string`
- 最初の3～5件の端末バスエラー（`GraphReport.bus`）
- run/validate で使用される環境変数の上書き設定

## 顧客のグラフパフォーマンスに関するアーティファクト

スループット、レイテンシー、電力に関するレポートを作成する際は、グラフ形式で出力される JSON 形式の使用を推奨します。

```cpp
RunOptions opt;
opt.enable_board_power();        // graph-level power when supported by the board/SOM
Run run = graph.build(opt);

// run your normal push/pull loop inside a measurement window, then:
auto report = run.start_measurement().stop();
std::cout << report.to_text();
```

エクスポートは、スコープを明確に保ちます。

- `run.graph_metrics.throughput_fps`と`run.graph_metrics.power`は、グラフ全体の主要な指標です。
- `run.node_metrics[]` には、ノード/プラグインの遅延のみが含まれており、ノード/プラグインの消費電力は意図的に含まれていません。
- `latency_semantics`と`aggregation`は、値がプログラム実行期間全体の値であるか、測定された特定の時間間隔における変化量であるかを示します。
- `plugin_metrics_unattributed[]` は、正確に1つのノードにマッピングできなかったカーネル/プラグインの行を保持します。

測定対象のウィンドウについては、`Run::start_measurement()` を使用し、返された `MeasureReport` を
`run_to_json(run, report, ...)` / `save_run_json(run, report, ...)` に渡します。測定ウィンドウノードの
`min_ms` / `max_ms` は、ウィンドウローカルカウンターなしで累積最小/最大カウンターを正確に減算できないため、利用不可とマークされます。

補足：現在のDVTボードは、オプションの配線とJSONの形式を検証できますが、そのワット数測定値は数値的に信頼できるものとして扱われません。電力数値の検証には、SOMハードウェアを意図したプラットフォームとして使用してください。

## よくある不具合とその修正方法

| 症状 | 考えられる原因 | 解決策 |
| --- | --- | --- |
| `missing ... plugin` | GStreamerプラグインが見つかりません。| `GST_PLUGIN_PATH`を確認し、`gst-inspect-1.0 <plugin>`を実行してください。|
| `appsink 'mysink' not found` | 終端ノードがありません `Output()` | `Output` が、実行/ビルド パイプラインの最後のノードになっていることを確認してください |
| `caps_override is set; renegotiation disabled` | Capsロックが有効 | 削除 `caps_override` または入力された大文字の状態を維持 |
| `tensor caps change not supported` | ランタイムにおけるテンソルの形状/データ型の変更 | テンソルの形状/データ型を安定した状態に保つ（再ネゴシエーションなし） |

構造化されたプラグインのエラーと、それに対する具体的な解決策については、[トラブルシューティング](/reference/troubleshooting) を参照してください。
