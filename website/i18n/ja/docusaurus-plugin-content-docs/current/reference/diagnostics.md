---
title: "診断とデバッグ"
description: "GraphReport の診断情報、ランタイムエラーコード、およびグラフメトリクスのアーティファクトを収集します"
sidebar_position: 9
---

# 診断とデバッグ

## GraphReport

`GraphReport` は、次の構造化された診断情報を記録します:
- パイプライン文字列（再現用）
- 正規の `error_code`（機械によるトリアージ用）
- `repro_note`（人が読むための要約とヒント）
- ノードレポートと所有する要素名
- バスメッセージとエラーの詳細
- オプションのフロー/タイミングカウンター

エラーが発生すると、`NeatError` は、ログに記録したり
シリアライズしたりできる `GraphReport` を保持します。

## エラーの分類体系

フレームワークのエラーは、次の安定したコードファミリーを使用します:

| エラーコード | 意味 | 一般的な対処 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | ノードの順序/形状の契約違反 | プッシュ型パイプラインでは `Input()` を先頭に、プル型パイプラインでは `Output()` を末尾に配置してください |
| `misconfig.caps` | フレームワークの caps オーバーライドまたは隣接ノードの契約の不一致 | `caps_override` と宣言されたノードの契約を一致させてください |
| `misconfig.input_shape` | 入力テンソル/フレーム/サンプルの形状またはデータ型がモデルの契約と一致しません | 期待される形状とデータ型を指定するか、モデルの前処理を設定してください |
| `misconfig.runtime_abi_mismatch` | Neat とランタイムプラグインが互換性のない ABI を使用しています | バージョンが一致する Neat Library とランタイムをインストールしてください |
| `misconfig.graph_element_name` | カスタム要素に安定したノード名を割り当てられません | カスタム要素に安定した一意の名前を付けてください |
| `misconfig.input_capacity` | ソース画像が前処理の入力容量を超えています | `input_max_width` / `input_max_height` を増やすか、モデルステージの前で縮小してください |
| `misconfig.media_caps` | 隣接する GStreamer ステージが互換性のないメディア caps を要求しています | 形式、解像度、フレームレートを一致させるか、変換を挿入してください |
| `misconfig.media_format` | ステージがサポートされていないメディア形式を受け取りました | サポートされている形式を設定するか、形式変換を挿入してください |
| `misconfig.tensor_dtype_missing` | テンソルの契約に dtype/形式がありません | 上流の契約で、サポートされているテンソルの dtype を宣言してください |
| `misconfig.option_out_of_range` | ステージのオプションが現在のテンソルに対して無効です | 診断に示された範囲内の値を選択してください |
| `build.parse_launch` | `gst_parse_launch` の失敗に、より具体的な分類がありません | 添付のレポートでパーサーのコンテキストを確認してください |
| `build.pipeline_syntax` | カスタム GStreamer フラグメントの構文が無効です | フラグメントを修正し、`gst-launch-1.0` で検証してください |
| `build.plugin_missing` | 必要な GStreamer 要素またはコーデックプラグインがインストールされていません | インストールまたは置き換えを行い、`gst-inspect-1.0` で確認してください |
| `build.property_invalid` | 要素のプロパティが不明または無効です | `gst-inspect-1.0` でプロパティ名と値を確認してください |
| `runtime.pull` | より具体的な根本原因がないまま pull が失敗しました | 添付のレポートと、最初の上流エラーを確認してください |
| `runtime.element_failed` | より具体的な対応付けがないままステージが失敗しました | 報告されたステージとその上流の入力を修正してください |
| `runtime.output_timeout` | 設定されたタイムアウトまでに出力が届きませんでした | ソースのフローを確認するか、想定されるタイムアウトを延ばしてください |
| `runtime.unexpected_eos` | 必要な出力の前にパイプラインが EOS に到達しました | ソースが早期に EOS を送っていないか確認し、十分な入力を供給してください |
| `io.parse` | JSON またはステージ設定の解析/スキーマの失敗 | 設定の構文と必須フィールドを検証してください |
| `io.open` | グラフの保存/読み込み時のファイルのオープン/読み取り/書き込みの失敗 | パスの存在、権限、ストレージの状態を確認してください |
| `io.file_not_found` | 入力ファイルが存在しません | パスを修正し、ファイルが DevKit 上に存在することを確認してください |
| `io.permission_denied` | ファイルまたはデバイスを読み取れません | 所有者/権限を修正してください |
| `io.rtsp_connection_failed` | RTSP ソースに接続できません | URL、到達可能性、サーバー、認証情報を確認してください |
| `io.camera_not_found` | 要求されたカメラを利用できません | 報告されたカメラを選択するか、デフォルトを使用してください |
| `io.model_not_found` | 要求されたモデルアーカイブが存在しません | モデルのパスを修正し、インストールされていることを確認してください |
| `io.source_ended` | 入力ソースが通常の終端に達しました | 消費を停止するか、追加の入力を提供してください |
| `io.response_too_large` | 上限付きのローカルプロトコル応答がサイズ上限を超えました | 互いに対応するバージョンのクライアントとサービスをインストールしてください |
| `codec.invalid_h264_stream` | 入力に有効な H.264 フレームがありません | 完全な H.264 ストリームを供給するか、コーデックを修正してください |
| `codec.decode_failed` | ストリームを受け入れた後にデコーダーが失敗しました | コーデックと入力の整合性を確認してください |
| `codec.encode_failed` | エンコーダーが供給されたフレームをエンコードできませんでした | 入力形式、解像度、エンコーダーの設定を確認してください |
| `resource.memory_allocation_failed` | 必要なメモリ割り当てに失敗しました | ワークロードのメモリ使用量を減らし、他のアプリケーションやパイプラインが使用しているメモリを解放してください |
| `resource.device_memory_exhausted` | デバイスの DMA/CMA 割り当てに失敗しました | 同時ストリーム数、解像度、またはバッファリングを減らしてください |
| `resource.output_pool_exhausted` | すべての出力バッファーが使用中のままです | ゼロコピー出力を解放するか、所有権のあるコピーを使用してください |
| `resource.buffer_too_small` | バッファーが宣言されたペイロードより小さいです | 寸法/ストライドを修正するか、必要なバイト数を割り当ててください |
| `resource.disk_full` | ストレージがいっぱいで書き込みに失敗しました | 空き容量を確保するか、別の保存先を選択してください |
| `infra.dispatcher_unavailable` | アクセラレーターのランタイムを取得できません | 競合するワークロードを停止し、DevKit との互換性を確認してください |
| `infra.accelerator_execution_failed` | アクセラレーターがモデルステージを実行できませんでした | パイプラインを再起動し、アクセラレーターで同時に実行する処理を減らしてください |
| `infra.peripheral_daemon_unavailable` | SiMa Sentinel がペリフェラルカタログを提供できません | `sima-cli neat install sentinel` で Sentinel をインストールまたは更新するか、`simaai-sentinel.service` を起動してください |
| `infra.peripheral_daemon_timeout` | ペリフェラルカタログの要求が時間内に完了しませんでした | `simaai-sentinel.service` とそのジャーナルを確認してから、再試行してください |
| `DispatcherUnavailable` | `infra.dispatcher_unavailable` の旧表記 | ハンドラーを正規のインフラストラクチャコードに移行してください |
| `internal.plugin_failure` | ユーザーが対処できる分類がないままプラグインが失敗しました | レポートを取得し、サポートに連絡してください |

`PullError.code` も同じ分類体系を使用します（例外経路に限りません）。
C++ と Python の定数名、および以前の粗いコードで照合していたアプリケーションの移行ガイダンスについては、
[エラーコードカタログ](/reference/error-codes) を参照してください。

本番環境向けのメッセージでは、GStreamer の内部情報を意図的に省略しています。プラグインのデバッグ
詳細度を上げると、生の GError のドメイン/コード、要素ファクトリ、メッセージ、および
構造化されたプラグインの詳細が追加されます。認識された認証情報と URL の秘密パラメーター（URI の
userinfo、`auth`、`playback-token`、`hdnts`、`stream-key`、`tkn` を含む）は、どちらの
形式でも保存前に秘匿化されます。レポート向けのパイプライン文字列、ノードフラグメント、再現コマンド、および
シリアライズされた JSON は、内部で保持される実行可能なパイプラインを変更せずに秘匿化されます。

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

## デバッグ用の設定（環境変数）

主な環境変数（詳細は [アーキテクチャ](/develop-apps/contribute/architecture) を参照）:
- `SIMA_GST_DOT_DIR`: 失敗時に DOT グラフを書き出します
- `SIMA_GST_BOUNDARY_PROBES`: 境界のフローカウンター
- `SIMA_GST_ELEMENT_TIMINGS`: 要素ごとのタイミング
- `SIMA_GST_FLOW_DEBUG`: 要素ごとのフローカウンター
- `SIMA_GST_ENFORCE_NAMES`: 命名規約を強制します

秘匿化済みの生の GStreamer コンテキストを `NeatError::what()` と
`GraphReport.repro_note` に追加するには、失敗するコマンドに両方の変数を設定します:

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` は Neat Library の設定ではありません。通常の運用では詳細な出力を
無効にしておいてください。詳細な出力は短時間の診断実行を想定したもので、認識された認証情報のフィールドは秘匿化されますが、
デプロイ固有のパスやメディアのアドレスが含まれる場合があります。

## デバッグワークフロー

1) まず `GraphReport.error_code` を取得し、分類体系に従って障害を分類します。
2) 具体的なコンテキストと組み込みのヒントを得るために、`GraphReport.repro_note` を取得します。
3) パイプラインのテキストを取得します: `Graph::describe_backend()` または `last_pipeline()`。
4) 構造化された診断情報を取得します: `MeasureReport::to_text()` または `NeatError::report()`。
5) `GraphReport.bus` を調べて、最初の終端的な `ERROR` の発生元と詳細を確認します。
6) ランタイムが停止またはタイムアウトする場合は、境界/要素プローブを有効にして、フローが止まる箇所を特定します。

推奨されるサポートバンドル:
- `error_code`
- `repro_note`
- 完全な `pipeline_string`
- 最初の 3～5 件の終端的なバスエラー（`GraphReport.bus`）
- run/validate で使用した環境変数の上書き

## 顧客向けのグラフパフォーマンスアーティファクト

スループット/レイテンシー/電力のレポートには、グラフ実行の JSON エクスポートを使用することを推奨します:

```cpp
RunOptions opt;
opt.enable_board_power();        // graph-level power when supported by the board/SOM
Run run = graph.build(opt);

// run your normal push/pull loop inside a measurement window, then:
auto report = run.start_measurement().stop();
std::cout << report.to_text();
```

エクスポートでは、スコープが明示的に区別されます:

- `run.graph_metrics.throughput_fps` と `run.graph_metrics.power` は、グラフレベルの主要指標です。
- `run.node_metrics[]` にはノード/プラグインのレイテンシーのみが含まれます。ノード/プラグインの電力は意図的に含まれていません。
- `latency_semantics` と `aggregation` は、値が実行期間全体のものか、測定ウィンドウ内の差分かを示します。
- `plugin_metrics_unattributed[]` は、ちょうど 1 つのノードに対応付けられなかったカーネル/プラグインの行を保持します。

測定ウィンドウを使うには、`Run::start_measurement()` を使用し、返された `MeasureReport` を
`run_to_json(run, report, ...)` / `save_run_json(run, report, ...)` に渡します。測定ウィンドウのノードの
`min_ms`/`max_ms` は、ウィンドウローカルのカウンターがないと累積の最小/最大カウンターを正確に
差し引くことができないため、利用不可としてマークされます。

電力に関する注意: 現在の DVT ボードではオプションの受け渡しと JSON の形式を検証できますが、そのワット数の
読み取り値は数値として信頼できるものとは見なされません。電力値の検証には、SOM ハードウェアを
対象プラットフォームとしています。

## よくある失敗 → 対処法

| 症状 | 考えられる原因 | 対処 |
| --- | --- | --- |
| `missing ... plugin` | GStreamer プラグインが見つかりません | `GST_PLUGIN_PATH` を確認し、`gst-inspect-1.0 <plugin>` を実行してください |
| `appsink 'mysink' not found` | 終端の `Output()` がありません | `Output` が run/build パイプラインの最後のノードであることを確認してください |
| `caps_override is set; renegotiation disabled` | caps が固定されています | `caps_override` を削除するか、入力 caps を固定したままにしてください |
| `tensor caps change not supported` | ランタイムでのテンソルの形状/dtype の変更 | テンソルの形状/dtype を一定に保ってください（再ネゴシエーションなし） |

構造化されたプラグインエラーと対処に役立つヒントについては、
[トラブルシューティング](/reference/troubleshooting) を参照してください。

## プラットフォームのランタイム復旧

Platform 3.0.0 では、利用できないディスパッチャーは調査すべきエラーです。従来のサービスを起動する要求ではありません。Core はディスパッチャーのエラーに応じて MLA メモリを初期化したり、リモートプロセッサーをリセットしたりしません。従来の復旧コードとスクリプトは削除されています。プラットフォームが承認する復旧手順を使用してください。

ドライバーが完了状態を不明と報告した場合は、その DMA バッファーと元のプールからの借用を保持してください。ファイル記述子を閉じること、アプリケーションを停止すること、サービスを再起動することは、ハードウェアによるメモリアクセスが停止した証拠にはなりません。障害情報を収集し、再試行の前にプラットフォームが承認する復旧手順を実行してください。

### Core と Internals の組み合わせを維持する

対応する B1157 Internals パッケージを使用して Core をビルドし、インストールしてください。Core はランタイムプロファイル、カーネルのソースリビジョン、SDK sysroot の記録を、公開 C++ ABI のバージョンとは別に確認します。公開 ABI が同じでも、古いパッケージは互換の代替品ではありません。

インストーラーは、パッケージを変更する前に同梱の `neat-runtime` プロファイルと対応する `neat-gst-plugins` バージョンを確認します。プラットフォームチェックの上書き設定でも、ランタイムの組み合わせの確認は回避できません。確認が失敗した場合は、対応するバンドルを入手してください。記録を書き換えたり、古いランタイムを強制的にインストールしたりしないでください。

ボードにインストールする前に、CVU またはハードウェアコーデックを使用するアプリケーションを停止してください。完全なインストーラーは、すべてのパッケージをインストールした後で配置済みの EV74 ファームウェアを有効にします。デバイスが開かれている間は EV74 のリセットを拒否します。`NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=OFF` を設定すると、ファームウェアを配置済みのままにして、後から `sudo /usr/libexec/sima-neat-firmware/install.sh --activate` で有効にできます。
