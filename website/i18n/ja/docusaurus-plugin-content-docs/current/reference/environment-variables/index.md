---
title: "環境変数"
description: "ランタイムとビルド環境の環境変数"
sidebar_position: 6
slug: /reference/environment-variables
---

# 環境変数

このページには、ランタイム、ビルダー、および SDK ツールで使用される環境変数がまとめられています。これらは主にデバッグ/診断用の設定であり、ほとんどのユーザーはトラブルシューティングを行う場合を除き、無視しても問題ありません。

詳細なリストについては、[環境変数のリスト](./inventory) を参照してください。

> 注：一部のパラメータは内部用またはテスト専用であり、変更される可能性があります。ここではそれらも含めて記載しています。
> なぜなら、それらは現在のコード実行パスに存在しているからです。

## SDKのウェブアクセス

- `NFS_SERVER_HOST_IP=<address>` — リモートを構築するために使用されるSDKホストのアドレス。
  Insightと、ブラウザベースの VS Code の URL。通常は、`sima-cli sdk setup`によって提供されます。
- `CONTAINER_HOST_IP=<address>` — リモートホストのアドレスを取得するための従来のフォールバック方法。
  `NFS_SERVER_HOST_IP` が利用できない場合。
- `OPENVSCODE_SERVER_HTTPS_PORT=<port>` — ブラウザ版 VS Code 用の HTTPS ポート。
  設定されていない場合、`neat` は VS Code の URL を表示しません。
- `OPENVSCODE_SERVER_TOKEN=<token>` — ブラウザベースの VS Code アクセストークン。
  sima-cliによって生成される値は、URLで安全に使用できます。このトークンを含むURLは、認証情報として扱い、共有しないでください。
- `OPENVSCODE_WORKSPACE=<path>` — ブラウザ版 VS Code で開かれたワークスペース
  （デフォルトの`/workspace`）。
- `OPENVSCODE_SERVER_WITHOUT_TOKEN=1` — ブラウザベースの VS Code を実行し、公開します。
  トークン認証なしで使用します。信頼できるローカル環境でのみ使用し、信頼できないネットワークからサービスにアクセスできる場合は、絶対に有効にしないでください。

## 統合デバッグプロファイル（推奨）

- `SIMA_DEBUG_PROFILE=<components>` — 共通の診断機能に対する、統合されたデバッグ有効化スイッチ。
  - コンポーネント：`pipeline`、`graph`、`gst`、`appsink`、`inputstream`、または`all`。
  - 複数のコンポーネントは、カンマまたはスペースで区切ることができます（例：`pipeline,gst,inputstream`）。
- `SIMA_DEBUG_LEVEL=<0..3>` — 統合プロファイルで使用されるデバッグの詳細レベル（デフォルトは`1`）。
  - `0`: 無効
  - `1`: コア デバッグログ
  - `2`: 詳細な診断情報／バッファーレベルのトレース
  - `3`：最大の詳細度

従来の変数ごとのデバッグ切り替え機能は引き続き動作し、明示的に設定された場合は、プロファイル設定のデフォルトを上書きします。

## コアビルド/実行

- `SIMA_PIPELINE_STRING_DEBUG=1` — ビルド時に最終的な gst-launch 文字列を出力します。
- `SIMA_PIPELINE_STATE_DEBUG=1` — 追加のステータス変更ログ。
- `SIMA_PIPELINE_TEARDOWN_DEBUG=1` — パイプラインの終了処理ステップのログを出力します。
- `SIMA_PIPELINE_DRAIN_BEFORE_TEARDOWN_MS=<ms>` — 破棄前のドレイン時間（デフォルトは1500ミリ秒）。
- `SIMA_PIPELINE_DRAIN_MIN_OUTPUTS=<n>` - パイプラインを終了する前に排出する必要がある最小の出力数（デフォルトは1）。

## GStreamer の初期化と抑制

- `SIMA_ALLOW_GST_INIT=1` — 既に初期化されている場合でも、手動による`gst_init`を許可します。
- `SIMA_GST_SUPPRESS_JSON_WARNINGS=0/1` — JSON警告を抑制します（デフォルトはtrue）。
- `SIMA_GST_SUPPRESS_GOBJECT_ASSERTS=0/1` — GLib のアサートログの出力を抑制します（デフォルトは true）。
- `SIMA_GST_SUPPRESS_DEVICE_LOGS=0/1` — デバイスログの出力を抑制します（デフォルトは有効）。

## GStreamer のタイムアウト

- `SIMA_STATE_CHANGE_TIMEOUT_MS=<ms>` — パイプラインの状態変更のタイムアウト時間（デフォルトは15000ミリ秒）。
- `SIMA_GST_TEARDOWN_TIMEOUT_MS=<ms>` — クリーンアップのタイムアウト時間（デフォルトは2000ミリ秒）。
- `SIMA_GST_TEARDOWN_REAPER_MS=<ms>` — ウォッチドッグのタイムアウト時間（デフォルトは250ミリ秒）。
- `SIMA_GST_TEARDOWN_ASYNC=1` — 非同期でのクリーンアップ。
- `SIMA_GST_POLL_SLICE_MS=<ms>` — アプリシンクからのデータ取得に使用するポーリング間隔（デフォルトは200ミリ秒）。
- 推奨される API 設定項目：
  - `ValidateOptions.preroll_timeout_ms` — validate() のプリロールタイムアウト。
  - `RunOptions.input_timeout_ms` — build()/run() の入力モードにおけるタイムアウト時間。
- 従来のフォールバック環境変数は、オプションを渡せない場合にのみ使用してください。
  - `SIMA_GST_VALIDATE_TIMEOUT_MS=<ms>` — validate() のタイムアウト時間（デフォルトは 2000/10000）。
  - `SIMA_GST_RUN_INPUT_TIMEOUT_MS=<ms>` — run() 関数の入力タイムアウト時間（デフォルトは 10000 ミリ秒）。

## 診断機能とプローブ

- `SIMA_GST_DOT_DIR=<dir>` — パイプラインの失敗時やデバッグのために、DOT形式のグラフを出力します。
- `SIMA_GST_BOUNDARY_PROBES=1` — 境界流プローブを取り付けます。
- `SIMA_GST_STAGE_TIMINGS=1` — ステージタイミングプローブ。
- `SIMA_GST_ELEMENT_TIMINGS=1` — 要素タイミングプローブ。
- `SIMA_GST_FLOW_DEBUG=1` — 要素フローの調査。
- `SIMA_GST_ENFORCE_NAMES=1` — ビルド時に名前の規則を強制する。
- `SIMA_GST_OPTIONS_DEBUG=1` — ビルド中に GStreamer のオプションをログに出力します。
- `SIMA_GST_BUFFER_DEBUG_LIMIT=<n>` — バッファーのデバッグ出力の制限。
- `SIMA_GST_DETESS_INPUT_DEBUG=1` — detess入力デバッグ。
- `SIMA_GST_DETESS_OUTPUT_DEBUG=1` — detess出力デバッグ。
- `SIMA_GST_DETESS_POOL_DEBUG=1` — detessプールのデバッグ。
- `SIMA_GST_APPSINK_BUFFER_DEBUG=1` — アプリシンクのバッファーに関するデバッグ情報。
- `SIMA_GST_ALL_BUFFER_DEBUG=1` — 詳細なバッファーデバッグ。
- `SIMA_GST_RUN_INSERT_BOUNDARIES=1` — run() の実行中に境界を挿入します。
- `SIMA_GST_VALIDATE_INSERT_BOUNDARIES=1` — validate() 時に境界線を挿入します。

## ディスパッチャー / ランタイム

- `SIMA_DISPATCHER_TRACE=1` — ディスパッチャーの処理ステップを追跡します。
- `SIMA_DISPATCHER_AUTO_RECOVER=0/1` — 自動復旧ディスパッチャー（デフォルトは有効）。
- `SIMA_ASYNC_TPUT_DIAG=1` — 非同期スループット診断。
- `SIMA_ASYNC_WARMUP=<n>` — 非同期ウォームアップフレーム。
- `SIMA_PERF_POWER=1` — パフォーマンス測定のシナリオで、SOM PMICレールへの電力供給を有効にします。
- `SIMA_PERF_POWER_INTERVAL_MS=<ms>` — 電力サンプリング間隔（デフォルトは100）。
- `SIMA_PULL_TIMEOUT_DIAG=0/1` — プル操作のタイムアウトに関するレポートを出力します（デフォルトは有効）。
- `SIMA_STAGE_DEBUG=1` — StageRun のデバッグログ。

## 入力ストリーム / サンプルデバッグ

- `SIMA_INPUTSTREAM_DEBUG=1` — 詳細な InputStream ログ。
- `SIMA_INPUTSTREAM_WARN=1` — InputStream イベントに関する警告。
- `SIMA_INPUTSTREAM_POLL_MS=<ms>` — InputStream のポーリング間隔（デフォルトは 50 ミリ秒）。
- `SIMA_INPUTSTREAM_DOT_ON_TIMEOUT=1` — タイムアウト時に DOT をダンプします。
- `SIMA_INPUTSTREAM_META_DEBUG=1` — GstSimaMeta の詳細をログに出力します。
- `SIMA_INPUTSTREAM_ALLOC_DEBUG=1` — メモリ割り当てデバッグ。
- `SIMA_INPUTSTREAM_PUSH_TIMING=1` — プッシュタイミングのログ。
- `SIMA_INPUTSTREAM_PREFLIGHT_RUN=1` — InputStream のための事前チェック実行。
- `SIMA_SAMPLE_DEBUG=1` — 変換のサンプルをログに記録します。
- `SIMA_SAMPLE_BYTES=1` — ログに記録するサンプルバイトサイズ。
- `SIMA_SAMPLE_FORCE_BUNDLE=1` — デバッグ用の強制バンドル出力。
- `SIMA_NEAT_CAPS_TRACE=1` — テンソルのトレースによるキャップの導出。

## 前処理 / 検出 / 配線

- `SIMA_PREPROC_DEBUG_CONFIG=1` — プリプロセッサ設定の連携状況をダンプします。
- `SIMA_KEEP_DETESS_CONFIG=1` — detess設定の出力を保持します。
- `SIMA_DETESS_ASSERT_ON_ZERO=1` — ゼロ値に対する detess 出力の検証。
- `SIMA_CLAMP_DETESS_NUM_BUFFERS=1` — クランプ検出バッファー数。
- `SIMA_DISABLE_SYNC_NUMBUFFERS_CVU_MLA=1` — 同期バッファー数の制限を無効にします。

## モデル（従来の環境変数名を引き継ぎ）
- `SIMA_MLA_NEXT_CPU=<domain>` — MLAのnext_cpuを上書きします。
- `SIMA_MPK_EXTRACT_ROOT=<dir>` — モデルアーカイブの読み込みに使用するベースディレクトリ。絶対パスに解決されます。
  パスはプロセスごとに一度だけ使用されるため、抽出された JSON に書き換えられたパスは、作業ディレクトリに依存することはありません。権限：書き込み可能でない場合、フォールバックするのではなく、読み込みに失敗します。設定されていない場合、ベースは、マウントされた NVMe ファイルシステムの最初の書き込み可能な候補、`/data`、`TMPDIR`、次に作業ディレクトリになります。NVMe 候補は、データマウント上の書き込み可能な `/dev/nvme*` ブロックデバイスである必要があります。root、`/boot`、`/efi`、およびその他のシステムマウントは除外され、vfat/ISO ファイルシステムも同様です。これらのチェックは NVMe の検出に適用されます。`/data`、`TMPDIR`、および作業ディレクトリのフォールバックは、通常のファイルシステム配置を維持します。NVMe は、容量、予測可能な配置、および eMMC への書き込みを回避するために優先されます。これはデコード速度の向上ではありません。この変数は、出力が書き込まれる場所を選択するものであり、デコードは CPU に依存します。

選択は、空き容量ではなく、書き込み可能性に基づいて行われます。`.tar.gz` は、展開後のサイズをどちらの方向にも制限しないため、デコード前に必要な容量を事前に知ることはできません。したがって、容量は、展開中および抽出前のマニフェストから、チャンクごとに適用されます。したがって、適切な NVMe は無条件で使用され、空き容量がない場合は、eMMC にフォールバックするのではなく、`output_storage_unavailable` で読み込みに失敗します。そのファイルシステムの空き容量、または `SIMA_MPK_EXTRACT_ROOT` が別の場所を指している場合が、その解決策です。
- `SIMA_MPK_CLEANUP_EXTRACTED=0/1` — 通常終了時に、プロセスごとに抽出されたモデルアーカイブデータを削除します（デフォルトは`1`）。
  クリーンアップが有効になっている場合、各プロセスは自身の `proc_<pid>` ルートに展開し、終了時にそれを削除します。クリーンアップが無効になっている場合、そのプロセスルートは検査のために保持され、古いルートのガベージコレクションから除外されます。別のプロセスによって自動的に検出または再利用されることはありません。不要になった場合は、手動で削除してください。

`Model` は、`etc`、`lib`、および `share` を含む、すでに整理されたパッケージルートも受け入れます。そのディレクトリは、解凍またはコピーなしでそのまま使用されます。呼び出し元は、そのライフサイクルを管理し、モデルが使用されている間は変更しないようにする必要があります。`tar -xzf` によって生成されたフラットなディレクトリは、整理されたパッケージではなく、直接受け入れられません。
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0/1` — 古い不要なものを削除`proc_*` 起動時に抽出するルート (デフォルト) `1`).
- `SIMA_MODEL_TAR=<path>` — 例やテストで使用される、ベースモデルパックのパス。
  モデルごとの上書き設定（`SIMA_RESNET50_TAR`、`SIMA_YOLO_TAR`など）は、引き続き優先されます。
- `SIMA_MPK_EXTRACT_MIN_FREE_BYTES=<bytes>` — ステージング時に確保しておく最小限の空き容量。
  モデルアーカイブの抽出（デフォルトは16MiB）。
- `TMPDIR=<dir>` — 上記のベースとして検討されたのは、比較的遅い段階であり、直接的に使用されたのは
  独自のベースを選択しない呼び出し元によってステージングが行われます。モデルのロードは、これ以上独立してステージングされません。展開されたスナップショットと抽出されたパッケージは、選択されたベースを共有します。メタデータのみの検査を含むすべてのロードでは、`.tar.gz` が一度だけ、プライベートディレクトリに解凍されます。抽出中、そのファイルシステムには、展開されたスナップショット、パッケージ、および `SIMA_MPK_EXTRACT_MIN_FREE_BYTES` のための十分な空き容量が必要です。150MB のリファレンスパックのスナップショットは約 354MB です。ローカルファイルシステムを使用してください。空き容量を読み取れない場合、ロードは失敗します。これは、ネットワークマウントが利用できなくなった場合に発生する可能性があります。ステージングディレクトリは、ロードが正常に完了した場合、および失敗した場合に削除されます。予期しない強制終了や電源喪失が発生すると、ステージングディレクトリが残ってしまう可能性があります。

`SIMA_MPK_EXTRACT_MIN_FREE_BYTES` によって設定された空き容量の予約は、可能な限り実行されます。これは、展開中にファイルシステムの報告する空き容量に対して、チャンクごとにチェックされます。そのため、無関係な同時書き込みによって、チェックの間にファイルシステムの空き容量が不足する可能性があり、その場合、書き込まれたバイト数と使用されていたパスとともにロードが失敗します。

## RTSP / H264

- `SIMA_H264_SDP_DUMP=<path>` — H.264 SDP をファイルにダンプします。
- `SIMA_H264_SPS_FIXUP_STREAM=<path>` — ストリーム内のSPSを修正します。

## テスト / 内部フック

- `SIMA_TENSOR_MAPFAIL_DEBUG=1` — テンソルのマッピング失敗をログに記録します。
