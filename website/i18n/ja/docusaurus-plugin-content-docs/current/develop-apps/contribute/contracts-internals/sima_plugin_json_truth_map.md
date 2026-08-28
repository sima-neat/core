---
title: "SIMAプラグインのJSON検証マップ"
description: "モデルのパイプラインにおけるSIMAステージで使用される、固定されたJSONフィールドのマッピング"
sidebar_position: 2
slug: /develop-apps/contribute/sima_plugin_json_truth_map
---

# SIMA プラグイン JSON 参照マップ（固定版）

_最終更新日：2026-02-17_

このドキュメントは、モデルパイプラインの SIMA ステージで使用される JSON フィールドの使用を固定し、削除を制御可能にし、テスト可能にします。

## 1. 固定範囲とプラグインマトリックス

### 1.1 範囲内（モデルパイプライン）

- `simaaiprocesscvu` は、次のように使用されます。
  - 前処理ステージ (`kernel=preproc`)
  - 量子化/テッセレーションステージ (`kernel=quanttess`)
  - detess/dequant のための後処理ステージラッパー（モデルシーケンスでは `kernel=detessdequant`、バックエンド要素は `src/nodes/sima/DetessDequant.cpp` の `simaaiprocesscvu` のまま）
- `simaaiprocessmla`
- `simaaiboxdecode`（汎用ボックスデコード）
- `tmp/gst/*` 内の detess/dequant/tess ペイロードステージ：
  - `detessdequant`
  - `detessellate`
  - `quantize`
  - `slicedequant`

### 1.2 範囲外

この参照マップの範囲外となるのは、汎用 CVU アプリプラグインとカスタムグラフユーティリティであり、これには以下が含まれますが、これらに限定されません。

- `overlay`、`genericrender`、`argmax`、`nms*`、`groupkeypoints`、`distancecalculation`、`cv_process`、`cvresize`、`fastbev*`、`PyGast-plugins/*`、非推奨のプラグイン、およびカスタムアプリ/テストのスクラッチコード。

### 1.3 対象となるソースツリー

静的抽出では、要求された両方のツリーが対象となりました。

- `tmp/gst_plugins_source/gst/*`
- `tmp/gst/*`

ミラーチェック：

- 両方のツリーで同じ：`genericboxdecode`、`detessdequant`、`detessellate`、`quantize`、`slicedequant`
- 異なる：`processcvu`、`processmla`

### 1.4 プラグインマトリックス

| プラグイン / ステージ | 現在必要な JSON キー (現在のコードパス) | 推論可能なキー | ランタイム プロパティ (静的な JSON であってはならない) | MLA 専用キー |
|---|---|---|---|---|
| `simaaiprocesscvu` (前処理/量子化/後処理ラッパー) | 推論優先: 利用可能な場合は `ConfigManager::getBuffers()` から配線。JSON `input_buffers`/`output_memory_order` はフォールバック専用。 | `input_width`/`input_height` は、グラフ 200/202 の caps/ランタイムから取得可能。配線配列は、CM メタデータから合成可能。 | ランタイムの次元はフレームごとに再ネゴシエートされる。フレームワークのビルドは、ステージごとの JSON 配線フィールドを書き換えない。 | 量子化/テッセレーションと後処理パスは、間接的に MLA テンソルの形状フィールド (`input_depth`、`slice_*`) を消費する。 |
| `simaaiprocessmla` | `simaai__params`、`model_path`、`batch_size`。`outputs[*]` が推奨されるが、出力形状フィールドからセグメント サイズを推論できる場合は必須ではなくなった。 | 出力セグメント サイズは、`output_*`/`slice_*` + データ型から推論可能。 | `input_segment_name` はオプションのランタイム配線補助。モデル パスは、パッケージから派生可能。 | `outputs`、`data_type`、`output_*`、`slice_*`、量子化パラメータ。 |
| `simaaiboxdecode` | 実質的にバックエンド構成ローダーによって必要: `buffers.output.size`、`memory.next_cpu`、`system.out_buf_queue`。クラス数の解決は、現在の実装バージョンに依存する。 | `num_classes` は、`input_depth`/`slice_depth` + `num_in_tensor` + `decode_type` (新しいソース ロジック) から推論可能。 | `buffers.input[*].name` は、上流から再配線される。閾値/トップ K は、ランタイムの調整ノブであることが多い。 | `input_*`、`slice_*`、`data_type`、`num_in_tensor`。 |
| `detessdequant` (レガシーのスタンドアロン GST 要素) | `simaai__params` とパーサー フィールド: `orig_img_width`、`orig_img_height`、`frame_width`、`frame_height`、`num_in_tensor`、`next_cpu`、`no_of_outbuf`、`out_sz`、`input_*`、`slice_*`、`q_scale`、`q_zp`。 | プラグイン内にはない。より高レベルの形状推論は、`StageConfig` に存在する。 | 上流のバッファー名/CPU ルーティングは、ラッパー フローでランタイムで実行される。 | `input_*`、`slice_*`、`q_scale`、`q_zp`、`num_in_tensor`。 |
| `detessellate` ペイロード (`tmp/gst/detessellate`) | `de_tess.*` またはルート/静的契約と同等のもの (`input_*`、`slice_*`/`output_*`) を受け入れる。`buffers.input[0].offset` はオプション (デフォルトは 0)。 | テンソルの数と次元は、マニフェスト ステージの静的フィールドから合成可能。 | 入力名/パスは、静的ではなく、ランタイムで配線される必要がある。 | 形状/スライス契約。 |
| `quantize` ペイロード (`tmp/gst/quantize`) | `quant_scale`、`zero_point` (JSON フォールバック)。 | 入力要素数は、受信バッファー サイズから推論される。 | 現在は、まず上流のランタイム メタデータから `q_scale`/`q_zp` メタデータを消費し、次に JSON フォールバックを使用する。 | n/a (汎用量子化)。 |
| `slicedequant` ペイロード (`tmp/gst/slicedequant`) | 形状のフォールバックには、セクション (`slice_dequant`/`simaai__params`/root) を読み取る。量子化 JSON はフォールバック専用。 | まずランタイム メタデータから量子化し、次元はマニフェストの静的契約から合成可能。 | ランタイム メタデータ (`q_scale`/`q_zp`) が推奨される転送方法。 | MLA 出力形状/量子化契約。 |

### 1.5 マニフェスト コンテキスト転送 (現在)

- パイプラインコンテキストタイプ: `sima.model.manifest.v1`
- ABI互換のプラグインアクセス: `manifest_accessor_v1` ( `include/gst/SimaPluginStaticManifestAbi.h` 内)
- ステージ検索キー:
  - `element_name` (デフォルト)
  - `logical_stage_id` ( `stage-id` または `stage_id` パイプラインプロパティから。設定されている場合)
- レガシーの `manifest_json` 文字列は、移行時のフォールバック用に残されています。
- フレームワークのノード/モデルフラグメントビルダーは、SIMAモデルパスプラグイン用に `stage-id=<element-name>` を出力するようになりました。これにより、追加の名前変換が適用された場合でも、論理的な検索が決定的に行われます。

## 2. 必要なキーの真理値マップ

### 2.1 静的抽出方法

明示的なアクセスポイントから抽出:

- `json["..."]`
- `contains("...")`
- パーサーヘルパー (`parser_get_int`、`parser_get_double_array` など)

キーとなる証拠の場所:

- `tmp/gst/processcvu/gstsimaaiprocesscvu.cpp:1667`
- `tmp/gst/processmla/gstsimaaiprocessmla.cpp:579`
- `tmp/gst/genericboxdecode/payload.cpp:61`
- `tmp/gst/detessdequant/gstsimaaidetessdequant.cpp:276`
- `tmp/gst/detessellate/detessellate.cpp:361`
- `tmp/gst/quantize/payload.cpp:124`
- `tmp/gst/slicedequant/payload.cpp:57`

ランタイムの配線/推論の証拠:

- `src/nodes/sima/Preproc.cpp:245`
- `src/nodes/sima/DetessDequant.cpp:238`
- `src/nodes/sima/SimaBoxDecode.cpp:158`
- `src/pipeline/runtime/StageConfig.cpp:296`
- `src/pipeline/runtime/StageConfig.cpp:411`

### 2.2 使用される動的な障害注入方法

登録されたプラグイン (`simaaiprocesscvu`、`simaaiprocessmla`、`simaaiboxdecode`、`detessdequant`):

- ベースライン: `gst-launch-1.0 ... num-buffers=0`
- 1つのキーを一度に操作 (フィールドを削除)
- スタートアップ/ランタイムの障害動作とメッセージを記録
- 注: 動的な結果は、このホストで現在登録されているランタイムプラグインを反映します。

登録されていないペイロードステージ (`detessellate`、`quantize`、`slicedequant`):

- このランタイムでは、直接的なGST要素は利用できません (`gst-inspect-1.0` は、見つからないことを報告します)
- したがって、動的なキーの削除は、このパスに対して静的/ソース分類に限定されました。

### 2.3 整理されたマップ (固定)

### `simaaiprocesscvu`

- 必須:
  - CMバッファー推論の成功、または次の情報を含むJSONフォールバック:
    - `input_buffers`
    - `output_memory_order`
    - 各入力の `memories[*].segment_name`
    - 各入力の `memories[*].graph_input_name`
- オプション/デフォルト化可能:
  - `graph_name`
  - `input_width`、`input_height` (オプションのJSON次元)
- 重複/派生:
  - `input_buffers[*].name` (ランタイムで配線)
  - マニフェストコンテキストからの `sink_pad_tensor_index_map` が、決定的なマルチ入力マッピングのために優先されます。
  - プリプロセスの次元は、caps/ランタイムから取得されます。
- デバッグ専用:
  - `debug` スタイルのフィールドは、実行には必要ありません。

動的な証拠:

- CM推論が失敗し、JSON配線が欠落している場合、バスエラーで起動が失敗します。
- CM推論が成功した場合、`input_buffers`/`output_memory_order` を省略できます。
- コンテキストがモデル管理のマルチ入力ステージを示し、`sink_pad_tensor_index_map` が欠落しているか曖昧な場合、バスエラーで起動が失敗します。

### `simaaiprocessmla`

- 必須:
  - `simaai__params`
  - `model_path`
  - `batch_size`
  - `outputs[*].name`
  - `outputs[*].size`
  - `batch_sz_model` (ただし、`batch_size != 1` の場合)
- オプション/デフォルト値あり:
  - `input_segment_name`
- 重複/派生:
  - 出力次元/型は、上位レイヤーのモデルメタデータから推測可能
- デバッグ専用:
  - 該当なし

動的な証拠:

- `model_path` を削除 -> キャッチされない `nlohmann::json` 型エラー (`ec=134`)
- `batch_size=2` と `batch_sz_model` を削除 -> キャッチされない型エラー (`ec=134`)
- `outputs` を削除 -> スタートアップは `num-buffers=0` で正常に完了するが、ランタイム (`num-buffers=1`) で SIGSEGV スピンパスが発生する

### `simaaiboxdecode`

- 必須 (現在のランタイムの動作):
  - `buffers.output.size`
  - `memory.next_cpu`
  - `system.out_buf_queue`
- オプション/デフォルト値あり (実装バージョンによって異なる):
  - `num_classes` は、新しいソースでは推測/フォールバックできる可能性があるが、現在のランタイムでは警告のみが表示される
  - `decode_type` は、現在のランタイムでは型不一致の警告にダウングレードされる可能性がある
- 重複/派生:
  - `buffers.input[*].name` は、ランタイムで固定
  - `num_classes` は、既知のデコードファミリーのテンソル形状 (`input_depth`/`slice_depth`) から派生可能
- デバッグ専用:
  - `system.debug`、`system.dump_data`

動的な証拠:

- `memory.next_cpu` を削除 -> キャッチされない型エラーでアボート (`ec=134`)
- `system.out_buf_queue` を削除 -> キャッチされない型エラーでアボート (`ec=134`)
- `num_classes` を削除 -> 致命的ではない `JSON type mismatch` 警告 (`ec=0`)

### `detessdequant` (レガシーのスタンドアロン GST プラグイン)

- 必須:
  - `simaai__params` オブジェクト
  - パーサーキー: `orig_img_width`、`orig_img_height`、`frame_width`、`frame_height`、
    `num_in_tensor`、`next_cpu`、`no_of_outbuf`、`out_sz`、
    `input_height`、`input_width`、`input_depth`、
    `slice_height`、`slice_width`、`slice_depth`、
    `q_scale`、`q_zp`
- オプション/デフォルト値あり:
  - 現在のコードには該当なし
- 重複/派生:
  - 一部のフレーム/元のサイズフィールドは、メタデータレベルであり、派生可能である
- デバッグ専用:
  - `debug`、`dump_data`、`inpath`、`ibufname`、`n_request` など

動的な証拠:

- `simaai__params` を削除 -> SIGSEGV スピンパス (タイムアウト)
- `num_in_tensor` を削除 -> SIGSEGV スピンパス (タイムアウト)

### `detessellate` ペイロード

- 必須:
  - 解決された入力/スライス テンソルフィールド (`de_tess.*` またはルート/静的契約から合成されたキー)
- オプション/デフォルト値あり:
  - `buffers.input[0].offset` (デフォルト値は `0`)
  - `num_in_tensor` (省略された場合に、ベクトルサイズから派生)
- 重複/派生:
  - 形状ベクトルは、マニフェストステージの静的テンソルから派生可能
- デバッグ専用:
  - 該当なし

### `quantize` ペイロード

- 必須:
  - `quant_scale`
  - `zero_point`
- オプション/デフォルト値あり:
  - 現在のコードには該当なし
- 重複/派生:
  - テンソル要素数は、入力バイトサイズから派生
- デバッグ専用:
  - 該当なし

### `slicedequant` ペイロード

- 必須項目：
  - 量子化解除のための量子化パラメータ（`q_scale`、`q_zp`）は、ランタイムメタデータまたはフォールバック設定から解決される。
  - テンソルのスライス次元（`input_*`、`output_depth`/`slice_depth`）は、セクション/ルート/静的契約の合成から解決される。
- オプション/デフォルト値：
  - 量子化と形状キーのエンコーディングにおけるスカラーとベクトルの使い分け。
- 重複/派生：
  - 量子化パラメータは、ランタイムメタデータから優先的に取得され、JSONはフォールバック専用として残る。
- デバッグ専用：
  - 該当なし。

## 3. 制御された削除ゲート（この固定マップから）

JSONフィールドの削除には、以下のすべてを含める必要がある。

1. フィールドに対するこの真理マップの分類を更新する。
2. 障害注入テストケースを追加/更新する。
   - スタートアップ時の障害は明示的に行う必要があり（バスエラー）、クラッシュしてはならない。
3. 推論可能な/派生したフィールドの場合：
   - まず推論を実装する。
   - キャップ/プラグインオプションのフォールバックを維持する。
   - 必要な場合に限り、JSONを最後のフォールバックとして残す。
4. MLA専用のJSONを最小限に保つ。
   - 未解決のMLAメタデータ（形状/サイズ/量子化パラメータ）のみを残す。
5. 厳密なCIゲートを有効にしておく。
   - `unit_sima_plugin_manifest_strict_model_pipeline_test`
   - `unit_sima_plugin_manifest_strict_fallback_test`
   - および、`.github/workflows/vulcan-ci.yml`内の対応するVulcan CIテストレーン。

## 4. 現在発見されたリスク

- 複数のキーパスが存在せず、バスエラーではなく、キャッチされない`nlohmann::json`例外またはSIGSEGVが発生してプログラムが停止する。
- `detessdequant`のレガシーパスは、パーサーキーが存在しない場合にクラッシュしやすい。
- `slicedequant`は、JSONを完全に無視し、コンパイルされた定数を使用する。
- ランタイム/ソースの乖離は、再構築された`.so`ファイルが`tmp/gst/*/build`から`deps/gst-plugins`にコピーされない場合に、再び発生する可能性がある。
