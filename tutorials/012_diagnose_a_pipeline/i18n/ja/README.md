# 012 パイプラインの診断とプロファイリング

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | <10 minutes |
| Model | None |
| Labels | diagnostics, debugging, observability |

## Concept

3つのチェック（`graph.validate()`、1回の測定による`run.run()`、および`MeasureReport`による診断）を使用してパイプラインをトリアージし、詳細なデバッグを行う前に、パイプラインが正しく構成されているかどうか、およびそのパフォーマンスがどのようであるかを判断します。

## Walkthrough

パイプラインが正常に動作しない場合、すぐに要素レベルのデバッグを開始したくなるでしょう。この章では、より手軽な最初のステップを教えます。それは、以下の3つの質問に順番に答える、再現可能なトリアージ処理です。*グラフの契約は有効ですか？ 1回の実行は成功しますか？ ランタイム診断は何を教えていますか？* これにより、数秒でほとんどの誤った設定を検出し、数時間にも及ぶデバッグセッションになる前に問題を解決できます。また、これは、第004章ですでに知っている、同じ最小限の入力→出力グラフで機能します。

この章の終わりには、グラフの契約を検証し、単一の測定フレームを実行し、パイプラインが正常に動作するかどうかを示す測定レポートを出力できるようになります。

### 契約を検証する {#step-validate-graph}

`validate()`は、`build()`の*前*に実行される、契約レベルのチェックです。ノードの順序、上限、およびバックエンドの解析パスを、データをストリーミングせずに実行し、標準的な`error_code`を含むレポートを返します。空の/`ok`コードは、グラフの構造が健全であることを意味します。それ以外の場合は、エラーを分類（以下に示すエラーの分類を参照）し、どこを確認すればよいかを把握できるようにします。最初にこれを行うことで、そもそも構築されないグラフでランタイムの動作をデバッグする時間を無駄にすることはありません。

### 単一の測定フレームを実行する {#step-run-with-measurement}

次に、`start_measurement()`ウィンドウ内で、単一の決定的なフレームを構築して実行します。`output_memory = Owned`は、所有された出力バッファーを要求するため、呼び出し後に結果が有効な状態を維持します。1つのフレームで十分です。成功した場合、パイプラインは正常に動作します。例外が発生した場合、例外には、`validate()`と同じように分類できる構造化されたレポートが含まれます。

### ランタイム診断を読み取る {#step-read-diagnostics}

1回の実行の記録に基づいて、`MeasureReport`は、パイプラインの健全性を要約します。カウンター（`inputs_enqueued`、`outputs_pulled`、ドロップ）、エンドツーエンドのレイテンシー、ノードメトリック、プラグイン/カーネルのタイミング、エッジのタイミング、およびオプションの電力などが含まれます。`MeasureReport::to_text()`は、[In Practice](#in-practice)で説明されているプローブとDOTグラフに移行する前に、取得するベースラインです。

## Run

実行すると、検証コードと測定レポートがstdoutに出力されるはずです。**Neatのインストールルート**（`share/`と`lib/`が含まれるディレクトリ）から、**Python**と**C++（事前に構築済み）**のコマンドを実行します。**ソースからビルド**するコマンドは、**リポジトリのルート**から実行します。この章では、モデルアーカイブは必要ありません。

**Python:**
```bash
python3 share/sima-neat/tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_012_diagnose_a_pipeline
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_012_diagnose_a_pipeline
./build/tutorials-standalone/tutorial_012_diagnose_a_pipeline
```

期待される出力（カウンターの値と要約文字列は実行ごとに異なります）。

```text
validate.error_code=
measure.inputs_enqueued=1 outputs_pulled=1
measure.text_size=...
[OK] 012_diagnose_a_pipeline
```

（Pythonビルドでは、`validate_error_code=`、`inputs_enqueued=... outputs_pulled=...`、および`measure_text_size=...`が出力されます。）この章のC++ソースを、カスタムの`CMakeLists.txt`（追加のフォルダーは不要）を使用して、独自のプロジェクトに統合する方法については、ランディングページにある[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## In Practice

構造化された診断、エラー分類、デバッグ用パラメータ、および問題が発生した場合に利用するプラグインの失敗時のワークフロー。`validate()` / `start_measurement()` / `MeasureReport` が問題を示した場合に使用します。

### グラフレポート

`GraphReport` は、構造化された診断情報を記録します。
- パイプライン文字列（再現用）
- 標準化された `error_code`（機械によるトリアージ）
- `repro_note`（人間が理解できる概要とヒント）
- ノードレポートと、そのノードが所有する要素の名前
- バスメッセージとエラーの詳細
- オプションのフロー/タイミングカウンタ

エラーが発生した場合、`NeatError` は、ログに記録またはシリアライズできる `GraphReport` を持ちます。

### エラー分類

フレームワークエラーは、安定したコードファミリーを使用します。

| エラーコード | 意味 | 一般的な修正方法 |
|---|---|---|
| `misconfig.pipeline_shape` | ノードの順序/形状の契約違反 | プッシュパイプラインの場合は最初に `Input()` を、プルパイプラインの場合は最後に `Output()` を配置することを確認します。 |
| `misconfig.caps` | フレームワークの caps オーバーライドまたは隣接するノードの契約の不一致 | `caps_override` と宣言されたノードの契約を一致させます。 |
| `misconfig.media_caps` | ランタイム GStreamer メディアネゴシエーションの不一致 | フォーマット、解像度、フレームレートを一致させるか、変換を挿入します。 |
| `misconfig.input_shape` | 入力テンソル/フレーム/サンプルの形状/レイアウトの不一致 | 幅/高さ/奥行き、レイアウト、dtype、ストレージを検証します。 |
| `build.plugin_missing` | 必要な GStreamer 要素またはコーデックが利用できません | インストール/置き換え、および `gst-inspect-1.0` を使用して検証します。 |
| `build.property_invalid` | GStreamer プロパティの名前または値が無効です | `gst-inspect-1.0 <element>` を使用して確認します。 |
| `build.pipeline_syntax` | カスタム GStreamer フラグメントの構文が無効です | 修正し、`gst-launch-1.0` を使用して検証します。 |
| `runtime.pull` | より具体的な原因がない状態でプルが失敗しました | 添付されたレポートと、最初のアップストリームエラーを確認します。 |
| `io.parse` | 保存されたグラフの JSON の解析/スキーマの失敗 | JSON と必要なノードフィールドを検証します。 |
| `io.open` | グラフの保存/読み込みファイルのオープン/読み取り/書き込みの失敗 | パスの存在、権限、およびストレージの状態を確認します。 |

`PullError.code` は、同じ分類を使用します（例外パスのみではありません）。
これは簡単なトリアージリストです。以前の大まかなランタイムおよびビルドコードからの移行を含む、[完全なエラーコードカタログ](/reference/error-codes)を参照してください。

### プログラムによる処理

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

### デバッグ設定（環境）

重要な環境変数（詳細は[アーキテクチャ](/develop-apps/contribute/architecture)を参照）：
- `SIMA_GST_DOT_DIR`：障害時に DOT グラフを書き出す
- `SIMA_GST_BOUNDARY_PROBES`：境界フローカウンター
- `SIMA_GST_ELEMENT_TIMINGS`：要素ごとのタイミング
- `SIMA_GST_FLOW_DEBUG`：要素ごとのフローカウンター
- `SIMA_GST_ENFORCE_NAMES`：命名規則を強制する

短期間の実行において、編集された生データを使用 GStreamer エラーに追加されるコンテキスト。使用方法：

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` ～ではありません Neat Library 設定。

### デバッグのワークフロー

1. まず `GraphReport.error_code` を記録し、分類に基づいてエラーをグループ化します。
2. 具体的な状況と組み込みのヒントを得るために `GraphReport.repro_note` を記録します。
3. `Graph::describe_backend()` または `last_pipeline()` でパイプラインのテキストを記録します。
4. `MeasureReport::to_text()` または `NeatError::report()` で構造化された診断情報を記録します。
5. `GraphReport.bus` で最初の終端 `ERROR` のソースと詳細を確認します。
6. ランタイムで処理が停止/タイムアウトした場合、フローの停止箇所を特定するために、境界/要素プローブを有効にします。

推奨されるサポートバンドル：
- `error_code`
- `repro_note`
- 完全 `pipeline_string`
- 最初の3～5件の端末バスエラー（`GraphReport.bus`)
- 実行/検証時に使用される環境設定の優先順位

### よくある問題点 → 解決策

| 症状 | 考えられる原因 | 解決策 |
|---|---|---|
| `missing ... plugin` | GStreamer プラグインが見つかりません | 確認 `GST_PLUGIN_PATH`、実行 `gst-inspect-1.0 <plugin>` |
| `appsink 'mysink' not found` | 終端の `Output()` がない | 実行/ビルドパイプラインで `Output` が最後のノードであることを確認する |
| `caps_override is set; renegotiation disabled` | caps が固定されている | `caps_override` を削除するか、入力 caps を固定したままにする |
| `tensor caps change not supported` | 実行時にテンソルの形状/dtype が変化した | テンソルの形状/dtype を安定させる（再ネゴシエーションなし） |

### デバッグプラグインの失敗

プラグインが失敗した場合、NEATはエラーを発生させます。 `NeatError` そのメッセージには、次の内容が含まれています。 GStreamer エラーと構造化されたデバッグ文字列が表示されます。フィールドを使用して、根本原因を迅速に特定してください。

1. **構造化されたフィールドを読み取ります。** 次の項目を探します。 `debug` エラーメッセージ内のキー/値フィールド：
   - `node`：パイプラインでエラーが発生した要素の名前
   - `config_path`：JSON形式の設定ファイル（該当する場合）
   - `model_path`：モデル／パックのパス（該当する場合）
   - `hint`：具体的な修正方法の指針
   - `detail`：不足しているキーやアロケーターの状態など、追加のコンテキスト

   完全なリストについては、[エラー形式リファレンス](/reference/error_format)を参照してください。
2. **パイプラインのコンテキストを確認します。** `Graph::last_pipeline()`またはエラーレポートからパイプライン文字列を使用します。
   - `node`名がパイプラインに含まれていることを確認します。
   - `config_path`が存在し、読み取り可能であることを確認します。
   - capsエラーの場合、エラーが発生したノードに接続されている上流の要素を確認します。
3. **一般的な修正を適用します。**
   - **設定エラー:** JSON構文、必須キー、およびモデルパスを検証します。
   - **capsエラー:** パーサー要素（例：`h264parse`）を追加または修正し、必要なフィールド（例：`parsed=true`、`stream-format=byte-stream`、`alignment=au`）がcapsに含まれていることを確認します。
   - **アロケーターエラー:** 上流の要素が、必要なアロケータータイプ（システム vs. simaai メモリ/セグメント）を使用していることを確認します。
4. **より多くの診断情報を収集します。** 上記のデバッグ設定（`SIMA_GST_DOT_DIR`、`SIMA_GST_FLOW_DEBUG`、`SIMA_GST_ELEMENT_TIMINGS`）を使用します。

## ソースファイル
- C++: `tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.cpp`
- Python: `tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.py`
