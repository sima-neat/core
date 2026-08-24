---
title: "プラグインのエラー形式"
description: "プラグインの失敗から得られる構造化されたエラーフィールド"
sidebar_position: 8
---

# プラグインのエラー形式

プラグインが致命的な状態に遭遇した場合、GStreamer バスに `GST_MESSAGE_ERROR` を送信します。Neat は、このエラーを `NeatError` に昇格させ、分類およびレンダリングのために、サポートされている構造化された詳細情報を保持します。

## エラーのドメインとコード

以下は、プラグイン全体で使用される推奨ドメイン/コードです。

- 設定の解析/検証：`GST_RESOURCE_ERROR_SETTINGS`
- ファイルが見つかりません：`GST_RESOURCE_ERROR_NOT_FOUND`
- ディスパッチャーが利用できません：`GST_RESOURCE_ERROR_BUSY`。代わりに[別の方法]を使用してください。
  `GST_RESOURCE_ERROR_NOT_FOUND` は、ディスパッチャー固有の診断 ID、または構造化されたディスパッチャーフィールドとともにのみ使用できます。
- 割り当てエラー：`GST_RESOURCE_ERROR_NO_SPACE_LEFT`
- キャプション/ネゴシエーションエラー：`GST_STREAM_ERROR_FORMAT`
- ランタイム処理の失敗：`GST_STREAM_ERROR_FAILED`

## バージョン管理された構造化された詳細情報

新しい Neat プラグインのエラーは、`simaai-neat-error` という名前の `GstStructure` を添付します。バージョン 1 には、符号なし整数フィールド `neat-schema-version=1` が含まれます。コアは、バージョン 1 から構造化フィールドを読み取り、不明または欠落しているバージョンに対して、通常の GStreamer ドメイン、コード、メッセージ、およびデバッグ文字列にフォールバックします。これにより、将来のスキーマが古い前提に基づいて解釈されるのを防ぎます。

一般的なフィールド：
- `neat-schema-version`
- `neat-diagnostic-id`
- `neat-reason`
- `plugin`
- `node`
- `stage`
- `graph-id`
- `frame-id`
- `stream-id`
- `input-caps`
- `output-caps`
- `allocator`
- `dispatcher-error`

入力容量エラーは、`actual-width`、`actual-height`、`actual-stride`、
`maximum-width`、`maximum-height`、`maximum-stride`、`resize-width`、`resize-height`、
`required-bytes`、`allocated-bytes`、および `input-format`の情報も提供します。

入力契約エラーは、`input-name`、`segment-name`、`required-bytes`、`actual-bytes`、
`expected-shape`、`expected-layout`、`expected-dtype`、`received-shape`、`received-layout`、および
`received-dtype`の情報も提供します。レイアウトフィールドは、`[3, 224, 224]`（`CHW`）や
`[224, 224, 3]`（`HWC`）などの形状を区別するために使用されます。

古いプラグインでは、デバッグ文字列にスペースで区切られた`key='value'`リストが含まれている場合があります。コアは、互換性のためにこれらのフィールドを引き続き使用します。

## 例

```text
simaai-neat-error, neat-schema-version=(uint)1,
neat-diagnostic-id=(string)neatprocesscvu.input_contract_mismatch,
plugin=(string)neatprocesscvu, node=(string)model_0,
expected-shape=(string)"[3, 224, 224]", expected-layout=(string)CHW,
expected-dtype=(string)Float32, received-shape=(string)"[224, 224, 3]",
received-layout=(string)HWC, received-dtype=(string)UInt8;
```

## 備考

- デフォルトでは、`NeatError::what()` には、正規化されたエラーコードと、ユーザー向けのメッセージが含まれます。
  コンテキスト、是正措置、および診断IDが含まれます。生のGStreamerメッセージとデバッグ文字列は省略されます。
- `SIMA_NEAT_VERBOSE_LEVEL=2`と`SIMA_NEAT_VERBOSE_TOPICS=gstreamer`を設定して、短時間だけ実行します。
  診断実行。これにより、技術的な詳細情報（一部削除）が、`NeatError::what()` および `GraphReport.repro_note` に追加されます。`NeatError::report()` は、診断のための構造化されたインターフェースとして機能し続けます。
- `NEAT_LOG_LEVEL=debug` は、Neat Library の設定項目ではありません。
- URIのユーザー情報と、`auth`、`playback-token`、`hdnts`を含む、認識された認証情報フィールド。
  `stream-key`と`tkn`は、レポートに表示されるパイプライン文字列、再現コマンド、構造化された詳細情報、およびJSONから削除されます。共有する前に、デプロイメント固有のパスとメディアアドレスについて、サポートバンドルを確認してください。
