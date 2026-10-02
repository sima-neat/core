---
title: ペリフェラルカタログ
description: ボードローカルのカタログサービスが報告する DevKit のペリフェラルを一覧表示します
sidebar_position: 8
---

# ペリフェラルカタログ

ローカル DevKit に現在接続されているデバイスを確認するには、ペリフェラルカタログを使用します。
この API は C++ と Python で利用でき、どちらの言語でも同じ型付きスナップショットを返します。

信頼できるカタログは、SiMa Sentinel (`simaai-sentinel.service`) が所有します。Sentinel は、
カタログを `GET /v1/peripherals` として提供するボードローカルのデーモンです。各 `list()` 呼び出しは、
Sentinel へ境界時間付きの要求を 1 回実行します。Core はハードウェアをスキャンせず、2 つ目のカタログをキャッシュせず、別の検出経路へフォールバックしません。

## ペリフェラルを一覧表示する

Python:

```python
import pyneat

catalog = pyneat.peripherals.list()
for peripheral in catalog:
    print(peripheral.id, peripheral.type, peripheral.provider)
```

C++:

```cpp
#include <neat.h>

auto catalog = simaai::neat::peripherals::list();
for (const auto& peripheral : catalog) {
  // Use peripheral.id, peripheral.type, and peripheral.provider.
}
```

返されたカタログは反復処理できます。また、デーモンの `instance_id`、カタログの `revision`、
イベントの `sequence`、完了した `scan_sequence`、鮮度の状態とタイムスタンプ、現在の構造化エラー、
プロバイダーの問題、および `devices` コレクションも含みます。

準備完了のカタログにデバイスが 0 台でも、正常な結果です。劣化状態のカタログには、最後に成功した
デバイス一覧が含まれる場合があります。その場合、`stale` は `true` であり、`error` または `issues` が
更新失敗の理由を示します。

## カメラの詳細

`peripheral.type == "camera"` の場合、`peripheral.camera` には次の情報が含まれます。

| フィールド | 意味 |
| --- | --- |
| `camera_name` | `CameraInputOptions` が受け付ける正確な libcamera 名（任意）。現在の入力 API で選択できないカメラでは存在しません。 |
| `model` | プロバイダーが報告した場合のデバイスモデル。 |
| `backend` | `mipi` や `v4l2` などの検出バックエンド。 |
| `modes` | 離散サイズまたは明示的なサイズ範囲、フレームレート、サポートフラグ、および拒否理由。 |

サポートの分類は、この Core パッケージが `/usr/share/simaai-sentinel/support/neat-core.json` にインストールするルールを Sentinel が適用して行うため、結果はインストール済みの `CameraInput` と一致します。クライアントはその結果を保持し、カメラのプローブ、再分類、
取得、設定、ストリーミングを行いません。そのため、カタログ上のサポートは、後で排他的取得が
成功することを保証しません。

未知のペリフェラル型も、共通の `id`、`type`、`provider` とともにカタログに残ります。
プロトコル v1 に追加された任意フィールドは、古いクライアントでは無視されます。
Sentinel が各スナップショットとともに公開するトップレベルの `changes` ログもこれに含まれます。
Core はこのログを受け入れますが、公開しません。

## 障害と適用範囲

サービスが存在しない、権限が拒否された、要求がタイムアウトした、デーモンが未準備である、または
応答が不正、過大、非互換である場合、`list()` は安定したコードを持つ `NeatError` を送出します。
メッセージには次の運用操作が含まれます。[エラーコードカタログ](./error-codes.md)を参照してください。

Sentinel がインストールされていない場合は、`sima-cli neat install sentinel` でインストールしてください。
インストール済みで実行されていない場合は、`simaai-sentinel.service` を起動してください。

この API は、ローカル DevKit 上の `/run/simaai-sentinel/api.sock` のみに接続します。
SSH を使用せず、リモートボードを選択しません。Insight と将来の CLI クライアントは、Core を
経由せず、同等のクライアントとしてデーモンに接続します。

このリリースで提供するのは 1 回限りのカタログ読み取りです。イベント購読、更新要求、および
デーモンのライフサイクル制御は、公開 Core API の一部ではありません。
