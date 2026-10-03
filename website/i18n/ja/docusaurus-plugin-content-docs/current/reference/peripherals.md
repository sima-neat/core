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

サポートの分類は、この Core パッケージが `/usr/share/simaai-sentinel/support/neat-core.json` にインストールするルールを Sentinel が適用して行うため、結果はインストール済みの `CameraInput` と一致します。カタログ上のサポートは、後で排他的取得が成功することを保証しません。

## あらゆるペリフェラル型の詳細

すべてのペリフェラルは、`type` に関係なく、型固有の詳細を `details_json` に保持します。
これは、Sentinel が `type` と同じ名前のレコードキー（例: `camera`、`microphone`、`lidar`）の下に
公開するコンパクトな JSON オブジェクトです。そのキーが存在しないか `null` の場合、
`details_json` は `"{}"` になります。Python では、`details_json` を新しい `dict` に
デコードする `details` も利用できます。

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details.get("channels"))
```

C++ では、任意の JSON ライブラリで `details_json` を解析します。新しいデバイス型は、Core を更新しなくても、
Sentinel が報告した時点ですぐに利用できます。`camera` 以外の型で、詳細の値が JSON オブジェクトでない場合でも
`list()` は失敗しません。そのペリフェラルはカタログに残り、`details_json` は `"{}"` になります。
不正な `camera` の詳細では、`list()` は解析エラーで失敗します。

## 障害

サービスが存在しないかカタログを提供できないほど古い、権限が拒否された、要求がタイムアウトした、
デーモンが未準備である、または
応答が不正、過大、非互換である場合、`list()` は安定したコードを持つ `NeatError` を送出します。
メッセージには次の運用操作が含まれます。[エラーコードカタログ](./error-codes.md)を参照してください。

Sentinel がインストールされていない場合、またはペリフェラルカタログを提供できないほど古い場合は、
`sima-cli neat install sentinel` でインストールまたは更新してください。
インストール済みで実行されていない場合は、`simaai-sentinel.service` を起動してください。
