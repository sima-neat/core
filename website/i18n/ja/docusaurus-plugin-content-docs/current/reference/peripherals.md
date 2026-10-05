---
title: "ペリフェラルカタログ"
description: "ボードローカルのカタログサービスが報告する DevKit のペリフェラルを一覧表示します"
sidebar_position: 8
---

# ペリフェラルカタログ

ローカルの DevKit に現在接続されているデバイスを確認するには、ペリフェラルカタログを使用します。
この API は C++ と Python で利用でき、
どちらの言語でも同じ型付きスナップショットを返します。

カタログは SiMa Sentinel (`simaai-sentinel.service`) が所有します。これはカタログを `GET /v1/peripherals` として提供する
ボードローカルのデーモンです。Sentinel はハードウェアの事実のみを報告します。各 `list()` 呼び出しは
Sentinel への上限付きの要求を 1 回実行し、その後 `CameraInput` がどのカメラモードを
サポートするかを Core が判定します。Core は
ハードウェアをスキャンせず、2 つ目のカタログをキャッシュせず、別の
検出経路へのフォールバックも行いません。

## ペリフェラルを一覧表示する

Python:

```python
import pyneat

catalog = pyneat.peripherals.list()
for peripheral in catalog:
    print(peripheral.id, peripheral.type)
```

C++:

```cpp
#include <neat.h>

auto catalog = simaai::neat::peripherals::list();
for (const auto& peripheral : catalog) {
  // Use peripheral.id and peripheral.type.
}
```

返されたカタログは反復処理できます。また、次の情報も含みます。

| フィールド | 意味 |
| --- | --- |
| `revision` | `devices` または `errors` が変化するたびに変わります。等価比較にのみ使用してください。 |
| `observed_at` | このスナップショットの元になったスキャンの開始時刻。Sentinel の最初のスキャンが完了するまでは未設定です。それまでは、デバイスが接続されていてもカタログは空です。 |
| `errors` | 最新のスキャンで失敗したプロバイダー。それぞれ `provider`、`code`、`reason` を持ちます。失敗したプロバイダーのデバイスのうち、最後に成功したスキャンで得られたものは `devices` に残ります。 |
| `devices` | ペリフェラル。 |

カタログに含まれるデバイスが 0 台の場合もあります。これは正常な結果です。

## カメラの詳細

`peripheral.type == "camera"` の場合、`peripheral.camera` には次の情報が含まれます。

| フィールド | 意味 |
| --- | --- |
| `camera_name` | `CameraInputOptions` が受け付ける正確な libcamera 名（任意）。現在の入力 API で選択できないカメラでは存在しません。 |
| `model` | プロバイダーが報告した場合のデバイスモデル。 |
| `backend` | `mipi` や `v4l2` などの検出バックエンド。 |
| `modes` | 離散サイズまたは明示的なサイズ範囲、フレームレート、サポートフラグ、および拒否理由。 |

各モードの `framerate_num`/`framerate_den` は、Sentinel がそのモードについて列挙する
フレーム間隔のうち最も速いレートです。何も列挙されていない場合は `0/1` になります。
DevKit の ISP は MIPI モードについて何も列挙しません。`CameraInput` はレートを
caps で設定します。

Core は、`type` が不明な間隔エントリと、最大値が最小値より短いステップ状
または連続的な範囲をスキップします。スキップされたエントリは
レートを設定せず、デフォルトレートもカバーしませんが、そのモードは引き続き
間隔を列挙しているものとして扱われます。

Core は、`CameraInput` のデフォルト libcamera プロファイル
（`camera_name` でカメラを選択する `profile=Default`）について各モードを分類します。モードは、
次の条件をすべて満たす場合にサポートされます。条件はこの順にチェックされ、`reason` は
最初に満たされなかった条件を示します:

1. カメラの `backend` が `mipi` であること。
2. 形式が `CameraInputOptions` のデフォルト形式（`NV12`）であること。
3. モードがフレーム間隔を列挙している場合、そのいずれかがデフォルトのフレームレート
   （`30/1`）をカバーしていること。つまり、1/30 秒の離散間隔、またはそれを含むステップ状
   もしくは連続的な範囲であること。間隔を列挙しないモードは、レートを理由に拒否されず、
   レートは `0/1` になります。列挙された間隔がすべてスキップされたモードは、レートを理由に拒否され、
   レートはやはり `0/1` になります。
4. モードが ISP 出力サイズであること（`isp_output` が true）。

Core はカメラのプローブ、取得、設定、ストリーミングを行いません。そのため、
カタログ上のサポートは、後で排他的な取得が成功することを保証する
ものではありません。

これらのルールは Metoak SIMOR の raw V4L2 プロファイル
（`CameraProfile::MetoakSimor`、RAW8 1920×360、`profile` と
任意の `device` で選択）を分類しません。ルールにはセンサーごとの条件がないため、SIMOR センサー
（名前が `simor_metoak` で始まるもの）を Sentinel が `mipi` カメラとして列挙した場合、
他の MIPI センサーと同じ ISP モード分類になります。このカメラでの
`supported: true` は、モードがデフォルトプロファイルの
バックエンド、形式、フレームレート、ISP 出力サイズに一致することだけを意味し、
libcamera がそのセンサーに対応済みであることは意味しません。SIMOR カメラは、
カタログの `camera_name` ではなく、
[`CameraInput`](/reference/nodes/camera-input) で説明している
`profile=MetoakSimor` で選択してください。

## あらゆるペリフェラル型の詳細

すべてのペリフェラルは、`type` に関係なく、Sentinel が公開したデバイスレコード全体を
コンパクトな JSON として `details_json` に保持します。Python では、
`details_json` を新しい `dict` にデコードする `details` も利用できます。

`details_json` は、Core に型付きアクセサーがないペリフェラル型（`microphone` など）を
読み取るための方法です。新しいデバイス型は、Core を更新しなくても、Sentinel が報告した
時点ですぐに利用できます。この Core リリースが認識しないフィールドも含め、すべてのフィールドと値が
保持されます。JSON は再シリアル化されるため、キーの順序と空白はデーモンの
応答と異なる場合があります。
各型のフィールドは Sentinel のドキュメントに記載されています。

Python:

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details["capture_target"])
```

C++:

```cpp
#include <nlohmann/json.hpp>

for (const auto& peripheral : simaai::neat::peripherals::list()) {
  if (peripheral.type == "microphone") {
    const auto details = nlohmann::json::parse(peripheral.details_json);
    // Read details["capture_target"] and other fields.
  }
}
```

`details_json` は任意の JSON ライブラリで解析できます。この例では nlohmann/json を使用しています。

Core が検証するのは、自身が読み取るフィールドのみです。カタログのフィールド（`revision`、
`observed_at`、`errors`、`devices`）が不正な場合や、有効な `id` と `type` を持たないデバイスがある場合は、
`list()` は解析エラーで失敗します。一方、Core がカメラフィールドを読み取れないカメラレコード
（たとえば、より新しい Sentinel からのもの）では失敗しません。そのデバイスは
`id`、`type`、`details_json` を保持し、`camera` は未設定のままとなり、
他のすべてのデバイスは通常どおり返されます。

## 障害と適用範囲

サービスが存在しない場合、カタログを提供できないほど古い場合、
ペリフェラル検出が無効または停止している場合、権限が拒否された場合、
要求がタイムアウトした場合、または応答が不正もしくは過大な場合、
`list()` は安定したコードを持つ `NeatError` を送出します。メッセージには次に取るべき運用上の対処が含まれ、
Sentinel がエラーテキストを返した場合は Sentinel 自身のエラーテキストも含まれます。
[エラーコードカタログ](./error-codes.md) を参照してください。

Sentinel がインストールされていない場合、またはペリフェラルカタログを提供できないほど古い場合は、
`sima-cli neat install sentinel` でインストールまたは更新してください。インストール済みで
実行されていない場合は、`simaai-sentinel.service` を起動してください。

この API は、ローカルの DevKit 上の `/run/simaai-sentinel/api.sock` にのみ
接続します。SSH は使用せず、リモートボードも選択しません。Insight と将来の CLI
クライアントは、Core を経由せず、同等のクライアントとしてデーモンに接続します。

このリリースで提供するのは 1 回限りのカタログ読み取りです。更新要求とデーモンの
ライフサイクル制御は、公開 Core API には含まれません。
