---
title: 周辺機器カタログデーモン
---

# 周辺機器カタログデーモン

`simaai-peripherals` は、DevKit 上のすべてのローカルクライアント向けに、現在の
周辺機器カタログを 1 つ維持します。Core の完全な配布には独立した
`sima-neat-peripherals` コンポーネントパッケージが含まれ、
`simaai-peripherals.service` により管理されます。

## アーキテクチャ

```text
Linux/udev notification or explicit refresh
                  |
                  v
       simaai-peripherals daemon
       - debounce refresh triggers
                  |
                  v
       private provider registry
       - isolate provider failures
       - retain last-good provider results
       - normalize and classify capabilities
                  |
                  v
       daemon catalog and event replay
                  |
                  v
       Unix socket clients
```

デーモンは監視、プロバイダー、更新スケジュール、現在のスナップショット、スキャンと
カタログのシーケンス、再生可能なイベントを所有します。udev 通知は更新の合図にすぎず、
プロバイダーコードがカメラを取得またはストリーミングせずに機能照会とサポート分類を行います。

非公開プロバイダーは MIPI/libcamera カメラと Linux ALSA キャプチャーデバイスを
検出します。カメラもマイクもない状態は有効な空のカタログです。各プロバイダーは独立して実行されます。障害が発生した
プロバイダーは構造化された問題を追加し、自身の最後の正常なレコードだけを保持します。
正常なプロバイダーの現在の結果は引き続き表示されるため、将来 LiDAR やその他の
プロバイダーを追加しても、1 つのバックエンド障害で無関係なデバイスは消えません。

## ローカル API

HTTP/JSON は次の Unix ソケットだけで公開され、TCP は使用しません。

```text
/run/simaai-peripherals/api.sock
```

systemd ユニットは非特権の `sima` ユーザーとして実行され、`sima` グループ用の
実行時ディレクトリを作成し、ソケットモードを `0660` に設定します。

| メソッドとパス | 動作 |
| --- | --- |
| `GET /v1/health` | 準備状態、鮮度、インスタンス ID、リビジョン、シーケンス、最新エラーを返します。 |
| `GET /v1/catalog` | 内部的に一貫した現在のスナップショットを返します。 |
| `GET /v1/events?after_sequence=N&wait_ms=M&instance_id=ID` | カーソル後のイベントを返します。`wait_ms` の上限は 30000 です。 |
| `POST /v1/refresh` | デーモンによる検出更新を予約し、完了トークン `target_scan_sequence` とともに `202` を返します。 |

要求は 8 KiB、応答本文は 4 MiB、同時接続は 64 に制限されます。大きすぎる
カタログは、無制限に送信される代わりに `response_too_large` で失敗します。

### V1 応答スキーマ

ヘルス応答とカタログ応答は、次のフィールドを共有します。

| フィールド | 意味 |
| --- | --- |
| `schema_version` | 整数のカタログスキーマバージョン。現在は `1`。 |
| `instance_id` | このデーモンプロセス用に作成された UUID。 |
| `state` | `starting`、`ready`、または `degraded`。 |
| `ready` | 有効な空の結果を含め、少なくとも 1 つのプロバイダーが正常に完了したかどうか。 |
| `stale` | 新しい失敗後に、最後に成功したスキャンのデバイスを返しているかどうか。 |
| `revision` | デバイススナップショットのリビジョン。エラーや回復だけでは進みません。 |
| `sequence` | 再生可能な最新イベントのシーケンス。 |
| `scan_sequence` | 完了した検出試行数。成功と失敗の両方で増加します。 |
| `last_success_at` | 最新の成功スキャンの UTC 時刻、または `null`。 |
| `last_attempt_at` | 最新の試行スキャンの UTC 時刻、または `null`。 |
| `error` | `null`、または `code` と対処可能な `reason` を持つオブジェクト。 |
| `issues` | プロバイダーごとの障害と、そのプロバイダーが最後の正常なレコードを保持したかどうか。 |

`GET /v1/health` は `api_version: "v1"` と `device_count` も返します。
`GET /v1/catalog` は代わりに `devices` を返し、種類固有の機能はデバイス種類の
下にネストされます。

```json
{
  "schema_version": 1,
  "instance_id": "7ff4c8a1-...",
  "state": "ready",
  "ready": true,
  "stale": false,
  "revision": 1,
  "sequence": 0,
  "scan_sequence": 1,
  "last_success_at": "2026-10-01T01:23:45.678Z",
  "last_attempt_at": "2026-10-01T01:23:45.678Z",
  "error": null,
  "issues": [],
  "devices": [
    {
      "id": "camera:imx477 5-001a",
      "type": "camera",
      "provider": "daemon.camera.libcamera",
      "camera": {
        "camera_name": "imx477 5-001a",
        "model": "imx477",
        "backend": "libcamera",
        "modes": [
          {
            "format": "NV12",
            "width": 1920,
            "height": 1080,
            "framerate_num": 30,
            "framerate_den": 1,
            "supported": true,
            "reason": ""
          }
        ]
      }
    }
  ]
}
```

プロバイダーが報告しない場合、`model` は省略されます。範囲モードには、離散的な
`width` と `height` の代わりに、`min_width`、`min_height`、`max_width`、
`max_height`、`step_width`、`step_height` を持つ `size_range` が含まれます。

ALSA キャプチャーデバイスは、次の種類固有の形式を使用します。

```json
{
  "id": "microphone:alsa:6f69fe8f648be64e",
  "type": "microphone",
  "provider": "daemon.microphone.alsa",
  "microphone": {
    "name": "Yeti Nano",
    "backend": "alsa",
    "connection": "usb",
    "capture_target": {
      "card_id": "Nano",
      "device": 0,
      "selector": "plughw:CARD=Nano,DEV=0"
    },
    "identity": {
      "stable_key": "usb:1-3.2:1-3.2:1.0:pcm0c",
      "card_index": 2,
      "card_id": "Nano",
      "card_name": "Yeti Nano",
      "card_driver": "USB-Audio",
      "pcm_name": "USB Audio",
      "pcm_node": "/dev/snd/pcmC2D0c",
      "usb": {
        "vendor_id": "b58e",
        "product_id": "0005",
        "bus_path": "1-3.2",
        "interface": "1-3.2:1.0"
      }
    },
    "modes": [
      {
        "interface": 3,
        "altset": 1,
        "format": "S24_3LE",
        "channels": 2,
        "sample_bits": 24,
        "rates_hz": [32000, 44100, 48000],
        "channel_map": ["FL", "FR"]
      }
    ],
    "availability": {
      "state": "available",
      "subdevices": 1,
      "subdevices_available": 1
    }
  }
}
```

`id` は安定した sysfs トポロジーとキャプチャー PCM デバイスから生成され、ALSA
カードインデックスは使用しません。`capture_target` と `identity.card_index` は現在の
スナップショットのルーティングデータです。マイクを開く前に、クライアントは同じ
デーモンの `instance_id`、カタログの `revision`、デバイスの `id` を再取得してから、
返されたセレクターを使用する必要があります。クライアントは ID を解析または生成してはいけません。

プロバイダーは PCM を開いたり外部プローブを実行したりせず、`/proc/asound` と sysfs を
読み取ります。USB ストリームモードでは、形式、チャンネル数、サンプルビット、離散的な
`rates_hz` または連続的な `rate_range_hz`、インターフェース、代替設定、チャンネルマップを
保持します。読み取り専用機能を公開しない非 USB ドライバーも、空の `modes` 配列と
デバイス固有の `issues` を付けて表示します。利用できない任意の識別文字列やリンクは
省略され、再生専用デバイスは表示されません。少なくとも 1 つのキャプチャーサブデバイスが
空いていれば `available`、すべて使用中なら `in_use`、有効な個数を取得できなければ
`unknown` です。

イベント応答には `schema_version`、`instance_id`、`revision`、最新の `sequence`、`scan_sequence`、
`resync_required`、`shutting_down`、`events` 配列が含まれます。各イベントには
`sequence`、`revision`、`kind` が含まれます。デバイスイベントには `device_id`、
`device_type`、`previous` または `current` が追加されます。`error` イベントには
構造化された `error` が含まれます。デバイスが変化しなくても、成功したスキャンで
劣化状態が解消された場合は、デバイスを持たない `recovered` イベントを返します。

```bash
curl --unix-socket /run/simaai-peripherals/api.sock \
  http://localhost/v1/catalog
```

明示的な更新を受け付けると、次のようなトークンを返します。

```json
{"accepted":true,"target_scan_sequence":2}
```

health または catalog の `scan_sequence` がこの値以上になるまで
ポーリングしてください。同じ保留中スキャンの開始前に受け付けた要求はトークンを共有
できます。実行中の検出中に受け付けた要求は次のスキャンを対象にします。

### カタログ ID

デーモン起動ごとに新しい `instance_id` が作成されます。最初の成功した検出は、合成
`added` イベントを生成せず、リビジョンを `1` にします。以後、1 回の検出で複数の
デバイスが変わっても、リビジョンは 1 回だけ進み、同じ検出のイベントは同じ
リビジョンを共有します。

完了した検出試行ごとに `scan_sequence` が増加します。各イベントには単調増加する
`sequence` があり、種類は `added`、`removed`、`changed`、`error`、`recovered` です。
カメラ ID には不安定な `/dev/videoN` ではなく、プロバイダーの正確な libcamera 名を使用します。カメラ詳細はモデル、モードまたはサイズ範囲、
フレームレート、サポート分類、拒否理由を保持します。

マイクの識別には、安定した sysfs トポロジーとキャプチャー PCM デバイスを使用します。
ALSA カードインデックス、`/dev/snd/pcmC*`、生成されたセレクターは再起動や再接続後に
変わる可能性があるため、カタログ ID には使用しません。

### クライアントの再同期

クライアントは `instance_id` と `sequence` の両方を保持します。

1. `/v1/catalog` を読み、`instance_id` と `sequence` を保存します。
2. それらを指定して `/v1/events` をロングポーリングします。
3. 返されたイベントをシーケンス順に適用します。
4. `resync_required` が true の場合、ローカル状態を破棄して `/v1/catalog` を再取得します。

デーモン再起動、将来のカーソル、または再生バッファより古いカーソルでは再同期が
必要です。遅いクライアントや切断されたクライアントは監視や他のクライアントを
ブロックしません。

## 監視と更新

デーモンは登録済みプロバイダーが宣言したサブシステムから udev フィルターを構築します。
カメラプロバイダーは `media` と `video4linux`、ALSA プロバイダーは `sound` を監視します。
250 ms のデバウンス後に各非公開プロバイダーを 1 回実行します。デバイスイベントを生成しない
ソフトウェア、設定、キャプチャー可用性の変更には `POST /v1/refresh` または
`systemctl reload` を使用します。

定期スキャンやアイドル時のヘルスタイマーはありません。起動後、モニター、API リスナー、
プロセス監視は、udev 通知、明示的な更新、クライアント接続、終了シグナル、または
ワーカー障害が届くまで、カーネルのブロッキング待機で休止します。デバウンスタイマーは
実際の udev 通知後のみ、クライアント I/O の期限は接続処理中のみ存在します。

ボードの電源が入っている間は MIPI リボンカメラを物理的に着脱しないでください。
追加・削除テストには安全にホットプラグできる周辺機器を使用してください。

## 運用

```bash
sudo systemctl status simaai-peripherals
sudo systemctl reload simaai-peripherals
sudo systemctl restart simaai-peripherals
journalctl -u simaai-peripherals
```

サービスは `sima-neat-peripherals` パッケージのインストール時に自動起動し、予期しない終了後に再起動します。
正常終了時には進行中の検出を待つ前にロングポーリング中のクライアントを起こし、
ソケットを削除します。クライアント I/O には絶対期限があり、外部プロバイダーが
戻らない場合は systemd が最終的な 10 秒の停止タイムアウトを適用します。

## トラブルシューティング

| 症状 | 対応 |
| --- | --- |
| ソケットがない | `systemctl status simaai-peripherals` とサービスジャーナルを確認します。 |
| ソケットへのアクセスが拒否される | クライアントが `sima` グループに属することを確認し、セッションを再接続します。 |
| 状態が `degraded` | `/v1/health` の `error` とプロバイダー別の `issues` を確認します。保持されたレコードは `retained_last_good` で示されます。 |
| マイクの `modes` が空 | デバイス固有の `issues` を確認します。一部の非 USB ALSA ドライバーは PCM を開かずに機能を公開しません。 |
| `resync_required` が true | 新しいカタログでローカル状態を置き換え、そのシーケンスから再開します。 |
| 設定変更が検出されない | `POST /v1/refresh` を呼ぶか、サービスをリロードします。 |
