---
title: "JSONメタデータを送信します。"
description: "メタデータ送信者のUDP JSON形式の通信プロトコル"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/metadata_sender
---

# JSONメタデータの送信

外部のビューワー、レコーダー、またはサービスがUDP経由でUTF-8 JSONメタデータを受け入れる場合に、`MetadataSender`を使用します。Insightはこのワイヤプロトコルを理解する受信側の1つです。

## ワイヤプロトコル

- デフォルトホスト：`127.0.0.1`
- デフォルトメタデータポートベース：`9100`
- チャネルポートルール：`metadata_port_base + channel`
- デフォルト送信モード：ノンブロッキング（`MSG_DONTWAIT`）
- ペイロードエンコーディング：UTF-8 JSONテキスト
- 必須の最上位フィールド：`type`、`data`
- 最大論理ペイロード：65,507バイト

`MetadataSender`は、各UDPペイロードを1200バイト以下に保ちます。1200バイトまでのJSONペイロードは、変更されていない1つのデータグラムのままです。より大きなペイロードは、この12バイトのバイナリヘッダーを持つチャンクに分割されます。

| バイト | サイズ | 値 |
|---|---|---|
| 0 | 1 | マジックバイト `0x4e` |
| 1 | 1 | プロトコルバージョン `0x01` |
| 2 | 8 | 符号なし64ビットビッグエンディアン整数としてのメッセージID |
| 10 | 1 | ゼロベースのチャンクインデックス |
| 11 | 1 | チャンクの総数 |

各チャンクは、最大1188 JSONバイトを伝送します。受信側は、同じ送信元アドレスとメッセージIDを持つチャンクを、チャンクインデックスの順に再構成してからJSONを解析します。UDP配信はベストエフォートのままです。送信側は、失敗したチャンクを再試行せず、`send_raw_json(...)`または`send_metadata(...)`は、最初のローカル送信の失敗後に`false`を返します。

受信側は、変更されていないJSONデータグラムとバージョン化されたチャンクの両方を受け入れる必要があります。このNeat Libraryバージョンと同時に、またはそれ以前に、チャンクの再構成機能を備えたInsightに更新してください。古いInsightバージョンは、最大1200バイトまでのペイロードを受信し続けますが、より大きなチャンク化されたペイロードをデコードすることはできません。

Insightの場合、メタデータチャネル`N`を、`9000 + N`上のビデオUDPストリームとペアにします。Insightまたは他の受信側がコンテナポートのリマッピングの背後で実行されている場合は、マッピングされたホストとポートをアプリケーションから明示的に渡します。

追跡、トラックレット、およびその他のカスタムメタデータは、汎用JSONとして送信できます。ビューワーオーバーレイのサポートは、受信側に依存します。Insightの追跡視覚化は、`sima-neat/insight#8`で別途追跡されます。

## C++

```cpp
simaai::neat::MetadataSenderOptions opt;
opt.host = "127.0.0.1";
opt.channel = 0;
opt.metadata_port_base = 9100;

std::string err;
simaai::neat::MetadataSender sender(opt, &err);

sender.send_metadata(
    "tracking",
    R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})",
    12345,
    "frame-7",
    &err);
```

`send_metadata(...)` は、`data_json` の検証を行い、以下の形式のメッセージを生成します。

```json
{
  "type": "tracking",
  "timestamp": 12345,
  "frame_id": "frame-7",
  "data": {
    "tracks": [
      {
        "id": "trk-1",
        "bbox": [10, 20, 30, 40]
      }
    ]
  }
}
```

呼び出し元がすでに最上位レベルのペイロード全体を構築している場合にのみ、`send_raw_json(...)` を使用してください。

```cpp
sender.send_raw_json(
    R"({"type":"object-detection","data":{"objects":[{"id":"obj_1","label":"car","confidence":0.92,"bbox":[120,80,96,64]}]}})",
    &err);
```

## リアルタイムディスパッチはデフォルトでノンブロッキング

`MetadataSender` は、デフォルトで各データグラムに `MSG_DONTWAIT` を適用するため、ローカルで輻輳している送信バッファーが、ビデオまたは推論処理もディスパッチするスレッドの処理を遅延させることはありません。カーネルがデータグラムをすぐに受け入れられない場合、送信処理は待機する代わりに `false` を返します。そのメタデータパケットは破棄されたものとして扱い、リアルタイム処理を続行します。UDP による配信は保証されません。

デフォルトのコンストラクターとデフォルトの送信オプションは同等です。

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

配達の試みを明確に拒否したいという希望を持つ発信者は、以下の設定を行うことができます。

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
send_opt.nonblocking = false;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

`stats()` を使用して、輻輳とその他の障害を区別し、明示的なブロックモードでは、応答の遅いコールを検出します。

```cpp
const auto stats = sender.stats();
std::cerr << "sent=" << stats.datagrams_sent
          << " would_block=" << stats.would_block
          << " enobufs=" << stats.no_buffer_space
          << " max_send_ns=" << stats.max_send_duration_ns << '\n';
```

`stats()` は、データの送信処理中に読み込んでも問題ありません。この結果は、特定の時点におけるトランザクションのスナップショットではなく、同時実行型の診断スナップショットとして扱ってください。

## Python

```python
import json
import pyneat

opt = pyneat.MetadataSenderOptions()
opt.host = "127.0.0.1"
opt.channel = 0
opt.metadata_port_base = 9100

sender = pyneat.MetadataSender(opt)

sender.send_metadata(
    "object-detection",
    json.dumps(
        {
            "objects": [
                {
                    "id": "obj_1",
                    "label": "car",
                    "confidence": 0.92,
                    "bbox": [120, 80, 96, 64],
                }
            ]
        }
    ),
    12345,
    "frame-7",
)

stats = sender.stats()
print(stats.datagrams_sent, stats.would_block, stats.max_send_duration_ns)
```

C++と同様に、ブロッキング動作が必要な場合に限り、`send_opt.nonblocking = False` を明示的に設定し、それを2番目のコンストラクタ引数として渡してください。
