---
title: "傳送 JSON 中繼資料"
description: "元資料傳送器 UDP JSON 訊息格式協定"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/metadata_sender
---

# 發送 JSON 中繼資料

當外部檢視器、錄影器或服務透過 UDP 接收 UTF-8 JSON 中繼資料時，請使用 `MetadataSender`。Insight 是一個能夠理解此通訊協定的接收器。

## 通訊協定

- 預設主機：`127.0.0.1`
- 預設中繼資料連接埠基數：`9100`
- 頻道連接埠規則：`metadata_port_base + channel`
- 預設傳送模式：非阻塞模式 (`MSG_DONTWAIT`)
- 負載編碼：UTF-8 JSON 文字
- 必需的最上層欄位：`type`、`data`
- 最大邏輯負載：65,507 位元組

`MetadataSender` 將每個 UDP 負載限制在 1200 位元組或更少。最多 1200 位元組的 JSON 負載將保持為一個未變更的資料包。較大的負載會分割成多個區塊，每個區塊都帶有這個 12 位元組的二進位標頭：

| 位元組 | 大小 | 值 |
|---|---|---|
| 0 | 1 | 魔術位元組 `0x4e` |
| 1 | 1 | 協定版本 `0x01` |
| 2 | 8 | 作為無符號 64 位元大端整數的消息 ID |
| 10 | 1 | 以零為基數的區塊索引 |
| 11 | 1 | 總區塊數 |

每個區塊最多包含 1188 個 JSON 位元組。接收器會重新組裝具有相同發送者位址和消息 ID 的區塊，並按照區塊索引的順序進行解析 JSON。UDP 傳遞仍然是盡力而為：發送者不會重試失敗的區塊，並且 `send_raw_json(...)` 或 `send_metadata(...)` 在第一次本地傳送失敗後會傳回 `false`。

接收器應接受未變更的 JSON 資料包和帶有版本的區塊。在發布此 Neat Library 版本之前或同時，將 Insight 更新到具有區塊重新組裝功能的版本。較舊的 Insight 版本將繼續接收最多 1200 位元組的負載，但無法解碼較大的分塊負載。

對於 Insight，將中繼資料頻道 `N` 與影片 UDP 串流在 `9000 + N` 上的連接。如果 Insight 或其他接收器在容器連接埠重新映射後執行，則從應用程式中明確傳遞映射的主機和連接埠。

追蹤、追蹤目標和其他自訂中繼資料可以作為通用 JSON 傳送。檢視器疊加層支援取決於接收器；Insight 追蹤視覺化效果在 `sima-neat/insight#8` 中單獨追蹤。

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

`send_metadata(...)` 會驗證 `data_json`，並建立這個封包：

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

僅在呼叫者已建立完整的頂層負載時，才使用 `send_raw_json(...)`。

```cpp
sender.send_raw_json(
    R"({"type":"object-detection","data":{"objects":[{"id":"obj_1","label":"car","confidence":0.92,"bbox":[120,80,96,64]}]}})",
    &err);
```

## 預設情況下，即時分派是非阻塞的

`MetadataSender` 預設會將 `MSG_DONTWAIT` 應用於每個資料封包，因此即使本機的傳送緩衝區壅塞，也不會延遲同時分派視訊或推論工作的執行緒。當核心無法立即接受資料封包時，傳送函數會傳回 `false`，而不是等待。將該中繼資料封包視為已丟棄，並繼續進行即時工作；無法保證 UDP 傳遞。

預設建構函式和預設傳送選項是等效的：

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

來電者如果明確表示希望拒絕投遞，則可以選擇啟用此功能：

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
send_opt.nonblocking = false;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

使用 `stats()` 來區分網路壅塞和其他類型的錯誤，並且在明確的封鎖模式下，檢測通話速度慢的情況：

```cpp
const auto stats = sender.stats();
std::cerr << "sent=" << stats.datagrams_sent
          << " would_block=" << stats.would_block
          << " enobufs=" << stats.no_buffer_space
          << " max_send_ns=" << stats.max_send_duration_ns << '\n';
```

在傳送過程中，您可以安全地讀取 `stats()`。將結果視為一個並行診斷快照，而不是單一時間點的交易快照。

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

如同在 C++ 中，明確設定 `send_opt.nonblocking = False`，並且僅在需要同步行為時，將其作為第二個建構函式參數傳遞。
