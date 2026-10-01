---
title: 周邊目錄常駐程式
---

# 周邊目錄常駐程式

`simaai-peripherals` 為 DevKit 上所有本機用戶端維護單一目前周邊目錄。完整 Core
發行版包含獨立的 `sima-neat-peripherals` 元件套件，並由
`simaai-peripherals.service` 管理。

## 架構

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

常駐程式擁有監控、提供者、重新整理排程、目前快照、掃描與目錄序號，以及可重播事件。
udev 通知只表示需要重新整理；提供者程式碼會在不取得或串流相機的情況下查詢功能並
分類支援狀態。

第一個私有提供者探索 MIPI/libcamera 相機。沒有相機是有效的空白常駐程式目錄。
各提供者會獨立執行。失敗的提供者會加入結構化問題，並只保留自己的最後良好記錄；
健康提供者的目前結果仍會顯示。因此，未來加入麥克風、LiDAR 或其他提供者時，單一
後端失敗不會隱藏不相關的裝置。

## 本機 API

HTTP/JSON 僅透過下列 Unix 通訊端公開，不監聽 TCP：

```text
/run/simaai-peripherals/api.sock
```

systemd 單元會以非特權的 `sima` 使用者執行、為 `sima` 群組建立執行階段目錄，
並將通訊端模式設為 `0660`。

| 方法與路徑 | 行為 |
| --- | --- |
| `GET /v1/health` | 傳回就緒狀態、新鮮度、執行個體 ID、修訂版、序號與最新錯誤。 |
| `GET /v1/catalog` | 傳回內部一致的目前快照。 |
| `GET /v1/events?after_sequence=N&wait_ms=M&instance_id=ID` | 傳回游標後的事件；`wait_ms` 上限為 30000。 |
| `POST /v1/refresh` | 排定由常駐程式擁有的探索重新整理，並傳回含完成權杖 `target_scan_sequence` 的 `202`。 |

要求限制為 8 KiB、回應本文限制為 4 MiB，並行連線限制為 64。過大的目錄會以
`response_too_large` 失敗，而不會無限制傳送。

### V1 回應結構

健康狀態與目錄回應共用下列欄位：

| 欄位 | 意義 |
| --- | --- |
| `schema_version` | 整數目錄結構版本；目前為 `1`。 |
| `instance_id` | 為此常駐程式程序建立的 UUID。 |
| `state` | `starting`、`ready` 或 `degraded`。 |
| `ready` | 是否至少有一個提供者成功完成，包括有效的空白結果。 |
| `stale` | 較新的失敗後，裝置是否來自最後一次成功的掃描。 |
| `revision` | 裝置快照修訂版；僅發生錯誤或復原不會增加修訂版。 |
| `sequence` | 最新可重播事件的序號。 |
| `scan_sequence` | 已完成探索嘗試的次數；成功與失敗都會增加。 |
| `last_success_at` | 最近成功掃描的 UTC 時間戳記，或 `null`。 |
| `last_attempt_at` | 最近嘗試掃描的 UTC 時間戳記，或 `null`。 |
| `error` | `null`，或包含 `code` 與可採取行動之 `reason` 的物件。 |
| `issues` | 各提供者的失敗，以及該提供者是否保留最後良好記錄。 |

`GET /v1/health` 也會傳回 `api_version: "v1"` 與 `device_count`。
`GET /v1/catalog` 則傳回 `devices`，且特定類型的功能會巢狀放在裝置類型之下：

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

若提供者未回報 `model`，該欄位會省略。範圍模式會包含 `size_range`，其中具有
`min_width`、`min_height`、`max_width`、`max_height`、`step_width` 與
`step_height`，而不是離散的 `width` 與 `height`。

事件回應包含 `schema_version`、`instance_id`、`revision`、最新 `sequence`、
`scan_sequence`、`resync_required`、`shutting_down` 與 `events` 陣列。每個事件包含 `sequence`、
`revision` 與 `kind`。裝置事件另有 `device_id`、`device_type` 及 `previous` 和／或
`current`。`error` 事件包含結構化的 `error`；`recovered` 事件沒有裝置，表示成功
掃描已清除 degraded 狀態，即使裝置沒有變更亦然。

```bash
curl --unix-socket /run/simaai-peripherals/api.sock \
  http://localhost/v1/catalog
```

接受明確重新整理時，會傳回如下權杖：

```json
{"accepted":true,"target_scan_sequence":2}
```

輪詢 health 或 catalog，直到 `scan_sequence` 大於或等於該值。在同一個待執行掃描開始
前接受的要求可以共用權杖；探索正在執行時接受的要求則以後續掃描為目標。

### 目錄識別

每次啟動常駐程式都會建立新的 `instance_id`。首次成功掃描會將目錄修訂版設為 `1`，
但不產生合成的 `added` 事件。之後一次掃描若變更多個裝置，修訂版只增加一次；該次
掃描的所有事件共用新修訂版。

每次完成探索嘗試都會增加 `scan_sequence`。每個事件都有單調遞增的 `sequence`。
事件種類為 `added`、`removed`、`changed`、`error` 與 `recovered`。相機識別使用
提供者的確切 libcamera 名稱，而不是不穩定的
`/dev/videoN` 索引。相機詳細資料保留模型、模式或明確大小範圍、幀率、支援分類與
拒絕原因。

### 用戶端重新同步

用戶端應同時保留 `instance_id` 與 `sequence`：

1. 讀取 `/v1/catalog` 並儲存 `instance_id` 與 `sequence`。
2. 使用這些值長輪詢 `/v1/events`。
3. 依序號順序套用傳回的事件。
4. 若 `resync_required` 為 true，捨棄本機狀態並重新讀取 `/v1/catalog`。

常駐程式重新啟動、游標超前或游標落後於有限重播緩衝區時，必須重新同步。緩慢或
中斷連線的用戶端不會阻塞監控或其他用戶端。

## 監控與重新整理

常駐程式會依已登錄提供者宣告的子系統建立 udev 篩選器。目前的相機提供者監聽
`media` 與 `video4linux`。它以 250 毫秒防彈跳合併一連串通知，然後執行每個私有
提供者一次。對不會產生裝置事件的軟體或設定變更，請使用
`POST /v1/refresh` 或 `systemctl reload`。

開發板通電時請勿實際插拔 MIPI 排線相機。新增與移除驗證應使用可安全熱插拔的周邊。

## 操作

```bash
sudo systemctl status simaai-peripherals
sudo systemctl reload simaai-peripherals
sudo systemctl restart simaai-peripherals
journalctl -u simaai-peripherals
```

安裝 `sima-neat-peripherals` 套件時服務會自動啟動，並在意外失敗後重新啟動。正常關閉時，它會先喚醒長輪詢
用戶端並移除通訊端，再等待進行中的探索。用戶端 I/O 具有絕對期限；若外部提供者
未返回，systemd 會強制執行最後的 10 秒停止逾時。

## 疑難排解

| 症狀 | 動作 |
| --- | --- |
| 找不到通訊端 | 檢查 `systemctl status simaai-peripherals` 與服務日誌。 |
| 通訊端存取遭拒 | 確認用戶端屬於 `sima` 群組並重新連線工作階段。 |
| 狀態為 `degraded` | 讀取 `/v1/health` 中的 `error` 與各提供者的 `issues`；保留的記錄會以 `retained_last_good` 標示。 |
| `resync_required` 為 true | 讀取新目錄、取代本機狀態，並從其序號繼續。 |
| 未偵測到設定變更 | 呼叫 `POST /v1/refresh` 或重新載入服務。 |
