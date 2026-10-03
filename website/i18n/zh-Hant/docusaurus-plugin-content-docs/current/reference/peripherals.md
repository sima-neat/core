---
title: 周邊裝置目錄
description: 列出開發板本機目錄服務所回報的 DevKit 周邊裝置
sidebar_position: 8
---

# 周邊裝置目錄

使用周邊裝置目錄來檢查目前連接到本機 DevKit 的裝置。此 API 可用於 C++ 和 Python，
並在兩種語言中傳回相同的具型別快照。

權威目錄由 SiMa Sentinel（`simaai-sentinel.service`）擁有，這是以 `GET /v1/peripherals`
提供目錄的開發板本機常駐程式。每次 `list()` 呼叫只會對 Sentinel 執行一次有時限的要求。Core 不會掃描硬體、快取第二份目錄，或退回其他探索路徑。

## 列出周邊裝置

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

傳回的目錄可以反覆運算。它也包含常駐程式 `instance_id`、目錄 `revision`、事件 `sequence`、
已完成的 `scan_sequence`、新鮮度狀態與時間戳記、目前的結構化錯誤、提供者問題，以及
`devices` 集合。

就緒的目錄可以不含任何裝置，這仍是成功結果。降級的目錄可能包含最後一次成功的裝置清單；
在此情況下，`stale` 為 `true`，而 `error` 和/或 `issues` 會說明重新整理失敗的原因。

## 相機詳細資料

當 `peripheral.type == "camera"` 時，`peripheral.camera` 包含：

| 欄位 | 意義 |
| --- | --- |
| `camera_name` | `CameraInputOptions` 接受的選用精確 libcamera 名稱。目前輸入 API 無法選取的相機不會有此欄位。 |
| `model` | 提供者有回報時的裝置型號。 |
| `backend` | 探索後端，例如 `mipi` 或 `v4l2`。 |
| `modes` | 離散尺寸或明確尺寸範圍、畫面更新率、支援旗標，以及拒絕原因。 |

支援狀態由 Sentinel 套用此 Core 套件安裝於 `/usr/share/simaai-sentinel/support/neat-core.json` 的規則進行分類，因此結果會與已安裝的 `CameraInput` 一致。用戶端會保留該結果，不會探查、重新分類、取得、設定或串流相機。
因此，目錄中的支援狀態不保證稍後的獨佔取得一定成功。

## 任何周邊裝置類型的詳細資料

每個周邊裝置不論其 `type` 為何，都會在 `details_json` 中帶有該類型專屬的詳細資料：這是 Sentinel
在名稱與 `type` 相同的記錄鍵（例如 `camera`、`microphone` 或 `lidar`）下發布的精簡 JSON 物件。
若該鍵不存在或為 `null`，`details_json` 即為 `"{}"`。Python 另外提供 `details`，可將
`details_json` 解碼為新的 `dict`。

`details_json` 是讀取 Core 沒有型別化存取子之周邊裝置類型的權威方式。新的裝置類型只要 Sentinel
回報即可立即使用，不需要更新 Core。所有欄位與值都會保留，包括此 Core 版本不認識的欄位；JSON
會重新序列化，因此鍵的順序與空白可能與常駐程式的回應不同。相機同時具有 `details_json` 與型別化的
`camera` 欄位。

Python:

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details.get("channels"))
```

C++:

```cpp
#include <nlohmann/json.hpp>

for (const auto& peripheral : simaai::neat::peripherals::list()) {
  if (peripheral.type == "microphone") {
    const auto details = nlohmann::json::parse(peripheral.details_json);
    // Read details.value("channels", 0) and other provider fields.
  }
}
```

任何 JSON 程式庫都能剖析 `details_json`；此範例使用 nlohmann/json。

詳細資料值若存在，必須是 JSON 物件；任何其他值都會使 `list()` 因剖析錯誤而失敗。對於沒有型別化
存取子的類型，Core 不會再做其他驗證。`camera` 等型別化欄位會忽略其不認識的通訊協定 v1 選用欄位；
這些欄位仍可在 `details_json` 中取得。Core 也會接受 Sentinel 隨每份快照發布的頂層 `changes` 記錄，
但不會公開它。

## 失敗與範圍

服務遺失、權限遭拒、要求逾時、常駐程式尚未就緒，或回應格式錯誤、過大或不相容時，
`list()` 會擲回具有穩定代碼的 `NeatError`。訊息會包含下一個操作步驟。請參閱
[錯誤代碼目錄](./error-codes.md)。

若尚未安裝 Sentinel，請使用 `sima-cli neat install sentinel` 安裝。若已安裝但未執行，
請啟動 `simaai-sentinel.service`。

此 API 只會連接本機 DevKit 上的 `/run/simaai-sentinel/api.sock`。它不使用 SSH，也不會選取
遠端開發板。Insight 和未來的 CLI 用戶端會以同層用戶端身分直接連接常駐程式，而非透過 Core。

此版本提供單次目錄讀取。事件訂閱、重新整理要求和常駐程式生命週期控制不屬於公開 Core API。
