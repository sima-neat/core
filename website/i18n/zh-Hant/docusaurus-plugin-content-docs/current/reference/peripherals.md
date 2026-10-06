---
title: "周邊裝置目錄"
description: "列出開發板本機目錄服務所回報的 DevKit 周邊裝置"
sidebar_position: 8
---

# 周邊裝置目錄

使用周邊裝置目錄來檢查目前連接到本機
DevKit 的裝置。此 API 可用於 C++ 與 Python，並在兩種語言中傳回相同的具型別
快照。

此目錄屬於 SiMa Sentinel（`simaai-sentinel.service`），也就是
以 `GET /v1/peripherals` 提供目錄的開發板本機常駐程式。Sentinel 只回報
硬體事實。每次 `list()` 呼叫都只會對
Sentinel 執行一次有時限的要求，接著由 Core 判斷 `CameraInput` 支援哪些相機模式。Core
不會掃描硬體、快取第二份目錄，或退回
其他探索路徑。

## 列出周邊裝置

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

傳回的目錄可以逐一走訪。它也包含：

| 欄位 | 意義 |
| --- | --- |
| `revision` | 每當 `devices` 或 `errors` 變更時就會改變。只能用來比較是否相等。 |
| `observed_at` | 此快照所依據的掃描開始的時間；在 Sentinel 完成第一次掃描之前不會設定。在此之前，即使已連接裝置，目錄仍是空的。 |
| `errors` | 在最近一次掃描中失敗的提供者，每一項都帶有 `provider`、`code` 與 `reason`。失敗提供者在其最後一次成功掃描中的裝置會保留在 `devices` 中。 |
| `devices` | 周邊裝置。 |

目錄可以不含任何裝置，這仍是成功的結果。

## 相機詳細資料

當 `peripheral.type == "camera"` 時，`peripheral.camera` 包含：

| 欄位 | 意義 |
| --- | --- |
| `camera_name` | `CameraInputOptions` 所接受的選用精確 libcamera 名稱。目前的輸入 API 無法選取的相機不會有此欄位。 |
| `model` | 提供者有回報時的裝置型號。 |
| `backend` | 探索後端，例如 `mipi` 或 `v4l2`。 |
| `modes` | 離散尺寸或明確的尺寸範圍、影格率、支援旗標，以及拒絕原因。 |

每個模式的 `framerate_num`/`framerate_den` 是 Sentinel 為該模式列出的
影格間隔中最快的速率；若未列出任何間隔，則為 `0/1`。
DevKit 的 ISP 不會為 MIPI 模式列出任何間隔；`CameraInput` 會透過
caps 設定速率。

Core 會略過其 `type` 無法辨識的間隔項目，以及最大值小於最小值的逐步或
連續範圍。被略過的項目
既不會設定速率，也不會涵蓋預設速率，但該模式仍算是
有列出間隔。

Core 會依 `CameraInput` 的預設 libcamera 設定檔
（`profile=Default`，以 `camera_name` 選取相機）分類每個模式。當下列條件全部成立時，
該模式即受支援；條件依此順序檢查，`reason` 會指出
第一個不成立的條件：

1. 相機的 `backend` 為 `mipi`。
2. 格式為 `CameraInputOptions` 的預設格式（`NV12`）。
3. 若該模式列出了影格間隔，其中之一涵蓋預設影格率
   （`30/1`）：可以是 1/30 秒的離散間隔，或是
   包含該速率的逐步或連續範圍。未列出任何間隔的模式不會因速率而遭拒，
   其速率為 `0/1`。列出的間隔全部被略過的模式會因速率而遭拒，
   其速率同樣為 `0/1`。
4. 該模式是 ISP 輸出尺寸（`isp_output` 為 true）。

Core 不會探查、取得、設定相機，也不會從相機串流。因此，目錄中的
支援狀態並不保證稍後的獨占取得一定會
成功。

## 任何周邊裝置類型的詳細資料

每個周邊裝置不論其 `type` 為何，都會在 `details_json` 中以精簡 JSON 形式攜帶
Sentinel 所發布的完整裝置記錄。Python 另外提供
`details`，可將 `details_json` 解碼為新的 `dict`。

`details_json` 是讀取 Core 沒有具型別存取子之周邊裝置類型（例如 `microphone`）的
方式。新的裝置類型只要
Sentinel 一回報即可使用，不需要更新 Core。所有欄位與值都會
保留，包括此 Core 版本不認識的欄位；JSON 會
重新序列化，因此鍵的順序與空白可能與常駐程式的
回應不同。Sentinel 會記載每種類型的欄位。

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

任何 JSON 程式庫都能剖析 `details_json`；此範例使用 nlohmann/json。

Core 只驗證它會讀取的欄位。無效的目錄欄位（`revision`、
`observed_at`、`errors`、`devices`），或缺少有效 `id` 與 `type` 的裝置，
會使 `list()` 因剖析錯誤而失敗。若相機記錄中有 Core
無法讀取的相機欄位（例如來自較新版 Sentinel 的記錄），則不會如此：該裝置仍保留
其 `id`、`type` 與 `details_json`，其 `camera` 維持未設定，而其他
每個裝置都會照常傳回。

## 失敗與範圍

當服務不存在、
版本過舊而無法提供目錄，或周邊裝置探索已停用或停止，
或權限遭拒、要求逾時，或回應
格式錯誤或過大時，`list()` 會擲回帶有穩定錯誤碼的 `NeatError`。訊息會包含下一個操作步驟，
並在 Sentinel 有提供時附上 Sentinel 本身的錯誤文字。請參閱
[錯誤碼目錄](./error-codes.md)。

若未安裝 Sentinel，或其版本過舊而無法提供周邊裝置目錄，
請使用 `sima-cli neat install sentinel` 安裝或更新。若已安裝
但未執行，請啟動 `simaai-sentinel.service`。

此 API 只會連接本機
DevKit 上的 `/run/simaai-sentinel/api.sock`。它不使用 SSH，也不會選取遠端開發板。Insight 與未來的 CLI
客戶端會以同層客戶端的身分直接連接常駐程式，而非透過 Core。

此版本提供單次目錄讀取。重新整理要求與常駐程式
生命週期控制不屬於公開的 Core API。
