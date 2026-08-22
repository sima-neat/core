# 010 在單個樣本中傳送多個輸入

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | None |
| Labels | multi-input, samples, sync |

## Concept

將多個具名稱的張量捆綁成一個 `Sample`，並將其作為單一的推論事件傳送——這是用於處理接受多個輸入的模型（例如，立體圖像、圖像 + 中繼資料或感測器融合）的模式。

## Walkthrough

許多實際應用程式在每次推論事件中會處理多個輸入。Neat 將其表示為一個「組合樣本」（bundle sample）：一個單一的 `Sample`，其 `fields` 列表包含多個具名稱的張量有效載體，每個有效載體都可以透過一個 `port_name` 來存取。執行階段將具名稱的欄位組合在一起，作為一個邏輯事件，因此 `left` 和 `right`（或影像和中繼資料）在整個管線中保持對齊。

本章將建立一個張量輸入/張量輸出的圖，將兩個具名稱的浮點張量組合在一起，將組合推送到管線中，然後讀取具名稱的欄位。在本章結束時，您將建立一個多欄位樣本，並確認兩個欄位都已完整地通過了整個流程，且其連接埠名稱保持不變。

### 設定張量輸入 {#step-configure-tensor-input}

此圖處理原始張量，而不是已解碼的影像，因此輸入合約宣告為張量有效載體（`FP32`，具有 `width`/`height`/`depth`），而不是像素格式。這會告訴輸入節點直接接受張量緩衝區。

**C++：** 設定 `in.payload_type = PayloadType::Tensor`。

**Python：** 設定 `inp.payload_type = pyneat.PayloadType.Tensor` 和 `inp.format = pyneat.Format.FP32`。

### 建立圖和種子執行 {#step-build-seed-run}

我們使用來自第 004 章的相同最小 `Input -> Output` 拓撲結構，並將其 `build()` 到一個 `Run` 中。`build()` 需要一個具有代表性的樣本來鎖定已協商的形狀，因此我們傳遞一個單一的種子張量（所有值都為零），其形狀與實際欄位將使用的形狀相同。種子僅用於形狀協商——實際資料稍後會傳遞。

### 建立組合 {#step-make-bundle}

現在，將多輸入事件組合在一起。每個輸入都透過 `make_tensor_sample(port_name, tensor)` 獲得一個名稱，並且模型會透過連接埠來存取這些具名稱的欄位。在這裡，`left` 填充了 `1.0`，而 `right` 填充了 `2.0`，以便您可以區分它們在輸出時的內容。

**C++：** `make_bundle_sample({...})` 將具名稱的欄位包裝到一個 `Sample` 中，其 `kind` 為 `Bundle`。

**Python：** 具名稱的樣本列表會直接傳遞到 `push(...)`；pyneat 會為您建立組合封套。

### 推送組合並讀取它 {#step-push-and-read}

最後，將組合推送到管線中並檢查結果。輸出本身也是一個組合 `Sample`，因此我們讀取 `out.fields`，而不是將其視為單一的張量——`out.fields.size()` 應該是 `2`，並且每個欄位都包含 `port_name` 和一個張量有效載體。

**C++：**`run.run(Sample{bundle}, timeout_ms)` 會傳回一個 `Sample`。由於邏輯結果具有多個欄位，因此傳回的 `Sample` 本身就是一個 `Bundle`，因此我們檢查 `out.kind == SampleKind::Bundle` 並迭代 `out.fields`，而不是 `front()`（這會表示「bundle 內的第一個欄位」）。

**Python：**`run.push(fields)`，然後 `run.pull(timeout_ms=...)` 會傳回輸出樣本；迭代 `out.fields` 並讀取每個 `field.port_name` 和 `field.tensor`。

## Run

從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **原始碼儲存庫根目錄** 執行 **從原始碼建置** 命令。本章不需要模型封存檔。

**Python:**
```bash
python3 share/sima-neat/tutorials/010_feed_multi_input_model/feed_multi_input_model.py \
  --width 64 --height 48
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_010_feed_multi_input_model
./build/tutorials-standalone/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

預期的輸出（C++）：

```text
bundle_fields=2
  field=left has_tensor=yes
  field=right has_tensor=yes
[OK] 010_feed_multi_input_model
```

（Python 建置會以 `port=left has_tensor=True` 行的形式輸出相同的欄位計數。）若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（不需要額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

如何將 bundle 模式應用於此兩個欄位的示範之外。

### 命名和路由

- `port_name` 是接線合約：這是多輸入模型用來尋址每個欄位的方式。將名稱與模型宣告的輸入連接埠相符。
- 輸出 bundle 會保留欄位結構，因此您可以根據名稱而不是位置將結果與輸入進行比對。

### 檢查輸出 bundle

- 始終首先根據 `kind` 進行分支：多欄位結果是 `SampleKind.Bundle`，將其作為單一張量讀取將無法運作。
- 在觸及有效載荷之前，請檢查每個欄位中是否存在張量（`field.tensor is not None` / `field.tensor.has_value()`）——欄位可能攜帶中繼資料，而不是張量。

## 原始程式碼檔案
- C++：`tutorials/010_feed_multi_input_model/feed_multi_input_model.cpp`
- Python：`tutorials/010_feed_multi_input_model/feed_multi_input_model.py`
