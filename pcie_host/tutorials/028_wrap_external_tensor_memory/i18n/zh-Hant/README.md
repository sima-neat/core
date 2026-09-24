# 028 包裝外部張量記憶體

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 分鐘 |
| Model | yolo_v8s |
| Labels | PCIe, C++, tensor, external memory, zero-copy wrapping |

## Concept

`Tensor::from_external()` 讓 C++ 應用程式包裝既有的連續配置，而不必將它複製到新的主機端張量。張量會保留 `std::shared_ptr<void>` 擁有者，因此 PCIe 和 GStreamer 使用期間，背後的配置會維持有效。應用程式也不得在提取對應推論結果之前修改這些位元組。

這可避免連續單一輸入額外產生主機暫存複本。但它不是端對端零複製：PCIe 傳輸仍會將承載資料複製到卡片擁有的記憶體。

## Walkthrough

程式會建立三個可重複使用的輸入槽位，並提交八個合成 FP32 畫面。只有在提取該槽位依序對應的結果後，槽位才會回到可用佇列。

### 檢查模型契約 {#step-inspect-contract}

先建構模型並讀取 `info().inputs`，再配置記憶體。本教學使用單一輸入的 YOLOv8s 封存檔，並確認回報的 dtype 是 FP32，而且其形狀正好占用 `size_bytes` 位元組。

外部檢視必須符合對應 `TensorInfo` 的 dtype、形狀、位元組大小和名稱。請勿從其他模型組建推導這些值。

### 包裝應用程式擁有的記憶體 {#step-wrap-memory}

每個環狀槽位都擁有 `std::shared_ptr<std::vector<float>>`。呼叫 `Tensor::from_external()` 時會傳入基底指標、完整的背後元素數、共享擁有者、模型形狀和路由名稱。因為檢視是連續的，PCIe 主機可直接包裝，而不建立暫存配置。

只保留原始指標並不足夠。傳輸可能在 `push()` 傳回後繼續保留張量，因此共享擁有者是必要的。

### 組建模型 {#step-build-model}

驗證模型契約和環狀配置後再組建。本範例將 `max_inflight` 設為環狀大小，讓應用程式和傳輸具有相同的明確上限。

### 提交並安全地重複使用環狀緩衝區 {#step-submit-ring}

填入可用槽位、呼叫 `push()`，再將該槽位移至處理中佇列。請勿只因 `push()` 已傳回就修改或重複使用其儲存空間。沒有可用槽位時，範例會呼叫 `pull()`，並只在相符的依序結果抵達後，才將最舊槽位送回可用佇列。

每個已接受的推送都對應一次提取，包括最後的清空。若逾時，程式會關閉模型，而不重複使用要求仍可能有效的記憶體。

## Run

請依照[教學設定](/tutorials/before-you-run)的說明安裝 PCIe 主機套件並下載教學套件組。將 YOLOv8s 下載到解壓縮後的 PCIe extras 根目錄：

```bash
sima-cli modelzoo get yolo_v8s
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_028_wrap_external_tensor_memory
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_028_wrap_external_tensor_memory
./build/tutorials-standalone/tutorial_028_wrap_external_tensor_memory
```

預設使用卡片 0 和佇列 0。只有使用其他卡片時才傳入 `--card N`。成功執行會列印：

```text
input=images
ring_slots=3
completed=8
[OK] 028_wrap_external_tensor_memory
```

## In Practice

直接主機包裝路徑需要連續儲存空間。只要描述元有效，具有非連續步幅的張量仍可接受，但主機會將它壓實到暫存配置。多個分別配置的輸入也會封裝到暫存記憶體。

對於多輸入模型，只有當所有張量都是同一個共享封裝配置中的連續檢視時，才能避免這個暫存配置。請依 `info().inputs` 回報的順序提交，並使用每個輸入的名稱和形狀。例如，當兩個輸入都是 FP32 時：

```cpp
const auto& first = info.inputs.at(0);
const auto& second = info.inputs.at(1);
const std::size_t first_count = first.size_bytes / sizeof(float);
const std::size_t second_count = second.size_bytes / sizeof(float);

auto packed =
    std::make_shared<std::vector<float>>(first_count + second_count);
pcie::Tensor input0 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, first.shape, first.name);
pcie::Tensor input1 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, second.shape, second.name,
    static_cast<std::int64_t>(first.size_bytes));

model.push({input0, input1});
```

這項最佳化只會移除主機端封裝複本。PCIe 在推論前仍會將封裝的承載資料複製到卡片擁有的傳輸記憶體。

## 原始檔案

- `run_external_tensor_memory.cpp`
