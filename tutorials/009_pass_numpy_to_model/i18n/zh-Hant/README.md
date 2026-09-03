# 009 將 NumPy 陣列傳遞給模型

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | numpy, pytorch, tensor, io |

## Concept

在 Neat 張量和您已有的結構（NumPy 陣列、PyTorch 張量或 `cv::Mat`）之間移動資料，控制佈局、dtype 和複製語義，以便您可以將 Neat 整合到現有的推論堆疊中。

## Walkthrough

如果您正在將 Neat 整合到現有的推論堆疊中，這就是您需要處理的互通介面：如何將主機資料轉換為 Neat `Tensor`，以及如何將 Neat `Tensor` 再次轉換為主機資料。一開始就正確處理，可以避免常見的整合錯誤——錯誤的佈局、靜默的資料類型轉換、兩個世界之間意外的別名。

這也是兩個語言差異最大的地方。Python 使用者來自 NumPy/PyTorch；C++ 使用者來自 OpenCV。轉換的*概念*是相同的，但 API 名稱和類型不同，因此以下針對不同語言的說明非常重要。到最後，您將會將主機資料轉換為 Neat 張量，檢查其有效載荷而不進行複製，並產生一個受擁有權管理的副本，該副本可以安全地保留，即使原始緩衝區不再存在。

### 將主機資料封裝為張量 {#step-to-tensor}

第一步將您已經擁有的資料轉換為 Neat `Tensor`。您明確標記圖像佈局（`RGB`），以便執行階段正確地解釋位元組，而不是進行推測。`copy=True`（或 C++ 中的 CPU 記憶體選擇）決定張量是否擁有其位元組或引用原始資料——明確的擁有權是安全的預設選項，因為原始緩衝區可能會更改或釋放。

**C++：** `simaai::neat::from_cv_mat(mat, ImageSpec::PixelFormat::RGB, TensorMemory::CPU)` 將 `cv::Mat` 封裝到由 CPU 支援的張量中。

**Python：** `pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)` 將 HWC `uint8` NumPy 陣列封裝起來。

### 檢查有效載荷 {#step-map-and-inspect}

一旦資料成為張量，您就可以讀取它。這是互通介面的單向轉換：在將任何資料傳遞到後續處理之前，確認形狀和位元組是否在轉換過程中保留下來。

**C++：** `tensor.map_read()` 傳回一個 `Mapping`，它公開了一個原始 `data` 指標和 `size_bytes`。這是一個張量儲存空間的*視圖*——沒有複製——這就是為什麼示例可以直接對前幾個位元組進行檢查和驗證。

**Python：** `tensor.to_numpy(copy=True)` 從張量中產生一個 NumPy 陣列；示例列印其 `.shape`，以確認 HWC 佈局在單向轉換中保持完整。

### 擁有副本 {#step-own-a-copy}

最後，產生與原始來源緩衝區完全分離的資料——即使在輸入消失後，也可以安全地保留。這是您傳遞給長期使用的消費者的副本。

**C++：** `tensor.clone()` 複製到新的、由 CPU 擁有的儲存空間中，與其來源的 `cv::Mat` 無關。

**Python：**透過 PyTorch 呈現相同的概念：`pyneat.Tensor.from_torch(t, copy=True, ...)` 和 `tensor.to_torch(copy=True)` 透過一個擁有的 PyTorch 張量進行來回轉換。（如果未安裝 `torch`，則會優雅地跳過。）

## Run

從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預建版本）** 命令；從 **原始碼儲存庫根目錄** 執行 **從原始碼建置** 命令。本章不需要模型封存檔。

**Python:**
```bash
python3 share/sima-neat/tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py \
  --width 128 --height 96
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_009_pass_numpy_to_model
./build/tutorials-standalone/tutorial_009_pass_numpy_to_model \
  --width 128 --height 96
```

預期的輸出（C++）：

```text
tensor_rank=3
tensor_bytes=36864
head_checksum=4342
clone_bytes=36864
[OK] 009_pass_numpy_to_model
```

預期的輸出（Python，已安裝 `torch`）：

```text
numpy_roundtrip_shape=(96, 128, 3)
torch_roundtrip_shape=(96, 128, 3)
```

（如果未安裝 `torch`，Python 建置會輸出 `torch_roundtrip_skipped=True`，而不是 torch 行。）若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（不需要額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

總結用於快速參考的互通介面，一旦您完成來回轉換示範後即可使用。

### 轉換 API

- NumPy：`pyneat.Tensor.from_numpy(array, copy=..., image_format=...)` 輸入；`tensor.to_numpy(copy=...)` 輸出。
- PyTorch：`pyneat.Tensor.from_torch(tensor, copy=..., image_format=...)` 輸入；`tensor.to_torch(copy=...)` 輸出。
- OpenCV（C++）：`simaai::neat::from_cv_mat(mat, pixel_format, memory)` 輸入；`tensor.map_read()` 用於零拷貝檢視；`tensor.clone()` 用於擁有副本。

### 複製與檢視

- `copy=True`（Python）/ `clone()`（C++）會提供與原始資料分離的資料——在原始資料被釋放或變更後，可以安全地保留。
- `copy=False`/ `map_read()` 會提供與原始資料關聯的檢視。成本較低，但僅在原始資料保持存在且未變更時才有效。

### 版面設定和 dtype

- 始終傳遞明確的 `image_format`/ `PixelFormat` 以進行影像資料的處理，以便解釋版面設定，而不是猜測。
- Neat 不會靜默地強制轉換 dtype——在將其饋送給模型之前，請將張量 dtype 與模型的輸入合約進行匹配。

## 原始碼檔案
- C++：`tutorials/009_pass_numpy_to_model/pass_numpy_to_model.cpp`
- Python：`tutorials/009_pass_numpy_to_model/pass_numpy_to_model.py`
