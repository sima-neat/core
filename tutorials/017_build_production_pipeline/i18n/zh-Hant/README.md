# 017 建立一個可供生產環境使用的管線

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | resnet_50 |
| Labels | production, reliability, deployment |

## Concept

將先前章節中教授的模式組合成一個生產風格的執行迴圈——明確的模型選項、明確的路由選項、明確的執行選項，以及一個非同步推送/拉取迴圈。這不是一個完整的產品框架；而是一個可靠的骨架，您可以根據需要進行調整。

## Walkthrough

這是總結章節。到目前為止，我們每次只處理一個概念；在這裡，它們將結合在一起，形成一個單一的藍圖，您可以將其應用到實際的部署程式碼中。這個範本的整體目的是明確說明預設設定中未明確說明的三個事項：模型的輸入範圍（以便在建置時失敗，而不是在執行過程中失敗）、階段命名（以便在多個模型共享一個程序時，診斷資訊仍然易於閱讀），以及佇列策略（以便在負載下，行為是可以觀察的，而不是神秘的）。

其結構如下：設定執行選項，設定並載入模型，建置一個執行器，然後使用有界非同步迴圈來驅動它。到最後，您將擁有一個 `Runner`，它將執行一個具有生產預設值的非同步管線，以及一個用於計算成功輸出的推送/拉取迴圈——這就是您希望在同一應用程式中的多個模型中標準化的執行階段框架。

### 設定執行選項 {#step-configure-run-options}

這些是生產執行階段的預設值。`queue_depth = 8` 提供一個小的有界緩衝區；`overflow_policy = Block` 使生產者等待，而不是靜默地丟棄幀（當您關心資料遺失時，這是更安全的選擇）；`output_memory = Owned` 確保傳回的張量在拉取後仍然存在。明確設定這些值（而不是依賴預設值）可以使負載下的行為更具可預測性。

### 設定並載入模型 {#step-configure-model}

在這裡，我們在模型上明確說明輸入合約。將 `preprocess.input_max_width/height/depth` 設定為幀的尺寸，意味著不匹配的輸入會在建置時以清晰的合約錯誤的形式失敗，而不是稍後產生一個令人困惑的執行階段錯誤。`name_suffix = "_prod"` 為此模型的階段添加標籤，以便在多模型應用程式中，它們可以在診斷資訊中被識別。然後，我們從檔案路徑和這些選項中建構 `Model`。

**C++：** `Model::Options` 還明確說明了模型預期的預處理——`InputKind::Image`、RGB 顏色轉換以及使用 `has_explicit_stats = true` 的 ImageNet 正規化——因為 C++ 路徑會提前宣告預處理，而不是依賴檔案的預設值。

**Python：** `ModelOptions` 在 `mopt.preprocess.*` 下設定圖像預處理、輸入範圍、ImageNet 正規化和字尾。

### 建置執行器 {#step-build-runner}

`ModelRouteOptions` (C++ `Model::RouteOptions`) 選擇路由包含哪些邊界——這裡 `include_input` 和 `include_output` 都為 true——並使用相同的 `_prod` 後綴，以便路由的元素與模型的命名匹配。然後，我們調用 `model.build(sample, route_options, run_options)`：這是一個單次調用路徑，它將 `Model` 直接連接到一個可運行的 `Runner`，並將路由和運行選項傳遞到底層的管線中。這個示例樣本允許建置鎖定已協商的形狀。

**C++：** 該樣本是一個使用 `Tensor::from_cv_mat(rgb, ..., TensorMemory::EV74)` 建置的 `TensorList`，它將輸入放置在適合設備的記憶體中。

**Python：** 該樣本是一個列表，包含一個來自 `Tensor.from_numpy(...)` 的 `Tensor`。

### 驅動生產迴圈 {#step-run-loop}

這是實際服務運行的迴圈。對於每次迭代，我們 `push(...)` 一個輸入——檢查布林傳回值，以便處理被拒絕的推送（在 `Block` 中，這是一種暫時性情況），而不是錯誤地計數——然後使用有限的超時進行 `pull(...)`，並計數成功的輸出。迴圈結束後，`close()` 會乾淨地關閉運行器。這種推送布林值/帶超時的拉取/明確關閉的模式是可靠的非同步框架；替換成您實際的輸入和輸出處理，結構保持不變。

## Run

本章需要一個模型封存檔（`resnet_50`）。從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）運行 **Python** 和 **C++（預建置）** 命令；從 **儲存庫根目錄** 運行 **從源碼建置** 命令。

**Python:**
```bash
python3 share/sima-neat/tutorials/017_build_production_pipeline/build_production_pipeline.py \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_017_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_017_build_production_pipeline
./build/tutorials-standalone/tutorial_017_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

預期輸出：

```text
outputs=4
[OK] 017_build_production_pipeline
```

（Python 建置會印出 `iters=4 ok=4`。）

要將本章的 C++ 原始程式碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始程式碼檔案
- C++：`tutorials/017_build_production_pipeline/build_production_pipeline.cpp`
- Python：`tutorials/017_build_production_pipeline/build_production_pipeline.py`
