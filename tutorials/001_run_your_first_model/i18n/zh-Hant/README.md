# 001 執行您的第一個模型

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Model | resnet_50 |
| Labels | model, inference, foundations |

## Concept

載入已編譯的 ResNet-50 模型封存檔，輸入一張圖片，並讀取前五名預測結果——從「我有一個模型封存檔」到「我得到了一個預測結果」，這就是最快的路徑。

## Walkthrough

這是入門章節。目標是實現盡可能簡潔的端到端推論：載入已編譯的模型，輸入一張圖片，然後輸出預測的類別索引。無需圖、無需執行緒、無需串流——僅僅是每個 Neat 程式所依賴的三個呼叫。

*已編譯的模型* 是一個可部署的 `.tar.gz` 檔案，其中包含一個 MPK 推論合約：模型成品以及 Neat 在目標裝置上執行它所需的執行階段中繼資料。您無需自行解壓縮或連接各個階段——只需將 Neat 指向該檔案，提供輸入，然後讀取輸出。最後，您將在三行程式碼中執行推論，並輸出一個 `top1=` 類別索引。

### 載入模型 {#step-load-model}

第一行將磁碟上的路徑轉換為一個可運行的 `Model`：建構函式載入檔案並為執行做好準備。

**C++：** 您將 `build_options(size)` 作為第二個參數傳遞，以宣告此模型所期望的輸入合約——RGB 顏色、`224×224`，以及 ImageNet 正規化，這是 ResNet-50 訓練時使用的。在這裡宣告它，可以告訴執行階段如何將原始圖片轉換為模型所需的張量。

**Python：** 您在建構 `pyneat.Model` 時，透過 `build_options(size)` 傳遞相同的合約。

### 準備輸入 {#step-prepare-input}

接下來，我們產生一張用於分類的圖片。如果您傳遞 `--image`，則會讀取該圖片，並將其調整大小為 `224×224`，然後轉換為 RGB，以符合輸入合約；否則，我們會合成一個純灰色框架，以便完整的載入 → 執行 → 讀取路徑仍然可以端到端地執行，而無需準備現成的素材。

**C++：** 框架是一個 `cv::Mat`，由 `load_rgb(...)` 產生，或作為一個灰色佔位符。

**Python：** 框架是一個由 `load_image(...)` 建構的 NumPy 陣列，並封裝為帶有 RGB 圖片中繼資料的 `Tensor`。

### 執行推論並讀取結果 {#step-run-inference}

第三行執行實際操作：`run()` 接收輸入和一個 `timeout_ms`，同步執行模型，並傳回輸出。`timeout_ms` 是最大等待時間——這裡的 `2000` 毫秒表示「如果裝置在兩秒內沒有產生輸出，則會立即失敗」，而不是無限期地等待。（傳遞 `-1` 會無限期地阻塞；在實際程式碼中，最好使用有限值。）然後，我們使用 `argmax` 將輸出縮減為單個類別索引，並輸出 `top1=`。

**C++：** `run()` 傳回一個 `TensorList`；透過 `map_read()` 讀取第一個張量的位元組。

**Python：** 使用張量/圖像輸入執行 `run()` 會傳回一個 `TensorList`；`outputs[0].to_numpy()` 會提供一個 NumPy 陣列，以便您對其執行 `argmax`。

這就是全部。後續章節中的所有內容——非同步、管線、自訂圖——都是基於這三個相同的步驟：建構、輸入、讀取。

## Run

執行它，您應該會看到預測的類別索引列印到標準輸出。從 Neat 安裝根目錄（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **原始碼建置** 命令的 **儲存庫根目錄** 執行。

**Python:**
```bash
python3 share/sima-neat/tutorials/001_run_your_first_model/run_your_first_model.py \
  --model /tmp/resnet_50.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_001_run_your_first_model
./build/tutorials-standalone/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

預期的輸出（確切的索引取決於圖像）：

```text
top1=285
[OK] 001_run_your_first_model
```

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

對於吞吐量、批次處理或即時串流，請繼續閱讀第 002 章。參考：[模型](/develop-apps/development-workflow/model)。

## In Practice

教學和測試會尋找模型封存檔（`.tar.gz`）和範例資源的位置，以及如何本機提供它們。這是所有基於模型的教學的先決條件。

### 確保 `sima-cli` 位於 PATH 環境變數中

某些測試會從非互動式 shell 呼叫 `sima-cli`。在安裝 `sima-cli` 後，只需執行一次：

```bash
SIMA_CLI_BIN_DIR="<path-to-sima-cli-bin>"
grep -Fqx "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" ~/.bashrc || echo "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

然後驗證：

```bash
/bin/sh -c 'command -v sima-cli'
```

### 模型封存檔位置和環境變數

提取/執行階段放置控制項：
- `SIMA_MPK_EXTRACT_ROOT=<dir>` 設定基本提取目錄。
- `SIMA_MPK_CLEANUP_EXTRACTED=0` 在程序結束後保留提取的 `proc_*` 模型資料。
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0` 在啟動時停用已停止的 `proc_*` 清理。

#### ResNet50

搜尋順序：
1. `SIMA_RESNET50_TAR`（每個模型的覆寫）
2. `SIMA_MODEL_TAR`（模型封存檔測試/範例的共享後備）
3. `tmp/resnet_50.tar.gz`
4. 如果找到，則將本機檔案移動到 `tmp/`：`resnet_50.tar.gz`、`resnet-50.tar.gz`

下載（如果 `sima-cli` 可用）：
```bash
sima-cli modelzoo get resnet_50
```

### 範例圖像

教學/測試中使用的預設圖像候選者：
- `tmp/coco_sample.jpg`（如果遺失則下載）
- `test.jpg`
- `tests/assets/preproc_dynamic/ilena_488.jpg`

您可以使用以下命令覆寫測試中使用的 COCO 圖像 URL：
```bash
SIMA_COCO_URL=<custom_url>
```

### 測試檔案下載的位置

測試和範例通常會將下載的資源放置在程式碼庫根目錄下的 `tmp/` 資料夾中。如果缺少所需的資源，教學課程會優雅地跳過。

### 疑難排解資源

- 如果教學課程顯示 `SKIP: missing ...`，請提供資源或傳遞一個旗標（例如，`--model <path>`、`--image <path>`）。
- 如果 `sima-cli` 無法使用，請設定環境變數以指向本機的模型封存檔。

## 原始檔案
- C++：`tutorials/001_run_your_first_model/run_your_first_model.cpp`
- Python：`tutorials/001_run_your_first_model/run_your_first_model.py`
