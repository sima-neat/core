# 006 在推論之前預處理圖像

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | resnet_50 |
| Labels | preprocessing, normalization, image |

## Concept

設定預處理階段——格式、尺寸和每個通道的正規化——使原始圖像輸入成為與模型訓練時使用的張量完全相同的張量。準確的預處理通常是決定模型是否有效以及是否出現問題的關鍵。

## Walkthrough

編譯後的模型會預期其輸入具有精確的形狀和值範圍：固定的色彩順序、固定的尺寸，以及模型訓練時使用的正規化方法。預處理是將原始解碼後的圖像轉換為精確的張量的階段。如果預處理不正確，模型仍然會執行，但它會產生毫無意義的結果，這就是為什麼在部署的模型「看起來有問題」時，預處理應該是首先驗證的內容。

本章設定了您最常用的預處理控制項：色彩格式、輸入/輸出尺寸、調整大小行為以及每個通道的 `mean`/`stddev` 正規化。然後，它會在執行一個確定性張量通過整個模型之前，檢查模型的預處理圖。到本章結束時，您將會定義一個完整的預處理合約，將其附加到一個模型，並確認已設定的流程存在。

### 設定預處理合約 {#step-configure-preproc}

這些選項定義了預處理階段強制執行的合約。`format`（或 `color_convert.input_format`）會在輸入時固定色彩順序；`input_max_*` 欄位會限制執行階段將接受的動態輸入；調整大小/輸出尺寸會設定用於推論的張量大小；`normalize` 加上每個通道的 `mean`/`stddev` 常數會應用值縮放。正規化常數必須與模型訓練時使用的配方相符——不匹配的統計資料是最常見的低信度輸出原因。

**C++：** 欄位位於 `Model::Options::preprocess` 下——`color_convert.input_format` 採用 `PreprocessColorFormat` 列舉，`normalize.enable` 是一個 `AutoFlag`，而 `normalize.mean` / `normalize.stddev` 是 `std::array<float, 3>`。

**Python：** 欄位位於 `ModelOptions.preprocess` 下——`color_convert.input_format` 採用 `PreprocessColorFormat` 列舉，`normalize.enable` 是一個 `AutoFlag`，常數是分配給 `normalize.mean` / `normalize.stddev` 的列表。

### 建立模型 {#step-load-model}

從檔案路徑加上選項建構 `Model`，將您的預處理合約繫結到已載入的模型。從此時起，模型會攜帶預處理定義，因此從其衍生出的任何階段或執行都會重複使用相同的配方。

### 獨立檢查預處理 {#step-inspect-preproc}

本章在執行整個模型之前檢查預處理片段，因此您可以確認流程存在，然後再對下游的任何內容進行除錯。

**C++：**`stages::Preproc(frames, model)` 會單獨執行預處理步驟，並直接傳回預處理後的 `Tensor` ——我們讀取 `pre.shape.size()`（階數）和 `pre.dtype`，以確認合約已生效。

**Python：**`model.preprocess()` 傳回預處理 `Graph` 片段，因此我們會列印 `describe()` 以檢查已設定的路由；後續的 `model.run([tensor])` 會執行完整的路徑，並回報輸出計數。

## Run

從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **原始碼儲存庫根目錄**執行 **從原始碼建置** 命令。

**Python:**
```bash
python3 share/sima-neat/tutorials/006_preprocess_images/preprocess_images.py \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_006_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_006_preprocess_images
./build/tutorials-standalone/tutorial_006_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

預期的輸出（C++ 建置會列印預處理後的張量階數和 dtype 列舉）：

```text
preproc_rank=3
preproc_dtype=1
[OK] 006_preprocess_images
```

（Python 建置會列印 `preproc_graph=ready`、圖的描述以及 `output_count=...`。）若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始碼檔案
- C++：`tutorials/006_preprocess_images/preprocess_images.cpp`
- Python：`tutorials/006_preprocess_images/preprocess_images.py`
