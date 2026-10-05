# 027 以 INT8 張量僅執行 MLA

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 分鐘 |
| Model | yolo26n-det-int8-b1（Model Zoo，為直接 MLA 輸入輸出而編譯） |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

預設的 PCIe 路徑會將 FP32 張量送到卡上，在卡上將輸入量化，接著執行 MLA，再將輸出反量化並以 FP32 傳回主機。已經持有 INT8 資料、或想自行控制量化步驟的應用程式，可以設定 `ModelOptions.mla_only`。卡上便只執行 MLA：主機提交符合 MLA 進入合約的 INT8 張量，並取得原始的 INT8 輸出頭。`model.info()` 會公開每個張量的量化參數，因此主機可以用同一個公式進行量化與反量化：

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

本教學示範 INT8 的情況。`mla_only` 也支援相容的 BF16 封存檔；BF16 張量沒有量化參數。

## Walkthrough

一支程式在主機上量化影像、執行僅 MLA 的路徑、反量化輸出頭，並在同一個佇列上與預設路徑進行比對。

### 檢視僅 MLA 的合約 {#step-inspect-contract}

在啟用 `mla_only` 的情況下建構 `Model`。`info()` 現在會回報 INT8 的輸入與輸出，每個都帶有 `quant.scale` 與 `quant.zero_point`。參考模型只有一個輸入 `images`，是 INT8 `[640, 640, 3]` HWC 張量。

### 在主機上量化 {#step-quantize-on-host}

參考模型期望一張像素值在 `[0, 1]` 範圍內的 RGB 影像。將影像縮放到 `640x640`，把 BGR 轉成 RGB，除以 255，再以進入端的參數套用量化公式。這個前處理屬於模型，而不是封存檔的合約：其他模型需要自己的處理方式。保留同一組編碼反量化後的值：這正是預設路徑進行同條件比較所需的 FP32 輸入。

### 執行 INT8 路徑 {#step-run-int8}

`build()` 會啟動只包含 MLA 的卡端管線。`run()` 接受 INT8 張量，並依 `info().outputs` 回報的順序與名稱，為每個輸出傳回一個緊密排列的 INT8 張量。此路徑會拒絕其他 dtype；推送 FP32 會失敗，而不會在卡上被量化。

### 反量化並比較 {#step-dequantize-and-compare}

在不啟用 `mla_only` 的情況下建構第二個 `Model`，並將反量化後的 FP32 值送入預設路徑。以每個輸出的參數反量化 INT8 輸出頭，並以該輸出頭的 scale 為單位印出每個輸出頭的最大偏差。兩條路徑以相同的編碼執行相同的 MLA 程式，因此誤差為零。

## Run

依照[教學設定](/tutorials/before-you-run)的說明安裝 PCIe 主機套件並下載教學套件包。

本教學執行 Model Zoo 的 YOLO26n INT8 封存檔，它是為直接 MLA 輸入與輸出而編譯的。將它下載到解壓縮後的 PCIe extras 根目錄：

```bash
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz
```

其他封存檔若以 Model SDK 的 `tessellate_parameters` 設定 `enable_mla=True`、每個輸入使用 `HWC` DRAM 版面配置、每個輸出使用 `HWC16` 編譯，也符合條件。改在 CVU 上進行鑲嵌的封存檔，例如 Model Zoo 的 `yolo_v8s` 建置，在啟用 `mla_only` 時會被拒絕：

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model yolo26n-det-int8-b1.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model yolo26n-det-int8-b1.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model yolo26n-det-int8-b1.tar.gz
```

兩個版本都會印出合約，且每個輸出頭的偏差皆為零：

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00390434 zero_point=-128
  output bbox_0 INT8 [80, 80, 4] scale=0.0302856 zero_point=-117
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 4] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

預設為卡 0 與佇列 0。只有在使用其他卡時才傳入 `--card N`。

## In Practice

對於 INT8 封存檔，當應用程式自行掌握量化時，請啟用 `mla_only`：它已經從感測器或前一個模型產生 INT8、需要原始 INT8 輸出頭進行自己的後處理，或想從卡端延遲中移除量化與反量化階段。每個 scale 與 zero point 都應從 `model.info()` 讀取，切勿從模型的另一個建置複製。

此路徑是全有或全無。每個輸入都必須符合其對應 `info().inputs` 項目的 dtype、形狀與位元組大小，且影像前處理或方框解碼無法與 `mla_only` 併用。當主機持有 FP32 資料且不需要自行進行轉換時，請保留預設路徑。

如需部署診斷，請繼續閱讀 [PCIe 模型工作流程](/develop-apps/development-workflow/pcie-model/)。

## 原始檔案

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`
