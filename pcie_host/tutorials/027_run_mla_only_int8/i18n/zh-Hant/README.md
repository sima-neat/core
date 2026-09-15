# 027 以 INT8 張量僅執行 MLA

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 分鐘 |
| Model | any archive compiled for direct MLA input and output |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

預設的 PCIe 路徑會將 FP32 張量送到卡上，由 EV74 進行量化，接著執行 MLA，再由 EV74 將結果反量化回 FP32。已經持有 INT8 資料、或想自行控制量化步驟的應用程式，可以設定 `ModelOptions.mla_only`。卡上便只執行 MLA：主機提交符合 MLA 進入合約的 INT8 張量，並取得原始的 INT8 輸出頭。`model.info()` 會公開每個張量的量化參數，因此主機可以用同一個公式進行量化與反量化：

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

## Walkthrough

一支程式在主機上量化影像、執行僅 MLA 的路徑、反量化輸出頭，並在同一個佇列上與預設路徑進行比對。

### 檢視僅 MLA 的合約 {#step-inspect-contract}

在啟用 `mla_only` 的情況下建構 `Model`。`info()` 現在會回報 INT8 的輸入與輸出，每個都帶有 `quant.scales[0]` 與 `quant.zero_points[0]`。輸入還帶有 `input_range`，也就是模型校正時使用的浮點數範圍。具有多個輸入的模型會依提交順序為每個輸入列出一個 INT8 張量。

### 在主機上量化 {#step-quantize-on-host}

將影像縮放到進入端的幾何尺寸，把 BGR 轉成 RGB，將像素對應到 `input_range`，再以進入端的參數套用量化公式。保留同一組編碼反量化後的值：這正是預設路徑進行同條件比較所需的 FP32 輸入。

### 執行 INT8 路徑 {#step-run-int8}

`build()` 會啟動只包含 MLA 的卡端管線。`run()` 接受 INT8 張量，並依 `info().outputs` 回報的順序與名稱，為每個輸出傳回一個緊密排列的 INT8 張量。此路徑會拒絕其他 dtype；推送 FP32 會失敗，而不會在卡上被量化。

### 反量化並比較 {#step-dequantize-and-compare}

在不啟用 `mla_only` 的情況下建構第二個 `Model`，並將反量化後的 FP32 值送入預設路徑。以每個輸出的參數反量化 INT8 輸出頭，並以該輸出頭的 scale 為單位印出每個輸出頭的最大偏差。兩條路徑以相同的編碼執行相同的 MLA 程式，因此誤差為零。

## Run

依照[教學設定](/tutorials/before-you-run)的說明安裝 PCIe 主機套件並下載教學套件包。

本教學需要一個為直接 MLA 輸入與輸出而編譯的封存檔：Model SDK 的 `tessellate_parameters` 需設定 `enable_mla=True`，每個輸入使用 `HWC` DRAM 版面配置，每個輸出使用 `HWC16`。Model Zoo 的封存檔改在 EV74 上進行鑲嵌，因此在啟用 `mla_only` 時會被拒絕：

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

將符合條件的封存檔複製到解壓縮後的 PCIe extras 根目錄，例如命名為 `model_mlatess_int8.tar.gz`，並以 `--model` 傳入其路徑。

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model model_mlatess_int8.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

使用為直接 MLA 輸入輸出而編譯的 YOLOv8n 封存檔時，兩個版本都會印出合約，且每個輸出頭的偏差皆為零：

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00391965 zero_point=-128 range=[0, 1]
  output bbox_0 INT8 [80, 80, 64] scale=0.0828159 zero_point=-60
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 64] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

預設為卡 0 與佇列 0。只有在使用其他卡時才傳入 `--card N`。

## In Practice

當應用程式自行掌握量化時，請啟用 `mla_only`：它已經從感測器或前一個模型產生 INT8、需要原始 INT8 輸出頭進行自己的後處理，或想從卡端延遲中移除 EV74 階段。每個 scale、zero point 與輸入範圍都應從 `model.info()` 讀取，切勿從模型的另一個建置複製。

此路徑是全有或全無。多輸入模型的每個輸入都必須以 INT8 送達，且影像前處理或方框解碼無法與 `mla_only` 併用。當主機持有 FP32 資料且不需要控制量化時，請保留預設路徑。

如需部署診斷，請繼續閱讀 [PCIe 模型工作流程](/develop-apps/development-workflow/pcie-model/)。

## Source Files

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`
