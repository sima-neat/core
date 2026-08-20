---
title: "CVU 核心和圖的目錄"
description: "這個框架的 CVU 端核心程式碼的功能是什麼，以及如何從這些核心程式碼中組成預處理/後處理圖。"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/cvu_kernels
---

# CVU 核心和圖的目錄

這個框架提供一組小型 CVU（EV74）核心程式庫，這些核心程式庫會結合起來，形成規劃器為每個模型選擇的預處理和後處理圖。本頁面描述這些核心程式庫以及構成它們的圖族。

## 核心家族

### 預處理核心程式碼

在輸入端和 MLA 之間使用 EV74 進行運算。標準系列：

- **調整大小** — 使用雙線性或最近鄰插值法，並可選擇加入黑邊或居中裁剪。
- **色彩轉換** — RGB ↔ BGR、NV12 → RGB、I420 → RGB、GRAY8 ↔ 封裝。
- **版面設定轉換** — HWC ↔ CHW，軸向排列。
- **正規化** — 針對每個通道計算平均值/標準差（輸入為 FP32，輸出為 FP32）。

### 邊界核函數

跨越 MLA 邊界，支援 FP32 / BF16 / INT8：

- **量化** — 將 FP32 轉換為 INT8，並搭配縮放比例和零點。
- **Dequant** — 將 INT8 轉換為 FP32，並使用縮放比例和零點。
- **轉換資料類型** — FP32 ↔ BF16（不進行縮放／不使用零點）。
- **Tess** / **Detess** — 將版面設定重新排列成 MLA 鑲嵌幾何圖形，或將其從鑲嵌幾何圖形中移除。資料相同，但排列順序不同。

### 融合式核心

當模型合約要求使用邊界核，但又沒有在 MLA 階段納入時，規劃者會選擇的組合方式：

- **QuantTess**——結合「Quant」與「Tess」。
- **DetessDequant** — 將 Detess 與 Dequant 結合。
- **CastTess** / **DetessCast** — 在 BF16 路徑上將 Cast 與 Tess 融合。

### 通用預處理器

當應用程式提供任意使用者定義的轉換時，規劃器會將預處理圖升級為通用變體，將這些轉換合併到單一的 CVU 核心中。在 MLA 邊界處的合約不會改變。

### BoxDecode

一種後處理核心，用於合併用於檢測模型的非最大值抑制 (NMS) / 解碼。它會在輸出樣本中產生 `DetectionMeta`。請參閱 [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h)。

## 圖表是如何構成的

這四個 `PreprocessGraphFamily` 值對應於四個核心鏈：

| 圖表家族 | 連結（輸入 → MLA） |
|--------------|---------------------|
| `Preproc` | 調整大小 → 顏色轉換 → 正規化 → MLA（內部進行鑲嵌處理） |
| `Quant` | 調整大小 → 顏色轉換 → 正規化 → 量化 → MLA（內部進行鑲嵌處理） |
| `Tess` | 調整大小 → 顏色轉換 → 正規化 → 網格化 → MLA |
| `QuantTess` | 調整大小 → 顏色轉換 → 正規化 → 量化 → MLA |

輸出端的雙重處理方式——`Postproc` / `Detess` / `DetessDequant` / 直接傳遞——取決於 MLA 編譯後的輸出核心是否包含 detess/dequant。

請參閱 [dtype 規範](/develop-apps/advanced-concepts/dtype_contract)，以了解這四個類別存在的原因。

## 核心命名慣例

在框架內部，核心元件會透過穩定的字串名稱來進行引用，這些名稱會顯示在「`RoutePlanner`」決策中，以及「`MeasureReport`」外掛程式/核心元件的時間序列資料中：

- `cvu/preproc/<variant>` — 預處理核心程式。
- `cvu/quant/<dtype>` — 量化變體。
- `cvu/tess/<geometry>` — tess 變體。
- `cvu/postproc/box_decode_<type>` — BoxDecode 的變體。

確切的目錄位於 `core/src/pipeline/internal/sima/`（框架的貫穿層）。

## 更多閱讀資料

- 「CVU 核心與圖表目錄」——設計深度探討的第 86 節和第 87 節。
- 「鑲嵌、量化、轉換」——設計深度探討的第 17 節。
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h)——四個角的列舉型別。
- [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h) — 對已解碼的方塊進行後處理。
