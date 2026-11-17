# 調整大小、色彩轉換與正規化

當您需要單一前處理操作，而不是模型融合的 `Preproc` 時，請使用具型別的影像前端。
這些前端在 EV74 上執行，不會加入 CPU 備援、重新封裝或負載複製。

```cpp
#include <neat.h>

simaai::neat::ResizeOptions size;
size.width = 640;
size.height = 480;
size.max_input_width = 1920;
size.max_input_height = 1080;
auto resize = simaai::neat::nodes::Resize(size);

simaai::neat::ColorConvertOptions color;
color.output_format = simaai::neat::ImageFormat::NV12;
color.surface = simaai::neat::ImageSurfaceLayout::EncoderNv12;
auto convert = simaai::neat::nodes::ColorConvert(color);
```

Python 提供相同的選項型別，以及 `pyneat.nodes.resize`、
`pyneat.nodes.color_convert` 和 `pyneat.nodes.normalization` 工廠函式。

## 輸入與輸出契約

- **Resize** 會變更幾何尺寸並保留原始影像格式。輸出尺寸固定。
  輸入尺寸可在已準備的容量內變動；省略最大值時會使用初始輸入尺寸。
  使用 `ResizeFit::Contain` 進行 letterbox 填補，或使用 `Stretch` 分別縮放各軸。
  輸入格式保持固定。
- **ColorConvert** 會保留尺寸並產生 RGB、BGR、GRAY8、NV12、I420 或 NV16。
  節點首次具體化時即固定輸入幾何尺寸與格式。若要變更，請建立新的節點與圖；
  重新建置同一個圖不會重設節點。
- **Normalization** 接受 UInt8 RGB、BGR 或 GRAY8，並產生密集 BF16 張量，
  而非原始視訊。它會保留尺寸和通道順序。如同 ColorConvert，若要變更已具體化的
  輸入幾何尺寸或格式，請使用新的節點與圖。係數採用語意上的 RGB 順序，
  即使儲存格式為 BGR 亦然。運算為
  `(x - float(mean * 255)) * float(1 / (stddev * 255))`，接著執行現有的
  Preproc BF16 截斷。標準差必須為有限正值。

目前 GRAY8 輸出需要 GRAY8 輸入；不提供 RGB/YUV 轉亮度的轉換。
請先將 YUV 轉成 RGB 或 BGR，再進行正規化。NV12 和 I420 的寬度與高度必須為偶數；
NV16 的寬度必須為偶數，色度則為完整高度。YUV letterbox 的放置位置必須符合相同的
色度網格。明確指定的來源色度學與色度取樣位置必須與設定的原則一致。
不接受交錯掃描或 DV／交替行色度。這些前端每次呼叫處理一個影格，而非 ROI 清單
或影格批次。具有完整仿射來源資訊的上游純量 ROI 結果不會被視為新的裁切要求。
缺少這些來源資訊的純量 ROI 要求會被拒絕。串接操作會合成輸出到原始影像的座標轉換。

EV74 將來源與輸出的每個維度限制為 4096 像素。`BoxAverage` 允許每軸最多觸及
八個分量單元，包含色度相位和取樣位置；因此，未對齊的八比一縮小可能會被拒絕。
不支援的幾何尺寸會使準備階段失敗，而不是變更插值方式或改用 A65。

## 影格率

這些操作不會改變影格率。已準備的原始影像輸出 caps 會保留已知的上游影格率，
包括分數影格率。若下游使用者需要，請在 Input 或上游 caps 邊界設定影格率；
設定編碼器的影格率不會建立輸入影格率。未知的影格率會保持未指定，
而不會被虛構的預設值取代。

影像發布會在準備邊界採用嚴格的投影：沒有 `OutputSpecProvider` 的節點會使繼承的
影格率變成未知，即使它實際上只是傳遞影格。明確的下游 caps 邊界可再次建立影格率。
這比一般盡力而為的 OutputSpec 投影更嚴格。變更已準備的輸入影格率需要新的執行個體；
有界的動態 Resize 輸入幾何尺寸並不代表允許變更影格率。

## 色彩與儲存

`ImageColorOptions` 選擇 BT.601 或 BT.709 矩陣系列。直接 YUV 到 YUV 操作支援
完整或有限範圍，以及各軸置中或共置的色度；它們保留編碼範圍與矩陣，不會進行轉換。
RGB/GRAY 到 YUV 僅支援完整範圍與置中色度。其他設定組合會在準備階段被拒絕。
這些選項不執行轉移函數轉換。封裝式 RGB/GRAY 輸出保留既有的完整範圍、置中 YUV
解讀方式；要求其他範圍／取樣位置設定會失敗，而不會暗中使用近似值。

多平面結果是**具有單一儲存擁有者的一張影像**。平面位移與正值列距來自
`GstVideoMeta`；使用者不可假設平面緊密排列。公開的 NV16 樣本帶有 Y 與 UV 平面，
其中 UV 的交錯位元組形狀為 `[height, width]`。

`ImageSurfaceLayout::Native` 使用緊密排列的 RGB/GRAY 列，並將 YUV 列距對齊為
16 位元組。它會保留 YUV 列填補及平面間的空隙。`EncoderNv12` 採用編碼器的佈局
權威來源：64 位元組對齊的列距，以及向上取整為八列倍數的亮度儲存高度。
EV74 在每次呼叫時都會寫入所有宣告的列尾及僅供儲存的列。不會加入逐影格 CPU
補零。此表面選項僅適用於 NV12；不會讓拒絕 NV16 的編解碼器新增 NV16 支援。

對於 YUV letterbox 填補，`ResizeOptions::yuv_padding` 為原始 `[Y, U, V]`。
若省略，輸出範圍決定黑色值：完整範圍為 `[0,128,128]`，有限範圍為 `[16,128,128]`。
`pad_value` 仍是封裝式 RGB/GRAY 的填充值，YUV 輸出時必須保持為零。

現有融合式 `Preproc` API 及其由模型推導的行為保持不變。
新的工廠函式為圖工具保留標準的 `Preproc` 節點種類。
