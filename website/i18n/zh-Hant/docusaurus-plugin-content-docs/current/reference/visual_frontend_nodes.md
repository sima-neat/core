---
title: "EV74 視覺前端節點"
description: "FeatureHistogram、GriderFast、TrackDescriptor 與 TrackKLT 的公開 Graph 契約"
sidebar_position: 8
---

# EV74 視覺前端節點

請透過公開 `Graph` 節點工廠使用 EV74 運算，並安裝相符的 Core、Scheduler、Runner、Kernels 與 EV74 韌體套件。舊版圖 ID 不能用於檢查原生韌體相容性；應用程式不需要分派器 ID。

品質選項保留既有原始碼用法與預設輸出位置，但會變更 `TrackKLTOptions` 的 C++ 配置，因此既有 C++ 二進位檔不具 ABI 相容性。請針對相符的 Core 版本重新建置 C++ 應用程式與 Python 擴充套件。v2 追蹤器要求須使用相符的 Kernels、Runner、Scheduler 與 EV74 韌體組合。既有的原生 v1 要求仍受支援，但這不代表 v2 要求與舊版韌體相容。

| 節點工廠 | 用途 |
| --- | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | 灰階直方圖 |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | 網格分布的 FAST 特徵點 |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | FAST 特徵點與 BRIEF 描述子 |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | 金字塔 KLT 追蹤及選用的目前影格特徵偵測 |

## 張量契約

影像張量的邏輯形狀是 `[B,H,W]`，其中 `B` 為 `batch_size`。不要將批次併入影像高度。裝置輸入可使用獨立配置或有效的共用儲存檢視，不需要 CPU 複製的替代方案。

| 節點 | 輸入 | 預設公開輸出 |
| --- | --- | --- |
| `FeatureHistogram` | UInt8 `[B,H,W]` | Int32 `[B,256]` |
| `GriderFast` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]` |
| `TrackDescriptor` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]`; Int32 `[B,max_features,8]` |
| `TrackKLT` | UInt8 `[B,H,W]`; UInt8 `[B,H,W]`; Int32 `[B,num_points,2]` | Float32 `[B,num_points,2]`; Int32 `[B,num_points,1]` |

每個批次項目的特徵格式為 `[count,x0,y0,score0,...]`。只讀取前 `count` 筆記錄，未使用容量不是特徵清單。TrackDescriptor 要求 `descriptor_words == 8`；每列描述子對應相同列的已選特徵。

## 網格選擇與邊界

`grid_x` 和 `grid_y` 將整張影像分成網格。容量為 `K`、網格數為 `M` 時，每格有 `K / M` 個名額，列優先順序的前 `K % M` 格各多一個。每格選擇最強分數，再將結果按分數遞減排列，同分按 `(y,x)` 遞增排列。稀疏網格不出借名額。`min_px_dist` 會跨網格邊界抑制過近的點，距離恰等於設定值則接受。抑制後不補入其他候選，因此結果可能少於 `max_features`。

一般 FAST 包含 `x=3`、`y=3`、`x=W-4` 與 `y=H-4` 的中心點。TrackDescriptor 需要完整的 31×31 BRIEF 區塊，因此中心必須距各邊至少 15 像素。第 3 列沒有描述子是預期行為。

## KLT 輸入與輸出

請依 **前一張影像、目前影像、輸入點** 順序傳入 `TensorList`。在此直接公開多輸入路徑上，非空的邏輯 `route.name` 必須符合設定的位置，不會重新排序清單。順序正確的未命名張量仍有效。缺少輸入、名稱矛盾、型別或形狀錯誤，以及無法存取的範圍，都會在提交前拒絕，後續影格也一樣。

重用相容的輸出張量時，建立淺層複本，僅依新公開位置重新命名或清除 `route.name`。請保留儲存、位移與實體路由中繼資料。此規則不會重新命名內部圖連接，也不取代明確的模型輸入路由處理器。

TrackKLT 預設公開點與狀態。`detect_new_features != 0` 會附加 **目前** 影像的偵測特徵。停用偵測時不配置隱藏特徵輸出。`publish_error = true` 在既有輸出後附加 Float32 `[B,num_points,1]`：未啟用偵測時索引為 2，啟用時為 3。既有輸出位置不變。

### C++ 範例

提供兩個 `640×480` UInt8 影像向量及 `64×2` Int32 初始座標。以下各輸入都有獨立 EV74 儲存，傳回的組依序為點、狀態、誤差。

```cpp
#include <neat.h>
#include <cstdint>
#include <vector>

using namespace simaai::neat;

TensorList track_pair(const std::vector<std::uint8_t>& previous,
                     const std::vector<std::uint8_t>& current,
                     const std::vector<std::int32_t>& xy) {
  constexpr int width = 640, height = 480, count = 64;
  auto image = [&](const auto& pixels, const char* name) {
    auto tensor = Tensor::from_vector(pixels, {1, height, width}, TensorMemory::EV74);
    tensor.layout = TensorLayout::HW;
    tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H,
                            TensorAxisSemantic::W};
    tensor.route.name = name;
    return tensor;
  };
  auto points = Tensor::from_vector(xy, {1, count, 2}, TensorMemory::EV74);
  points.route.name = "input_points";
  TensorList inputs{image(previous, "prev_image"), image(current, "cur_image"), points};

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = input.max_width = width;
  input.height = input.max_height = height;
  input.depth = input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "prev_image";
  input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                        "storage=(string)tensorbuffer";

  TrackKLTOptions klt;
  klt.width = width;
  klt.height = height;
  klt.num_points = count;
  klt.win_half = 7;
  klt.max_level = 2;
  klt.max_iters = 20;
  klt.publish_error = true;

  Graph graph;
  graph.add(nodes::Input(input));
  graph.add(nodes::TrackKLT(klt));
  graph.add(nodes::Output());
  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  Run run = graph.build(inputs, options);
  TensorList outputs;
  try {
    outputs = run.run(inputs, 30000);
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  return outputs;
}
```

## 品質控制

| 選項 | 預設值 | 意義 |
| --- | --- | --- |
| `publish_error` | `false` | 公開最終座標的平均絕對亮度殘差。`output_error_name` 預設為 `output_error`。 |
| `error_threshold` | `-1.0` | `-1` 表示停用，否則為 `[0,255]` 的有限門檻。殘差大於門檻時拒絕，相等時接受。未公開誤差時也可過濾。 |
| `min_eig_threshold` | `0.0` | `0` 表示停用，否則為有限正值的正規化最小特徵值門檻，用於紋理條件判定。 |
| `require_convergence` | `false` | 要求最後的全解析度更新符合收斂條件。設為 false 時，達到反覆運算上限不代表失敗。 |

殘差是亮度差，不是像素位移或信心值。它在傳回座標上評估；`+inf` 表示未評估有效殘差。紋理指標將中央差分梯度矩陣的最小特徵值除以視窗面積及 `255²`，其尺度不可與 OpenCV 門檻互換。請使用代表性校正資料選擇門檻，並同時驗證有效追蹤保留率及錯誤追蹤拒絕率。

使用座標前請檢查狀態。求解前無效的初始點保留原座標且狀態為 0。有限的影像外計算點，或被品質門檻拒絕的候選，可能保留計算座標且狀態為 0。預設狀態只代表求解與幾何有效性，不保證真實世界對應正確。

## 移動、邊界與固定點數

細化位置離開目前影像區塊時，會重新置中並載入區塊，不會默默限制在 ±4 像素搜尋範圍。重新載入及反覆運算次數仍有上限。邊界區塊只從影像內進行 DMA，並複製邊緣像素。這會移除舊版隨金字塔層級改變的排除帶，但不保證任意移動、遮蔽或模糊紋理都可追蹤。

Core 初始點仍為 Int32，結果為 Float32。`num_points` 在建圖時固定。未使用的位置請填入 `(-1,-1)` 等無效座標並檢查狀態。`(0,0)` 是有效邊緣座標，不是停用標記。每次呼叫都會重建兩張影像的金字塔，不表示會重用前一影格金字塔。

## Python 選項

Python 提供相同選項。使用 `Tensor.from_numpy(..., memory="ev74")` 建立輸入張量，維持輸入順序契約，並如 C++ 一樣建立公開節點。

```python
import pyneat

options = pyneat.TrackKLTOptions()
options.width, options.height = 640, 480
options.num_points = 64
options.win_half, options.max_level, options.max_iters = 7, 2, 20
options.publish_error = True
options.error_threshold = -1.0
options.min_eig_threshold = 0.0
options.require_convergence = False
node = pyneat.nodes.track_klt(options)
```

## 驗證

設定的連接埠名稱必須非空且唯一。無效尺寸、點數、視窗、金字塔層級、描述子大小，以及非有限或超出範圍的品質參數都會被拒絕。請勿略過檢查，也不要只改張量型別來轉換值。

## 執行正確性測試

在 DevKit 上，將 `CORE_TEST_DIR` 設為已安裝的 Core 測試目錄或原生建置的測試目錄，再執行：

```bash
ctest --test-dir "$CORE_TEST_DIR" \
  -R '^(grider_fast|track_descriptor|track_klt)_ev_test$' --output-on-failure
```

每個註冊測試都會執行該核心的完整案例。GriderFast 涵蓋網格、距離與 FAST 邊界；TrackDescriptor 也檢查描述子的精確對應；TrackKLT 涵蓋獨立輸入、影格重用、驗證、品質、移動、邊界及偵測器。獨立的快速預設是直方圖、未偵測 KLT 與批次拒絕的基本測試，不能代替數值驗證。

## Metoak 深度（C++）

`nodes::MetoakDepth` 恢復既有的 `MetoakDepthOptions` API。
它接收六個具名張量，並同時發布 RGB、深度及 XYZ。
此節點執行 EV74 運算，而非攝影機擷取；沒有攝影機也能使用合成張量。

與上述支援批次的節點不同，Metoak 固定使用批次一，且沒有前置批次維度。
將 `W` 設為 `[8,2048]` 內的偶數，`H` 設為 `[8,1536]` 內的偶數。
保留標準路由名稱，並依下列順序提供輸入。

| 張量 | 資料型別 | 形狀 |
| --- | --- | --- |
| y_src | UInt8 | [H,W] |
| u_src, v_src | UInt8 | [H/2,W/2] |
| disp_src | UInt16 | [H,W] |
| bf_mm_src | Float32 | [1] |
| proj_src | Float32 | [3] |
| rgb_dst | UInt8 | [H,W,3] |
| depth_dst | UInt16 | [H,W] |
| points_dst | Float32 | [H,W,3] |

`y_src`、`u_src`、`v_src` 為各自獨立的 I420 平面。視差縮放倍率為 32。
每個影格都需提供正有限值 `bf_mm_src`（基線 × 焦距，mm）及
`proj_src = {fx_fy, cx, cy}`；`fx_fy` 必須為正值，三個值皆須有限。
校正值可以在影格之間變更。

深度會四捨五入為 mm，並在 UInt16 上限飽和；XYZ 使用未捨入的深度，單位為 m。
視差為零時，深度為零，XYZ 為 NaN。即使深度是主要輸出描述，三個輸出仍可全部取得。

```cpp
MetoakDepthOptions depth;
depth.width = 640;
depth.height = 360;
graph.add(nodes::MetoakDepth(depth));
```

既有的 `debug`、`num_buffers`、`element_name` 欄位仍可使用。
`num_buffers = 0` 使用圖的預設容量；正值指定此階段的執行容量。
此次恢復的是 C++ 公開 API，不會新增 Python Metoak 工廠函式。
