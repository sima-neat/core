---
title: "EV74 視覺前端節點"
description: "FeatureHistogram、GriderFast、TrackDescriptor、TrackKLT 與 MetoakDepth 的 Neat Graph 用法"
sidebar_position: 8
---

# EV74 視覺前端節點

Neat 將 EV74 視覺前端圖表以一般方式呈現。 `Graph` 節點。請使用公開的節點工廠和選項結構體；請勿直接呼叫。 `processcvu`，或直接從應用程式碼中使用 ConfigManager 或分派器 API。

| 節點工廠 | 圖的名稱 | 圖的 ID | 目的 |
| --- | --- | ---: | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | `feature_histogram` | 235 | 灰階影像直方圖 |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | `grider_fast` | 236 | 網格分佈的 FAST 特徵 |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | `track_descriptor` | 237 | 快速特徵加上描述符 |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | `track_klt` | 238 | 金字塔式 KLT 追蹤，可選擇性地包含偵測到的替換特徵 |
| `nodes::MetoakDepth` | `simor_depth_map` | 20 | 將 I420 與視差轉為 RGB、公制深度及 XYZ |

圖表 ID 對於診斷以及韌體/套件一致性檢查非常有用。它們並非應用程式碼中必需的。

## 張量收縮

特徵與追蹤張量使用**邏輯批次形狀**。如果 `batch_size == B`，則灰階影像的形狀為 `[B,H,W]`，而不是 `[B*H,W]`。執行階段會內部處理所有 EV74 傳輸封裝。

| 節點 | 輸入 | 公開輸出 |
| --- | --- | --- |
| `FeatureHistogram` | `input_image`: UInt8 `[B,H,W]` | `output_hist`: Int32 `[B,256]` |
| `GriderFast` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]` |
| `TrackDescriptor` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]`; `output_descriptors`: Int32 `[B,max_features,8]` |
| `TrackKLT` | `prev_image`: UInt8 `[B,H,W]`; `cur_image`: UInt8 `[B,H,W]`; `input_points`: Int32 `[B,num_points,2]` | `output_points`: Float32 `[B,num_points,2]`; `output_status`: Int32 `[B,num_points,1]`; 此外，還有 `output_features`: Int32 `[B,1 + max_features*3]`，僅在 `detect_new_features != 0`時提供。 |

「特徵列表」張量採用此批次內佈局：

```text
[count, x0, y0, score0, x1, y1, score1, ...]
```

目前的描述圖需要 `descriptor_words == 8`。更改此設定會導致 EV74 ABI 發生變更，並且在發送之前會被拒絕。

## C++ 快速入門

```cpp
#include <neat.h>

#include <cstdint>
#include <vector>

using namespace simaai::neat;

Tensor make_gray_batch(int width, int height, int batch) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * batch);
  // Fill pixels in batch-major order: b*height*width + y*width + x.
  auto tensor = Tensor::from_vector(pixels, {batch, height, width}, TensorMemory::EV74);
  tensor.layout = TensorLayout::HW;
  tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H, TensorAxisSemantic::W};
  tensor.route.name = "input_image";
  tensor.route.segment_name = "input_image";
  return tensor;
}

int main() {
  constexpr int width = 320;
  constexpr int height = 240;
  constexpr int batch = 2;

  Graph graph;

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = width;
  input.height = height;
  input.depth = 1;
  input.max_width = width;
  input.max_height = height * batch; // transport capacity; public tensor remains [B,H,W]
  input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "input_image";

  graph.add(nodes::Input(input));

  GriderFastOptions fast;
  fast.width = width;
  fast.height = height;
  fast.batch_size = batch;
  fast.max_features = 64;
  fast.threshold = 30;
  graph.add(nodes::GriderFast(fast));

  graph.add(nodes::Output());

  RunOptions run_opt;
  run_opt.output_memory = OutputMemory::Owned;

  Tensor image = make_gray_batch(width, height, batch);
  Run run = graph.build({image}, run_opt);
  TensorList outputs = run.run({image}, /*timeout_ms=*/30000);
  run.close();
}
```

## 具有三個輸入的 KLT

`TrackKLT` 會處理一組張量：先前的影像、目前的影像以及輸入點。請為這些路徑命名，使其與選項欄位相符。

```cpp
TrackKLTOptions klt;
klt.width = 320;
klt.height = 240;
klt.batch_size = 2;
klt.num_points = 32;
klt.max_features = 64;
klt.detect_new_features = 1; // publish output_features as the third output

graph.add(nodes::TrackKLT(klt));
```

預期在啟用 `detect_new_features == 1` 時的公開輸出：

```text
output_points   Float32 [2,32,2]
output_status   Int32   [2,32,1]
output_features Int32   [2,193]
```

當 `detect_new_features == 0` 啟動時，Neat 只會發布 `output_points` 和 `output_status`；可供 EV 檢視的功能緩衝區仍為內部執行階段設定。

## 六個輸入的 Metoak 深度

`MetoakDepth` 是使用 `simor_depth_map`（圖 20）的 C++ 專用節點。它接收已解碼的 I420 平面、原始視差及每影格校正，而非原始 SIMOR 相機影格。應用程式或 ROS 配接器必須在此節點之前解包 SIMOR 並選擇校正資料；Neat 不會取代該配接器。

使用 `[8,2048]` 內的偶數 `width` 及 `[8,1536]` 內的偶數 `height`。S315 原生深度尺寸為 `640x360`。批次固定為一，不含前導批次維度。保留以下標準路由名稱與輸入順序；別名會在合約編譯時遭拒絕。

| 輸入路由 | 型別 | 形狀 | 意義 |
| --- | --- | --- | --- |
| `y_src` | UInt8 | `[H,W]` | I420 Y |
| `u_src` | UInt8 | `[H/2,W/2]` | I420 U |
| `v_src` | UInt8 | `[H/2,W/2]` | I420 V |
| `disp_src` | UInt16 | `[H,W]` | 原始視差；固定次像素倍率 32 |
| `bf_mm_src` | Float32 | `[1]` | 校正後基線 × 焦距，單位 mm |
| `proj_src` | Float32 | `[3]` | 投影 `{fx_fy,cx,cy}` |

| 輸出路由 | 型別 | 形狀 | 意義 |
| --- | --- | --- | --- |
| `rgb_dst` | UInt8 | `[H,W,3]` | 交錯 RGB |
| `depth_dst` | UInt16 | `[H,W]` | 深度，單位 mm；0 表示無效 |
| `points_dst` | Float32 | `[H,W,3]` | 交錯 XYZ，單位公尺；NaN 表示無效 |

三個輸出一律一起發布；`depth_dst` 是主要邊界描述，而非輸出選擇器。校正 BF 與焦距必須為正的有限值，主點座標必須為有限值。

使用與輸入表相符、位於 EV74 記憶體的六個具名張量建立 Graph。此範例設定節點；請從配接器提供實際解碼影格及校正張量。

```cpp
#include <neat.h>

using namespace simaai::neat;

// inputs contains the six named, decoded EV74 tensors from the table above.
Run build_metoak_depth(const TensorList& inputs) {
  Graph graph;
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.caps_override =
      "application/vnd.simaai.tensor, representation=(string)tensor-set, storage=(string)tensorbuffer";
  graph.add(nodes::Input(input));

  MetoakDepthOptions depth;
  depth.width = 640;
  depth.height = 360;
  graph.add(nodes::MetoakDepth(depth));
  graph.add(nodes::Output());

  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  return graph.build(inputs, options);
}
```

僅在包含圖 20 的相符 Internals 與 EV74 韌體上執行此 Graph。下方特徵/追蹤驗證命令涵蓋其他四個圖，不包含 `MetoakDepth`。

## Python 表面

四個特徵/追蹤節點提供符合 C++ 選項/工廠風格的 Python 繫結。`MetoakDepth` 目前沒有 Python 繫結。建立選項物件、設定公開組態，並將節點加入 `Graph`。

```python
import numpy as np
import pyneat

width, height, batch = 320, 240, 2

opt = pyneat.GriderFastOptions()
opt.width = width
opt.height = height
opt.batch_size = batch
opt.max_features = 64
print(opt.summary())

graph = pyneat.Graph()
input_opt = pyneat.InputOptions()
input_opt.payload_type = pyneat.PayloadType.Tensor
input_opt.format = pyneat.Format.UINT8
input_opt.width = width
input_opt.height = height
input_opt.max_width = width
input_opt.max_height = height * batch
input_opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
input_opt.buffer_name = "input_image"

graph.add(pyneat.nodes.input(input_opt))
graph.add(pyneat.nodes.grider_fast(opt))
graph.add(pyneat.nodes.output())

image_np = np.zeros((batch, height, width), dtype=np.uint8)
image = pyneat.Tensor.from_numpy(image_np, memory="ev74")
image.layout = pyneat.TensorLayout.HW
# If setting route metadata from Python in a custom app, keep it aligned with
# the option names used above.
```

## 安全檢查

四個特徵/追蹤節點會在 EV 分派之前驗證圖的邊界。它們會拒絕：

- 非正數的尺寸或數量；
- 不支援的批次大小；
- 超出 `[0,255]` 的閾值；
- 重複或空白的張量名稱；
- `TrackDescriptorOptions.descriptor_words != 8`;
- 無效的 KLT 視窗、層級和偵測模式值；
- 在預先分派協商期間，執行階段的張量大小不足。

這很重要，因為不合法的緩衝區可能會導致 EV74 出現問題。請將驗證失敗視為主機端的錯誤，並且不要繞過節點合約路徑。

## 快速驗證指令

快速的客戶端式 DevKit 閘道是：

```bash
ctest --test-dir /workspace/core_graph_changes/build/tests \
  -R visual_frontend_ --output-on-failure
```

它正在執行：

- 包含所有四個視覺圖，分別為 `320x240`、`batch_size=2` 和 `detect_new_features=1`。
- 針對特定 KLT 進行的無偵測 ABI 檢查；
- 一個在發送之前執行的負面安全檢查，用於確認是否存在非法的批次輸入。
  在 EV74 之前已被拒絕。
