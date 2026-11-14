---
title: "語意分割解碼"
description: "使用 SimaResizeArgmax 或 Model 語意分割選項，在 EV74 上將各類別分數轉換為縮放後的類別圖"
sidebar_position: 9
---

# 語意分割解碼

`SimaResizeArgmax` 會將分割模型的各類別分數轉換為您所選尺寸的類別圖。它會為每個輸出像素
以雙線性方式內插每個類別的分數，並傳回分數最大的類別。這正是 SegFormer 與類似模型在
MLA 之後需要的解碼：先以 `interpolate(mode="bilinear")` 縮放到目標尺寸，再對類別取
`argmax`。它產生的類別圖與 Hugging Face
`SegformerImageProcessor.post_process_semantic_segmentation` 搭配 `target_sizes` 的結果相同，
且永遠不會儲存縮放後的分數。

解碼在 EV74 上執行，沒有 A65 實作，也沒有後援路徑。

## 解碼模型輸出

設定 `Model::Options::semantic_segmentation`，即可以此解碼取代模型的 dequantize 或 cast 後處理。
節點會就地讀取 MLA 分數，包括儲存通道有填補的 MLA 輸出，而模型會傳回一張類別圖，而非分數。

```cpp
#include <neat.h>

using namespace simaai::neat;

Model::Options options;
options.semantic_segmentation.enable = true;
options.semantic_segmentation.output_width = 1280; // camera frame size
options.semantic_segmentation.output_height = 720;
Model model("segformer_b0_ade_512_mpk.tar.gz", options);
```

```python
import pyneat

options = pyneat.ModelOptions()
options.semantic_segmentation.enable = True
options.semantic_segmentation.output_width = 1280  # camera frame size
options.semantic_segmentation.output_height = 720
model = pyneat.Model("segformer_b0_ade_512_mpk.tar.gz", options)
```

請將這些選項與模型所需的前處理選項搭配使用。模型輸出是單一類別圖張量 `[N,H,W]`，
且 `model.info().selection.selected_post_kind` 為 `"resize_argmax"`。

- 將 `output_width` 與 `output_height` 都保留為 0 以使用模型輸入尺寸，或兩者都設為來源
  影格尺寸，讓每個來源像素都取得類別。
- 預設的 `align_corners = false` 與 PyTorch 及 Hugging Face 相同，使用半像素中心。
  只有在模型以角落對齊內插訓練時，才設為 `true`。
- 模型的後處理只能包含 dequantize 或 cast 階段，且不得設定 `decode_type`。否則 `Model`
  建構函式會擲回 `std::invalid_argument`（在 Python 中為 `ValueError`）。

## 在圖中使用節點

C++ 的 `nodes::SimaResizeArgmax(...)` 與 Python 的 `pyneat.nodes.sima_resize_argmax(options)`
會對任何符合下列合約的張量執行相同解碼。此片段會將一個分數張量 `scores` 解碼為 512 x 512
的類別圖：

```cpp
SimaResizeArgmaxOptions decode;
decode.output_width = 512;
decode.output_height = 512;

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaResizeArgmax(decode));
graph.add(nodes::Output());
auto run = graph.build(TensorList{scores});
TensorList class_map = run.run(TensorList{scores}, 1000);
```

## 輸入與輸出合約

| 埠 | 張量 |
| --- | --- |
| `scores` | `[h,w,C]` 或 `[N,h,w,C]`，類別位於最後一軸；INT8、UINT8、BF16 或 FP32；密集，或像素間距大於 `C` 的跨步 |
| `class_map` | `[H,W]` 或 `[N,H,W]`，密集；`C` 不超過 256 時為 UINT8，否則為 INT32 |

- 整數分數不經反量化即直接比較。當單一正值比例套用於所有類別時，結果與反量化後的分數
  相同。具有逐軸量化或比例非正值的分數，會在建置圖時遭到拒絕。
- 分數相同時選擇最小的類別索引。對 BF16 與 FP32 分數而言，內插結果為 NaN 者優先，並選擇
  第一個 NaN 類別。
- 儲存時的填補通道（例如有填補 MLA 輸出的額外通道）永遠不會被當成類別讀取。
- MLA 平鋪格式的分數會遭到拒絕；請先進行反鑲嵌。具有邏輯 CHW 或 NCHW 軸的分數也會遭到
  拒絕；類別必須位於最後一軸。
- DevKit 必須執行包含此運算的 EV74 韌體。使用較舊韌體時，會在提交第一個影格之前失敗。
