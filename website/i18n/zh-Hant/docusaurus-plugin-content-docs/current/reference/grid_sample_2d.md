---
title: 使用動態網格取樣特徵
description: 透過 Neat 圖執行 A65 NEON GridSample2D
sidebar_position: 9
---

# 使用動態網格取樣特徵

`SimaGridSample2D` 在 A65 上進行雙線性取樣。每個輸入樣本依序包含特徵 `[N,H,W,C]`（FP32 或 BF16），以及帶有正規化 `(x,y)` 座標的網格 `[N,Ho,Wo,2]`（FP32）。輸出為 `sampled` `[N,Ho,Wo,C]`，採用與特徵相同的資料型別與通道置後佈局。所有維度皆須為正值。

## 建置與執行

使用包含 `grid_sample_2d` 且互相匹配的 Kernels、Runner、Scheduler 與 Core 套件組。此 C++ 片段在連結 `SimaNeat::sima_neat` 的應用程式內執行；建置圖之前，先在兩個向量中填入連續 NHWC 特徵值與 NHW2 網格值。

```cpp
#include <neat.h>
using namespace simaai::neat;

std::vector<float> features(1 * 16 * 16 * 16, 1.0f);
std::vector<float> grid(1 * 32 * 8 * 2, 0.0f);
auto feature = Tensor::from_vector(features, {1,16,16,16}, TensorMemory::CPU);
feature.layout = TensorLayout::HWC;
auto coordinates = Tensor::from_vector(grid, {1,32,8,2}, TensorMemory::CPU);

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::SystemMemory;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";
SimaGridSample2DOptions sampling;
sampling.padding = SimaGridSample2DPadding::Zeros;
sampling.align_corners = false;
Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaGridSample2D(sampling));
graph.add(nodes::Output());
auto run = graph.build(TensorList{feature, coordinates});
auto output = run.run(TensorList{feature, coordinates}, 30000);
run.close();
```

保持形狀與資料型別固定，即可在每次提交時變更網格值。多個在途樣本使用一般非同步 push/pull API。保留的零複製輸出會持有生產端緩衝區借用，須釋放才能讓池繼續運作。擁有資料的輸出可獨立於執行物件保持有效。

## 取樣選項與精度

`SimaGridSample2DPadding::Zeros` 將無效特徵取樣點替換為零；`SimaGridSample2DPadding::Border` 將座標限制到邊緣像素中心。`align_corners=false` 使用半像素中心；true 將正規化 -1/+1 映射到角點像素中心。NaN 網格分量代表正規化 -1。無限大分量在零填補下產生零樣本，在邊界填補下選擇邊緣。

座標、權重與累加使用 FP32。每個總和先做一次乘法，再做三次明確的融合乘加。BF16 特徵精確擴寬，最終輸出僅做一次最接近偶數捨入。此精度與將網格值或中間座標捨入至 BF16 的 MLA 實作不同。

NCHW、分塊儲存、非密集檢視與非恆等量化都須在此節點之前明確轉換。不支援最近鄰、三次與反射模式。加入此節點不會取代已編譯 MLA 模型中的 GridSample 運算。效能比較須使用相同的已儲存輸入與計時範圍；網格初始化未知的 ELF 剖析，其網格來源尚未確認。