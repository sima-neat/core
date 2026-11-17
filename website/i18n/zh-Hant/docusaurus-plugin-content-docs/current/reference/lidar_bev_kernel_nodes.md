---
title: "LiDAR 與 BEV 核心節點"
description: "以圖節點執行 SimaVoxelize、SimaPillarScatter、SimaBEVPoolPlan 與 SimaBEVPool 核心函式庫運算"
sidebar_position: 8
---

# LiDAR 與 BEV 核心節點

`SimaVoxelize`、`SimaPillarScatter`、`SimaBEVPoolPlan` 與 `SimaBEVPool` 執行 PointPillars 與 BEVFusion
中非 MLA 的階段。每個節點都在您選擇的目標上執行一個 SiMa 核心函式庫運算。節點絕不會
退回其他目標執行，也不會為了改變配置而複製張量。若核心函式庫拒絕請求，圖的建置會失敗，
並附上核心函式庫提供的原因。

| 節點 | 運算 | 目標 |
| --- | --- | --- |
| `SimaVoxelize` | 將 LiDAR 點轉為經過裝飾與量化的柱體張量 | A65、EV74 |
| `SimaPillarScatter` | 將柱體特徵轉為密集的 BEV 畫布 | A65、EV74 |
| `SimaBEVPoolPlan` | 將相機校正資料轉為 BEV 池化計畫 | A65 |
| `SimaBEVPool` | 將相機深度與上下文特徵轉為 BEV 網格 | A65、EV74 |

以下是放在應用程式內的圖建構片段。請在應用程式中實作 `read_sweep()`，
傳回非空的 float 向量，依 x、y、z、時間偏移量的順序交錯排列，共 `4*N` 個值。
`N` 必須能以 INT32 表示。這個函式不是 Neat API。此片段只建構圖，不會提交影格
或取得輸出。

```cpp
#include <neat.h>

using namespace simaai::neat;

SimaVoxelizeOptions voxelize;
voxelize.voxel_size = {0.25f, 0.25f, 8.0f};
voxelize.point_cloud_range = {-50.0f, -50.0f, -5.0f, 50.0f, 50.0f, 3.0f};
voxelize.quantization.multiplier = 2.720007f; // 1 / the consuming MLA input's scale
voxelize.quantization.zero_point = 8;
voxelize.max_pillars = 40000;
voxelize.max_points_per_pillar = 64;
voxelize.target = SimaKernelTarget::EV74;

// One sample holds every input tensor of the node, in order.
InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

std::vector<float> xyzt = read_sweep();  // x, y, z, time offset per point
const auto n = static_cast<std::int64_t>(xyzt.size() / 4);
auto points = Tensor::from_vector(xyzt, {n, 4}, TensorMemory::EV74);
auto point_count = Tensor::from_vector(std::vector<std::int32_t>{static_cast<std::int32_t>(n)},
                                       {1}, TensorMemory::EV74);

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaVoxelize(voxelize));
graph.add(nodes::Output());
auto run = graph.build(TensorList{points, point_count});
```

Python 提供相同的選項型別，以及 `pyneat.nodes.sima_voxelize`、
`pyneat.nodes.sima_pillar_scatter`、`pyneat.nodes.sima_bev_pool_plan` 與
`pyneat.nodes.sima_bev_pool` 工廠函式。

## 輸入與輸出契約

每個節點依下表順序讀取輸入樣本中的張量，並依下表順序發布輸出。張量數量、資料型別與形狀
會在建置圖時檢查。

| 節點 | 輸入 | 輸出 |
| --- | --- | --- |
| `SimaVoxelize` | FP32 `points` `[N,4]`（x、y、z、時間位移）；INT32 `point_count` `[1]` | INT8 或 BF16 `decorated` `[1,M,T,10]`；以 (row, column) 表示的 INT32 `coordinates` `[M,2]`；INT32 `count` `[1]` |
| `SimaPillarScatter` | INT8 或 BF16 `features` `[M,C]`；INT32 `coordinates` `[M,2]`；INT32 `count` `[1]` | 與 `features` 型別相同的 `canvas` `[1,H,W,C]` |
| `SimaBEVPoolPlan` | FP32 `camera2lidar`、`camera_intrinsics` 與 `img_aug_matrix` `[N,4,4]`；FP32 `lidar_aug_matrix` `[4,4]` | UINT8 `plan` `[P]` |
| `SimaBEVPool` | INT8 或 BF16 `depth`；型別相同的 `context`；UINT8 `plan` `[P]` | INT8 或 BF16 `bev` `[1,nZ*C,nX,nY]` |

- **SimaVoxelize** 只讀取前 `point_count` 個點。`max_pillars`（`M`）與
  `max_points_per_pillar`（`T`）是固定容量，超出的點與柱體會被捨棄。未使用的列存放量化
  零點與零座標。
- **SimaPillarScatter** 只讀取前 `count` 個柱體。它也接受帶有額外單一維度軸的特徵，例如柱體
  特徵模型輸出的 `[1,M,1,C]`。`C` 必須是 16 的倍數。空白格存放 `zero_point`，此值必須是
  讀取畫布的模型其輸入量化對應到 0.0 的 INT8 代碼。INT8 畫布的位元組採用密集的 NHWC 順序。
- **SimaBEVPoolPlan** 從檢查點的視角轉換取得視錐（`xs`、`ys`、`ds`）與網格（`bx`、`dx`、
  `nx`）。預設的 `SimaBEVPoolPlanFormat::V1` 依 BEV 格子分組各點；若要供 EV74 的
  `SimaBEVPool` 使用，請將 `SimaBEVPoolPlanFormat::V2` 與 `context_channels` 搭配使用。根據
  預設，`P` 是最壞情況的計畫大小；計畫實際使用長度之後的位元組不會寫入。
- **SimaBEVPool** 針對每個格子的計畫點加總 `depth x context`，再以 `quantization` 量化總和。
  請將 `cells` 設為計畫的 `nx`、`plan_format` 設為計畫的 `format`，並將
  INT8 輸入時將 `quantization.multiplier` 設為 `depth_scale * context_scale / output_scale`。使用預設的
  `SimaBEVPoolStorage::ChannelsLast` 時，輸入會以相機模型產生的 `[N,fH,fW,D]` 與
  `[N,fH,fW,C]` 形式傳入。輸出會以帶有通道在後步幅的邏輯形狀 `[1,nZ*C,nX,nY]` 發布，
  因此其位元組為 NHWC `[1,nX,nY,nZ*C]`。`SimaBEVPoolStorage::Contiguous` 使用
  `[N,D,fH,fW]` 與 `[N,C,fH,fW]` 輸入以及密集輸出。

### 邏輯軸與實體儲存順序

`SimaBEVPool` 區分邏輯軸順序與實體位元組順序。標記為 `TensorLayout::HWC` 的張量
具有 `[N,H,W,K]` 邏輯軸，標記為 `TensorLayout::CHW` 的張量則具有 `[N,K,H,W]` 邏輯軸。
節點以標記為準，並檢查步幅是否符合 `input_storage`。在 `ChannelsLast` 模式下，
HWC 輸入必須密集，CHW 輸入則必須具有通道在後的步幅。在 `Contiguous` 模式下，
輸入必須是密集的 CHW 張量；HWC 標記會被拒絕。
也可以透過 `Tensor.axis_semantics` 宣告相同的標準軸順序；若同時提供兩種註記，
兩者必須一致。

未標記的 `ChannelsLast` 輸入沿用相容規則：密集 NHWC，或具有通道在後步幅的非密集
邏輯 NCHW。如果大小為 1 的維度讓邏輯 NCHW 檢視也呈密集排列，請明確設定
`Tensor.layout = TensorLayout::CHW`。這種情況無法只靠步幅辨識邏輯軸。Pool 輸出
帶有 CHW 邏輯軸與符合 `output_storage` 的步幅；節點不會為了改變排列而轉置或
複製承載資料。

### 量化中繼資料

節點知道量化參數時，INT8 輸出會在 `Tensor.semantic.quant` 中帶有該量化：`SimaVoxelize` 與
BF16 輸入的 `SimaBEVPool` 為 `scale = 1 / quantization.multiplier`；兩個 INT8 輸入都帶有
縮放比例的 `SimaBEVPool` 為 `depth_scale * context_scale / quantization.multiplier`；
`SimaPillarScatter` 則沿用特徵的量化。其他情況不會發布量化參數。

INT8 Scatter 輸入只支援最後一個特徵軸 `C` 的逐通道量化，包括 `[1,M,1,C]`
這類形狀。請提供 `C` 個有限正縮放比例與 `C` 個 INT8 零點；所有零點都必須等於
Scatter 用來填滿空格子的 `zero_point`。畫布保留這些值，並將量化軸對應至
`[1,H,W,C]` 的軸 `3`。其他量化軸會被拒絕：柱體依座標分散後，柱體軸參數無法
描述固定畫布的量化。

`SimaBEVPool` 不支援逐通道量化中繼資料，會明確拒絕而非悄悄移除。既有的逐張量
中繼資料與明確指定的零點選項維持原有行為；沒有輸入中繼資料時仍視為未知。

## BF16 張量

- **SimaVoxelize** 在 `output_dtype = TensorDType::BFloat16` 時，會將每個 binary32 裝飾值以
  最近偶數捨入為 BF16，將所有 NaN 存為 `0x7FC0`，並以 +0.0 填滿空白槽位。請將
  `quantization` 保留為預設值；其他值都會被拒絕。
- **SimaPillarScatter** 保留特徵的型別。BF16 特徵需要 `zero_point = 0`，因此空格子為
  +0.0。BF16 畫布的每個格子先儲存 `C` 個通道的低位元組，再儲存高位元組。
  `semantic.tess` 明確描述此實體配置：`slice_shape = [1,1,1,C]`、`axes = [0,1,2,3]`、
  `format = "MLA_TILED_BF16_COMPACT"`，而 `storage_bytes` 是已驗證的有效負載大小。
  D/H/W/C 區塊依 C/D/H/W 順序排列；區塊內依 D/H/W/C 順序儲存。BF16 使用各格子的
  低／高位元組平面，每個區塊均對齊 16 位元組。
  `copy_dense_bytes_tight()`、具型別的 `data_ptr()` 和 DLPack 會拒絕此非密集配置。
  `copy_payload_bytes()` 讀取指定範圍的儲存位元組；`clone()` 與 Owned 輸出保留位元組和
  中繼資料，不會自動去區塊化。`SimaArgMax` 等支援此配置的下游節點直接取得產生端
  已驗證的配置（限同一張圖）；只接受密集資料的使用端與不支援的區塊圖輸入會拒絕它。全通道 INT8 格子區塊與密集 NHWC
  逐位元組相同，因此仍支援一般密集複製和 DLPack。既有僅記錄格式的 `TessSpec` 註記
  維持原有行為；非零 `storage_bytes` 表示明確的區塊儲存配置。
- **SimaBEVPoolPlan** 接受 `input_dtype`，即 `SimaBEVPool` 輸入的型別。`V2` 計畫以位元組
  為單位配置 EV74 快取，因此 BF16 計畫可能較大。`V1` 計畫對兩種型別都相同。
- **SimaBEVPool** 接受零點為 0 的 BF16 `depth` 與 `context`。每個格子的總和依計畫中點的
  順序以 binary32 計算，因此 `V1` 與 `V2` 計畫可能產生不同的 BF16 位元組。BF16
  `bev` 會將每個總和以最近偶數捨入儲存，NaN 存為 `0x7FC0`，並需要預設的
  `quantization`。由 BF16 輸入產生的 INT8 `bev` 會量化總和：請將
  `quantization.multiplier` 設為 `1 / output_scale`。INT8 輸入無法產生 BF16 `bev`。

新增的 `TessSpec::axes` 與 `TessSpec::storage_bytes` 欄位具有預設值，因此現有的雙欄位聚合初始化仍保有原始碼相容性。不過，公開結構的二進位配置已改變：請使用相符的 Core 標頭與程式庫重新建置 C++ 消費端和 Python 繫結，不要重複使用以舊配置建置的二進位檔案。

## 將節點連接成管線

這些節點是一般的圖階段，但其輸入來自多個上游階段：

- 在 PointPillars 中，`SimaPillarScatter` 需要柱體特徵模型的柱體特徵，以及 `SimaVoxelize` 的
  `coordinates` 與 `count`。
- 在 BEVFusion 中，`SimaBEVPool` 需要相機模型的 `depth` 與 `context`，以及 `SimaBEVPoolPlan` 的
  計畫。

請依上述順序將這些張量合併成一個輸入樣本，再傳給節點。例如，在應用程式中取出兩者的結果，
再一起推送到節點所在的圖。

`SimaPillarScatter` 會複製特徵位元組而不重新量化，因此特徵必須已經使用讀取畫布的模型的輸入
量化。`SimaBEVPool` 會以一個 `[N,...]` 張量接收所有相機。

## 計畫與狀態

計畫只取決於校正資料與 `SimaBEVPoolPlanOptions`。設定 `cache_plan = true` 時，若某個影格的
校正資料位元組與前一個影格相同，就會複製前一個影格的計畫而不重新計算。此快取位於執行階段為
每個處理中影格所擁有的核心函式庫內容中，因此同時處理多個影格時，可能命中但不保證命中。
無論哪種情況，計畫位元組都相同。

`SimaBEVPool` 不會在影格之間保留狀態。它每次執行時都會檢查計畫的標頭與內容，並拒絕與自身
形狀不符的計畫。

## 目標限制

EV74 在 A65 契約之外另有下列限制：

- `SimaBEVPool` 需要通道數為 16 倍數的通道在後 `context`，以及通道在後的 `bev`。`V2`
  計畫還需要通道在後的 `depth`。
- `SimaBEVPoolPlan` 只在 A65 上執行。

某個目標不支援的請求會在該目標上失敗。節點不會改為替您選擇其他目標。
