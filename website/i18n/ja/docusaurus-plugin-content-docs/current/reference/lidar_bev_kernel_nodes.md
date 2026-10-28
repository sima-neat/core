---
title: "LiDAR と BEV のカーネルノード"
description: "SimaVoxelize、SimaPillarScatter、SimaBEVPoolPlan、SimaBEVPool のカーネルライブラリ演算をグラフノードとして実行する"
sidebar_position: 8
---

# LiDAR と BEV のカーネルノード

`SimaVoxelize`、`SimaPillarScatter`、`SimaBEVPoolPlan`、`SimaBEVPool` は、PointPillars と BEVFusion の
MLA 以外のステージを実行します。各ノードは、選択したターゲット上で SiMa カーネル
ライブラリの演算を 1 つ実行します。ノードが別のターゲットへフォールバックしたり、
レイアウトを変えるためにテンソルをコピーしたりすることはありません。カーネル
ライブラリがリクエストを拒否した場合、グラフのビルドはカーネル側の理由とともに
失敗します。

| ノード | 演算 | ターゲット |
| --- | --- | --- |
| `SimaVoxelize` | LiDAR 点群から、装飾済みで量子化されたピラーテンソルへ | A65、EV74 |
| `SimaPillarScatter` | ピラー特徴量から密な BEV キャンバスへ | A65、EV74 |
| `SimaBEVPoolPlan` | カメラキャリブレーションから BEV プーリングプランへ | A65 |
| `SimaBEVPool` | カメラの深度特徴量とコンテキスト特徴量から BEV グリッドへ | A65、EV74 |

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

Python では同じオプション型と、`pyneat.nodes.sima_voxelize`、
`pyneat.nodes.sima_pillar_scatter`、`pyneat.nodes.sima_bev_pool_plan`、
`pyneat.nodes.sima_bev_pool` のファクトリーを使用できます。

## 入力と出力の契約

各ノードは入力サンプルのテンソルを下表の順序で読み取り、出力を下表の順序で公開
します。テンソルの数、データ型、形状はグラフのビルド時に検査されます。

| ノード | 入力 | 出力 |
| --- | --- | --- |
| `SimaVoxelize` | FP32 `points` `[N,4]`（x、y、z、時刻オフセット）、INT32 `point_count` `[1]` | INT8 または BF16 の `decorated` `[1,M,T,10]`、(row, column) 形式の INT32 `coordinates` `[M,2]`、INT32 `count` `[1]` |
| `SimaPillarScatter` | INT8 または BF16 の `features` `[M,C]`、INT32 `coordinates` `[M,2]`、INT32 `count` `[1]` | `features` と同じ型の `canvas` `[1,H,W,C]` |
| `SimaBEVPoolPlan` | FP32 `camera2lidar`、`camera_intrinsics`、`img_aug_matrix` `[N,4,4]`、FP32 `lidar_aug_matrix` `[4,4]` | UINT8 `plan` `[P]` |
| `SimaBEVPool` | INT8 または BF16 の `depth`、同じ型の `context`、UINT8 `plan` `[P]` | INT8 または BF16 の `bev` `[1,nZ*C,nX,nY]` |

- **SimaVoxelize** は先頭の `point_count` 個の点だけを読み取ります。`max_pillars`（`M`）と
  `max_points_per_pillar`（`T`）は固定の容量で、それを超える点とピラーは破棄されます。
  未使用の行には量子化のゼロポイントとゼロ座標が入ります。
- **SimaPillarScatter** は先頭の `count` 個のピラーだけを読み取ります。ピラー特徴量モデルが
  出力する `[1,M,1,C]` のように、余分な大きさ 1 の軸を持つ特徴量も受け付けます。`C` は
  16 の倍数である必要があります。空のセルには `zero_point` が入ります。これは、
  キャンバスを読むモデルの入力量子化が 0.0 に割り当てる INT8 コードでなければ
  なりません。INT8 キャンバスのバイトは密な NHWC 順です。
- **SimaBEVPoolPlan** は、錐台（`xs`、`ys`、`ds`）とグリッド（`bx`、`dx`、`nx`）を
  チェックポイントのビュー変換から取得します。既定の `SimaBEVPoolPlanFormat::V1` は
  点を BEV セルごとにまとめます。EV74 の `SimaBEVPool` には、`context_channels` とともに
  `SimaBEVPoolPlanFormat::V2` を使用します。既定では `P` は最悪ケースのプランサイズで、
  プランが使用する長さを超えるバイトは書き込まれません。
- **SimaBEVPool** は各セルのプラン点について `depth x context` を合計し、その和を
  `quantization` で量子化します。`cells` にはプランの `nx` を、`plan_format` には
  プランの `format` を、INT8 入力の場合は `quantization.multiplier` に
  `depth_scale * context_scale / output_scale` を設定します。既定の
  `SimaBEVPoolStorage::ChannelsLast` では、入力はカメラモデルが出力するとおり
  `[N,fH,fW,D]` と `[N,fH,fW,C]` で渡されます。出力はチャネルラストのストライドを持つ
  論理形状 `[1,nZ*C,nX,nY]` として公開されるため、そのバイトは NHWC
  `[1,nX,nY,nZ*C]` です。`SimaBEVPoolStorage::Contiguous` は `[N,D,fH,fW]` と
  `[N,C,fH,fW]` の入力と、密な出力を使用します。

INT8 の出力は、ノードが量子化を把握している場合、それを `Tensor.semantic.quant` に
保持します。`SimaVoxelize` と BF16 入力の `SimaBEVPool` では
`scale = 1 / quantization.multiplier`、INT8 入力の両方がスケールを持つ `SimaBEVPool` では
`depth_scale * context_scale / quantization.multiplier`、`SimaPillarScatter` では特徴量の
量子化です。それ以外の場合、量子化は公開されません。

## BF16 テンソル

- **SimaVoxelize** は `output_dtype = TensorDType::BFloat16` のとき、binary32 の各装飾値を
  最近接偶数丸めで BF16 にし、NaN はすべて `0x7FC0` として格納し、空のスロットを +0.0 で
  埋めます。`quantization` は既定値のままにしてください。それ以外の値は拒否されます。
- **SimaPillarScatter** は特徴量の型を保持します。BF16 の特徴量には `zero_point = 0` が
  必要なため、空のセルには +0.0 が入ります。BF16 キャンバスの各セルは `C` チャネルの
  下位バイトを先に、上位バイトを後に格納します。`semantic.tess` はこの物理配置を
  明示します。`slice_shape = [1,1,1,C]`、`axes = [0,1,2,3]`、
  `format = "MLA_TILED_BF16_COMPACT"` で、`storage_bytes` は検証済みのペイロード長です。
  D/H/W/C タイルは C/D/H/W 順に並び、タイル内は D/H/W/C 順です。BF16 はセルごとの
  下位・上位バイト面を使用し、各タイルは 16 バイト境界に整列します。
  `copy_dense_bytes_tight()`、型付き `data_ptr()`、DLPack は、この非密配置を拒否します。
  `copy_payload_bytes()` は指定範囲の格納バイトを読み出し、`clone()` と Owned 出力は
  バイトとメタデータを非タイル化せずに保持します。`SimaArgMax` など配置を理解する
  同じグラフ内の後続ノードには、プロデューサーの検証済み配置が直接渡されます。
  密配置専用のコンシューマーや、未対応のタイル化グラフ入力は拒否されます。全チャネルの INT8 セルタイルは密な NHWC と
  バイト単位で同一なので、通常の密コピーと DLPack を引き続き利用できます。
  既存の形式名だけの `TessSpec` 注釈は動作を変えません。非ゼロの `storage_bytes` が
  明示的なタイルストレージを表します。
- **SimaBEVPoolPlan** は `SimaBEVPool` の入力の型である `input_dtype` を受け取ります。
  `V2` のプランは EV74 のキャッシュをバイト単位で割り当てるため、BF16 のプランは
  大きくなることがあります。`V1` のプランはどちらの型でも同じです。
- **SimaBEVPool** は、ゼロポイントが 0 の BF16 `depth` と `context` を受け付けます。各
  セルの和はプランの点の順序で binary32 として計算されるため、`V1` と `V2` の
  プランでは BF16 のバイトが異なることがあります。BF16 の `bev` は各和を最近接偶数丸めで
  格納し、NaN は `0x7FC0` とし、既定の `quantization` を必要とします。BF16 入力からの
  INT8 `bev` は和を量子化します。`quantization.multiplier` には `1 / output_scale` を
  設定します。INT8 入力から BF16 の `bev` は生成できません。

## ノードをパイプラインに接続する

これらのノードは通常のグラフステージですが、入力は複数の上流ステージから来ます。

- PointPillars では、`SimaPillarScatter` はピラー特徴量モデルのピラー特徴量と、
  `SimaVoxelize` の `coordinates` と `count` を必要とします。
- BEVFusion では、`SimaBEVPool` はカメラモデルの `depth` と `context` と、`SimaBEVPoolPlan` の
  プランを必要とします。

これらのテンソルを、上記の順序で 1 つの入力サンプルにまとめてからノードに渡します。
たとえば、アプリケーションで両方の結果を取り出し、それらをまとめてノードのグラフに
プッシュします。

`SimaPillarScatter` は特徴量のバイトを再量子化せずにコピーするため、特徴量はキャンバスを
読むモデルの入力量子化をすでに使用している必要があります。`SimaBEVPool` はすべての
カメラを 1 つの `[N,...]` テンソルで受け取ります。

## プランと状態

プランはキャリブレーションと `SimaBEVPoolPlanOptions` だけに依存します。
`cache_plan = true` の場合、キャリブレーションのバイトが前のフレームと等しいフレームは、
プランを再計算せずに前のフレームのプランをコピーします。このキャッシュは、処理中の
フレームごとにランタイムが所有するカーネルライブラリのコンテキストにあるため、複数の
フレームを同時に処理しているときは、ヒットする可能性はあっても保証はされません。
どちらの場合もプランのバイトは同じです。

`SimaBEVPool` はフレーム間で状態を保持しません。実行のたびにプランのヘッダーと内容を
検査し、自身の形状と一致しないプランを拒否します。

## ターゲットの制限

EV74 では、A65 の契約に次の制限が加わります。

- `SimaBEVPool` は、チャネル数が 16 の倍数のチャネルラスト `context` と、チャネルラストの
  `bev` を必要とします。`V2` のプランでは、チャネルラストの `depth` も必要です。
- `SimaBEVPoolPlan` は A65 上でのみ実行されます。

あるターゲットがサポートしないリクエストは、そのターゲットで失敗します。ノードが
代わりに別のターゲットを選ぶことはありません。
