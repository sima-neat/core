---
title: "LiDAR と BEV のカーネルノード"
description: "SimaVoxelize、SimaPillarScatter、SimaBEVPoolPlan、SimaBEVPool、SimaCenterHeadDecode のカーネルライブラリ演算をグラフノードとして実行する"
sidebar_position: 8
---

# LiDAR と BEV のカーネルノード

`SimaVoxelize`、`SimaPillarScatter`、`SimaBEVPoolPlan`、`SimaBEVPool` は、PointPillars と BEVFusion の
MLA 以外のステージを実行します。`SimaCenterHeadDecode` は、CenterPoint 系モデルのヘッド出力を 3D ボックスに
変換します。各ノードは、選択したターゲット上で SiMa カーネル
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
| `SimaCenterHeadDecode` | CenterHead 出力から 3D ボックスへ | A65 |

以下は、アプリケーション内に記述するグラフ構築のコード断片です。
`read_sweep()` はアプリケーション側で実装し、x、y、z、時刻オフセットを交互に
並べた、空でない `4*N` 個の float 値を返してください。`N` は INT32 で表現できる
必要があります。この関数は Neat API ではありません。この断片はグラフを構築する
だけで、フレームの送信や出力の取得は行いません。

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
`pyneat.nodes.sima_bev_pool`、`pyneat.nodes.sima_centerhead_decode` のファクトリーを使用できます。

## 入力と出力の契約

各ノードは入力サンプルのテンソルを下表の順序で読み取り、出力を下表の順序で公開
します。テンソルの数、データ型、形状はグラフのビルド時に検査されます。

| ノード | 入力 | 出力 |
| --- | --- | --- |
| `SimaVoxelize` | FP32 `points` `[N,4]`（x、y、z、時刻オフセット）、INT32 `point_count` `[1]` | INT8 または BF16 の `decorated` `[1,M,T,10]`、(row, column) 形式の INT32 `coordinates` `[M,2]`、INT32 `count` `[1]` |
| `SimaPillarScatter` | INT8 または BF16 の `features` `[M,C]`、INT32 `coordinates` `[M,2]`、INT32 `count` `[1]` | `features` と同じ型の `canvas` `[1,H,W,C]` |
| `SimaBEVPoolPlan` | FP32 `camera2lidar`、`camera_intrinsics`、`img_aug_matrix` `[N,4,4]`、FP32 `lidar_aug_matrix` `[4,4]` | UINT8 `plan` `[P]` |
| `SimaBEVPool` | INT8 または BF16 の `depth`、同じ型の `context`、UINT8 `plan` `[P]` | INT8 または BF16 の `bev` `[1,nZ*C,nX,nY]` |
| `SimaCenterHeadDecode` | 2～6 個のポート上の FP32 または BF16 ヘッドマップ | FP32 `boxes` `[K,9]` または `[K,7]`、FP32 `scores` `[K]`、INT32 `labels` `[K]`、INT32 `count` `[1]` |

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
- **SimaCenterHeadDecode** は、`heatmap_input` から `vel_input` までの値でヘッドを配置します。デフォルトでは
  3 つのポートが、ヒートマップ、reg・height・dim・vel、rot をそれぞれ保持します。`K` は
  `tasks.size() * post_max_per_task` です。`boxes` は底面中心の x、y、z、dx、dy、dz、yaw を保持し、
  `velocity` を使うと vx、vy も保持します。`labels` はヘッドタスク全体で数えたヒートマップチャネルです。
  `count` 以降の行はゼロです。

### 論理軸と物理ストレージ

`SimaBEVPool` では、論理軸の順序と物理的なバイト順序を区別します。
`TensorLayout::HWC` のテンソルは論理軸 `[N,H,W,K]`、`TensorLayout::CHW` のテンソルは
論理軸 `[N,K,H,W]` を持ちます。ノードはそのタグを優先し、ストライドが
`input_storage` に一致するか検査します。`ChannelsLast` では HWC 入力は密で、CHW
入力はチャネルラストのストライドを持つ必要があります。`Contiguous` では密な
CHW 入力が必要で、HWC タグは拒否されます。
同じ標準的な軸順序は `Tensor.axis_semantics` でも指定できます。両方の注釈を
設定する場合は、一致している必要があります。

タグのない `ChannelsLast` 入力では、互換性のため、密な入力を NHWC、非密な入力を
チャネルラストのストライドを持つ論理 NCHW として扱います。大きさ 1 の軸によって
論理 NCHW ビューも密になる場合は、`Tensor.layout = TensorLayout::CHW` を明示して
ください。この場合、ストライドだけでは論理軸を判別できません。Pool の出力は
CHW の論理軸と `output_storage` に対応するストライドを持ちます。レイアウトを
変えるためにノードがペイロードを転置またはコピーすることはありません。

### 量子化メタデータ

INT8 の出力は、ノードが量子化を把握している場合、それを `Tensor.semantic.quant` に
保持します。`SimaVoxelize` と BF16 入力の `SimaBEVPool` では
`scale = 1 / quantization.multiplier`、INT8 入力の両方がスケールを持つ `SimaBEVPool` では
`depth_scale * context_scale / quantization.multiplier`、`SimaPillarScatter` では特徴量の
量子化です。それ以外の場合、量子化は公開されません。

INT8 の Scatter 入力では、`[1,M,1,C]` のような形状を含め、特徴量の最後の軸 `C`
だけがチャネル単位の量子化に対応します。有限かつ正のスケールを `C` 個と、空の
セルを埋める Scatter の `zero_point` にすべて等しい INT8 ゼロポイントを `C` 個
指定してください。キャンバスはこれらの値を保持し、量子化軸を `[1,H,W,C]` の
`3` に移します。それ以外の量子化軸は拒否されます。座標に従ってピラーを配置した
後のキャンバスは、ピラー軸のパラメーターでは固定的に記述できません。

`SimaBEVPool` はチャネル単位の量子化メタデータに対応せず、黙って削除する代わりに
拒否します。既存のテンソル単位のメタデータと明示的なゼロポイントオプションの
動作は変わりません。入力メタデータがない場合は、不明のままです。

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

追加された `TessSpec::axes` と `TessSpec::storage_bytes` フィールドにはデフォルト値があるため、既存の 2 フィールドの集成体初期化子はソース互換性を維持します。ただし、公開構造体のバイナリレイアウトは変わります。一致する Core ヘッダーとライブラリを使って C++ コンシューマーと Python バインディングを再ビルドし、以前のレイアウトでビルドしたバイナリは再利用しないでください。

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

## CenterHead 出力をデコードする

`SimaCenterHeadDecode` は、CenterPoint や BEVDet などのモデル向けに、mmdetection3d の CenterPoint ボックス
コーダーとヘッドタスクごとの NMS を実行します。`model.inference()` の直後では MLA の BF16 ヘッドマップを
MLA が書き込んだ場所で読むため、モデルのポストステージは実行されません。

以下は、アプリケーション内に記述するグラフ構築のコード断片です。値は CenterPoint の nuScenes 構成です。

```cpp
#include <neat.h>

using namespace simaai::neat;

Model model("centerpoint_mpk.tar.gz");

const auto circle = [](int classes, float radius) {
  SimaCenterHeadTask task;
  task.classes = classes;
  task.nms = SimaCenterHeadNms::Circle;
  task.nms_radius = radius;
  return task;
};
SimaCenterHeadDecodeOptions decode;
decode.tasks = {circle(1, 4.0f), circle(2, 12.0f), circle(2, 10.0f),
                circle(1, 1.0f), circle(2, 0.85f), circle(2, 0.175f)};
decode.out_size_factor = 4;
decode.voxel_size = {0.2f, 0.2f};
decode.range_min = {-51.2f, -51.2f};
decode.score_threshold = 0.1f;
decode.top_k = 500;
decode.post_max_per_task = 83;
decode.post_center_range = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};

Graph graph;
graph.add(nodes::Input());
graph.add(model.inference());
graph.add(nodes::SimaCenterHeadDecode(decode));
graph.add(nodes::Output());
```

pull した各 Sample は、`boxes`、`scores`、`labels`、`count` をこの順に保持します。先頭の `count` 行を読み取って
ください。`model.inference()` は MLA から始まるため、グラフの入力はすでに MLA が読む形式である必要があります。
ノードの並列処理は、グラフの A65 ワーカー（`GraphOptions.kernel_runner_a65_workers`）を共有します。

### ヘッドマップ

ポート `i` は `i` 番目の上流テンソルです。どのポートも FP32 または BF16 のチャネルラストのマップ `[H,W,C]` または
`[1,H,W,C]` で、すべてのポートで H と W が同じです。

- ヒートマップは専用のポートを持ち、クラスごとに 1 チャネルをヘッドタスクの順に保持します。
- それ以外のポートは、1 つ以上のボックスヘッドをヘッドタスクの順に保持します。各ヘッドタスクのヘッドは、
  reg（2 チャネル：x と y のオフセット）、height（1）、dim（3：dx、dy、dz の対数）、rot（2：サインとコサイン）、
  `velocity` を使う場合は vel（2）の順に並びます。
- デフォルトは、出力が 3 つのモデルに合います。ヒートマップ、reg・height・dim・vel、rot です。ヘッドタスクが 6 つで
  クラスが 10 個の nuScenes 向け CenterPoint では、チャネル数は 10、48、12 です。ヘッドごとに出力が 1 つのモデル
  では、`heatmap_input` から `vel_input` までをその出力の順に設定します。

ヘッドタスクとヘッドの組ごとに出力が 1 つあるモデルは、出力が 6 つを超えます。モデルをエクスポートするときに、
ヘッドごとに連結してください。

マップは密なもの、またはセルと行の後に隙間のある BF16 です。`model.inference()` の後、SDK 3.0 のパッケージの MLA は
ヘッドをこの形式（セルごとに 16 チャネルにパディングされた BF16）で公開し、ノードはそれをその場で読み取ります。
ボックスはヘッドの格納方法に依存しません。

MLA の出力が量子化されている、またはタイル化されているパッケージは、`model.inference()` の後に整数を公開し、
グラフのビルドはノードが示す理由とともに失敗します。その場合は `model.inference()` の代わりに `model.graph()` を
使ってください。これにより MLA の前後にモデル自身の変換が追加されるため、グラフはモデルの入力を受け取り、
ノードは FP32 のヘッドマップを受け取ります。

### オプション

オプションは、モデルの mmdetection3d 設定にあるデコードの設定です。

| オプション | mmdetection3d の設定 |
| --- | --- |
| `tasks[i].classes` | ヘッドタスク `i` の `class_names` の数 |
| `tasks[i].nms`, `tasks[i].nms_radius` | `test_cfg.nms_type = 'circle'`, `test_cfg.min_radius[i]` |
| `tasks[i].nms`, `tasks[i].nms_iou_threshold` | `test_cfg.nms_type = 'rotate'`, `test_cfg.nms_thr` |
| `nms_size_scale` | `test_cfg.nms_rescale_factor`。クラスごとに 1 つの値 |
| `out_size_factor`, `voxel_size` | `bbox_coder` の同名のキー |
| `range_min` | `bbox_coder.pc_range` の x と y の最小値 |
| `score_threshold`, `top_k`, `post_center_range` | `bbox_coder` の `score_threshold`、`max_num`、`post_center_range` |
| `post_max_per_task` | `test_cfg.post_max_size` |
| `velocity` | ヘッドに `vel` ブランチがある |

どのヘッドタスクも NMS を明示します。circle NMS は、mmdetection3d と同じく、2 つのボックス中心間の距離の 2 乗を
`nms_radius` の値そのものと比較します。rotate NMS は、回転した 2 つのボックスの鳥瞰図での IoU を
`nms_iou_threshold` と比較します。`nms_size_scale` を設定する場合は、ヒートマップチャネルごとに値を 1 つ指定し、
circle のヘッドタスクのクラスには 1 を指定します。`labels` はヒートマップチャネルなので、2 番目のヘッドタスクの
最初のクラスは、1 番目のヘッドタスクの最後のクラスのすぐ次になります。

Python では、ヘッドタスクの値をコンストラクターに渡し、リストは丸ごと代入します。オプションオブジェクトから
読み取ったリストはコピーなので、`decode.tasks` に append しても何も変わりません。

```python
import pyneat

circle = pyneat.SimaCenterHeadNms.Circle
decode = pyneat.SimaCenterHeadDecodeOptions()
decode.tasks = [
    pyneat.SimaCenterHeadTask(classes, circle, nms_radius=radius)
    for classes, radius in [(1, 4.0), (2, 12.0), (2, 10.0), (1, 1.0), (2, 0.85), (2, 0.175)]
]
decode.out_size_factor = 4
decode.voxel_size = [0.2, 0.2]
decode.range_min = [-51.2, -51.2]
decode.score_threshold = 0.1
decode.top_k = 500
decode.post_max_per_task = 83
decode.post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
node = pyneat.nodes.sima_centerhead_decode(decode)
```

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
- `SimaBEVPoolPlan` と `SimaCenterHeadDecode` は A65 上でのみ実行されます。

あるターゲットがサポートしないリクエストは、そのターゲットで失敗します。ノードが
代わりに別のターゲットを選ぶことはありません。
