---
title: "CVUカーネルとグラフのカタログ"
description: "フレームワークのCVU側のカーネルが何を行い、それらからグラフの前処理／後処理がどのように構成されるのか。"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/cvu_kernels
---

# CVUカーネルとグラフのカタログ

このフレームワークには、CVU（EV74）カーネルの小さなカタログが含まれており、これらは、プランナーが各モデルに対して選択する前処理および後処理グラフに組み込まれます。このページでは、カーネルと、それらを構成するグラフファミリーについて説明します。

## カーネルファミリー

### 前処理カーネル

入力とMLAの間で実行されます。標準ファミリー：

- **Resize** — バイリニア/ニアレストスケーリング。オプションでレターボックスまたはセンタークロップを使用。
- **色変換** — RGB ↔ BGR、NV12 → RGB、I420 → RGB、GRAY8 ↔ パック。
- **Layout convert** — HWC ↔ CHW、軸の並べ替え。
- **Normalize** — チャンネルごとの平均/標準偏差（FP32入力、FP32出力）。

### 境界カーネル

MLA境界を越えてFP32 / BF16 / INT8をブリッジします。

- **Quant** — FP32 → INT8（スケール+ゼロポイント付き）。
- **Dequant** — INT8 → FP32（スケール+ゼロポイント付き）。
- **Cast** — FP32 ↔ BF16（スケール/ゼロポイントなし）。
- **Tess** / **Detess** — MLAタイルジオメトリへの/からのレイアウトシャッフル。バイト数は同じで、順序が異なります。

### 融合カーネル

モデルのコントラクトが境界カーネルを要求するが、MLAステージにそれらを含まない場合に、プランナーが選択する組み合わせ：

- **QuantTess** — Quant + Tessを融合。
- **DetessDequant** — Detess + Dequantを融合。
- **CastTess** / **DetessCast** — BF16パスでCastとTessを融合。

### ジェネリック前処理

アプリケーションが任意のユーザー定義変換を供給する場合、プランナーは前処理グラフをジェネリックバリアントにアップグレードし、それらの変換を単一のCVUカーネルに融合させます。MLA境界でのコントラクトは変更されません。

### BoxDecode

検出モデルのNMS / デコードを融合する後処理カーネル。出力サンプルに`DetectionMeta`を生成します。[`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h)を参照してください。

## グラフの構成方法

4つの`PreprocessGraphFamily`の値は、4つのカーネルチェーンに対応します。

| グラフファミリー | チェーン（入力→MLA） |
|--------------|---------------------|
| `Preproc` | Resize → ColorConvert → Normalize → MLA（内部でテッセレーション） |
| `Quant` | Resize → ColorConvert → Normalize → Quant → MLA（内部でテッセレーション） |
| `Tess` | Resize → ColorConvert → Normalize → Tess → MLA |
| `QuantTess` | Resize → ColorConvert → Normalize → QuantTess → MLA |

出力側の対応する処理 — `Postproc` / `Detess` / `DetessDequant` / パススルー — は、MLAのコンパイルされた出力カーネルにdetess/dequantが含まれているかどうかに依存します。

これらの4つのファミリーが存在する理由については、[dtype の契約](/develop-apps/advanced-concepts/dtype_contract)を参照してください。

## カーネルの命名規則

フレームワーク内では、カーネルは安定した文字列名で参照され、これらの名前は`RoutePlanner`の決定と`MeasureReport`プラグイン/カーネルのタイミング行に表示されます。

- `cvu/preproc/<variant>` — 前処理カーネル。
- `cvu/quant/<dtype>` — 量子化バリアント。
- `cvu/tess/<geometry>` — テッセレーションバリアント。
- `cvu/postproc/box_decode_<type>` — BoxDecodeバリアント。

正確なカタログは、`core/src/pipeline/internal/sima/`（フレームワークのreach-throughレイヤー）にあります。

## 詳細

このドキュメントでは、CVUカーネルとグラフのカタログについて説明しました。

- 「CVUカーネルとグラフのカタログ」— デザインの詳細な解説の第86節、第87節。
- 「テッセレーション、量子化、キャスト」— デザインの詳細な解説の第17節。
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — 4つのコーナーを持つ列挙型。
- [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h) — 後処理によるボックスのデコード。
