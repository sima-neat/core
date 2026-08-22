---
title: "dtype の規約"
description: "テンソルのデータ型、量子化、テッセレーション、およびパブリックペイロード契約がどのように連携するか"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/dtype_contract
---

# dtype契約

Neatモデルのルートには、次の2つの契約があります。

- **パブリック契約**: アプリケーションが`Tensor`、`Sample`、`InputOptions`、モデル仕様、およびグラフエンドポイントを通じて見るもの。
- **モデルルート契約**: Neatが、コンパイルされたモデルアーカイブと選択された前処理/後処理パスから解決するもの。

すべてのパブリック境界がFP32であると想定しないでください。一部の境界は、画像、エンコードされたメディア、パックされた検出ペイロード、INT8テンソル、BF16テンソル、またはアプリケーション定義のテンソルセマンティクスを伝達します。まず仕様を確認してください。仕様が契約です。

ルート内では、Neatは、コンパイルされたモデルの契約で必要な場合に、量子化、テッセレーション、キャスト、非テッセレーション、非量子化、および後処理ステージを挿入します。

## 4つのMLA入力ケース

モデルアーカイブは、最初のMLAステージについて、Neatに次の2つの重要なことを伝えます。

- MLA入力のdtype。通常は**BF16**または**INT8**。
- MLA側のテッセレーションが、すでにコンパイルされたカーネルの一部であるかどうか。

これにより、次の4つの前処理グラフファミリーが得られます。

| MLA dtype | MLA tess | 前処理グラフファミリー | NeatがMLAの前に挿入するもの |
|---|---|---|---|
| BF16 | はい | `Preproc` | リサイズ、カラー変換、正規化。MLAステージは内部でテッセレーションを実行します。 |
| BF16 | いいえ | `Tess` | リサイズ、カラー変換、正規化、テッセレーション。 |
| INT8 | はい | `Quant` | リサイズ、カラー変換、正規化、量子化。MLAステージは内部でテッセレーションを実行します。 |
| INT8 | いいえ | `QuantTess` | リサイズ、カラー変換、正規化、量子化、テッセレーション。 |

[`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan)を確認して、プランナーが選択した内容を確認してください。

## テッセレーションの意味

テッセレーションは、テンソルバイトを、MLA入力スクラッチパッドが期待するタイルジオメトリに配置します。これはレイアウト変換です。同じ論理テンソルですが、メモリの順序が異なります。

対応する非テッセレーションは、ルートが次のステージまたはアプリケーションに自然なテンソルレイアウトを返す必要がある場合に、MLA出力の後に発生します。

## 境界のアップグレード

Neatは、4つのケースのdtype決定の上に、より高レベルのルートステージを追加できます。

- **汎用前処理**: `PreprocessOptions`を使用して、リサイズ、カラー、レイアウト、正規化、量子化、テッセレーション、または明示的な変換意図を推論前に適用します。
- **BoxDecode**: 検出後処理ステージを必要とするモデルの検出ヘッドをデコードします。アプリケーションは、`BoxDecodeType`（例：`YoloV8`）などのファミリーと、`score_threshold`、`nms_iou_threshold`、および`top_k`などのフィルタリングフィールドを選択します。

これらのアップグレードにより、実行されるカーネルと、アプリケーションが受信する出力契約が変更されます。たとえば、生のモデル出力テンソルとデコードされた検出テンソルは、同じパブリック契約ではありません。

## アプリケーションコードへの影響

- 入力または出力のデコードを行う前に、`model.input_specs()`と`model.output_specs()`を確認してください。
- `ModelOptions.preprocess`を使用して、どのような種類の入力を提供するか（画像入力、テンソル入力、リサイズ、色、レイアウト、正規化、量子化、またはテッセレーションの意図など）を指定します。
- `model.resolved_preprocess_plan()`/`model.preprocess_plan()`を使用して、Neatが、指定したオプションとモデルアーカイブに基づいてどのような処理を計画したかを確認します。
- 出力dtype、形状、またはレイアウトを推測しないでください。出力仕様を読み、必要に応じて、返されたテンソルのメタデータを読み取ります。
- 出力コントラクトが一致するパック形式の場合にのみ、ボックス、ポーズ、またはセグメンテーションをデコードします。
- INT8/BF16/テッセレーションの詳細については、公開された仕様またはテンソルで明示的に公開されていない限り、ランタイムの動作として扱います。

余計なことはせず、コントラクトを読み、次にバイトを移動します。

## 出力を意図的にデコードする

出力コントラクトに一致するデコードヘルパーを使用します。

| 出力コントラクト | C++ | Python |
|---|---|---|
| 生のテンソル | 返された`Tensor`/`TensorList`を直接使用します | 返されたテンソルを直接使用するか、`to_numpy(...)`/`to_torch(...)`を使用します |
| パックされたボックス | `simaai::neat::decode_bbox(...)` | `pyneat.decode_bbox(...)` |
| パックされたポーズ | `simaai::neat::decode_pose(...)` | `pyneat.decode_pose(...)` |
| パックされたセグメンテーション | `simaai::neat::decode_segmentation(...)` | `pyneat.decode_segmentation(...)` |

デコードされたボックスは、`[N, 6]`のテンソル（float32）を使用し、列は`x1`、`y1`、`x2`、`y2`、`score`、および`class_id`です。ポーズとセグメンテーションのデコーダーは、ボックスに加えて、キーポイントまたはマスク用のタスク固有のテンソルを返します。

## 座標メタデータを保持する

検出座標は、多くの場合、モデル空間からソースフレーム空間にマッピングするために、前処理メタデータが必要です。レターボックス、リサイズ、ROIリスト、レンダリング、または検出デコードを使用する場合は、グラフを通じてメタデータを保持します。

関連するメタデータには、ターゲットサイズ、スケーリングされたサイズ、パディング、カラー変換、軸の順序変更、正規化、量子化、テッセレーション、ROIウィンドウ、およびROIごとのアフィン変換が含まれる場合があります。

デコードされたボックスが間違った場所に配置されている場合は、NMSを非難する前に、メタデータの伝播を確認してください。[データ形式](/develop-apps/advanced-concepts/data_formats#preprocess-metadata-and-roi-breadcrumbs)と[ROIリストを前処理する](/reference/preproc_roi)を参照してください。

## 関連する型

- [`PreprocessOptions`](/reference/cppapi/structs/simaai-neat-preprocessoptions) — アプリケーションの前処理の意図。
- [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) — プランナーがコンパイルしたもの。
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — 選択された前処理ファミリー。
- [`Tensor`](/reference/{lsa}/structs/simaai-neat-tensor) — 公開テンソルのペイロードとメタデータ。
- [`Sample`](/reference/{lsa}/structs/simaai-neat-sample) — ペイロードとランタイムメタデータ。

## 詳細

- [テンソルとサンプル](/develop-apps/development-workflow/core_types)
- [データ形式](/develop-apps/advanced-concepts/data_formats)
