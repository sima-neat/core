---
title: "データ形式"
description: "フォーマットタグ、ペイロードファミリー、レイアウト、およびテンソルの意味論"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/data_formats
---

# データ形式とテンソルの意味

このページでは、`InputOptions::format`、`OutputTensorOptions::format`、テンソル画像メタデータ、およびサンプルペイロードタグで使用される公開形式の語彙について説明します。

タスクレベルでの使用については、[テンソルとサンプル](/develop-apps/development-workflow/core_types)から開始します。グラフの境界に明示的な形式契約が必要な場合は、ここを参照してください。

## 形式タグ

`FormatTag` / `FormatSpec`は、ペイロードの形式を指定します。Pythonでは、形式フィールドに`pyneat.Format`または`pyneat.FormatTag`の値を使用します。Pythonの形式フィールドに生の文字列を割り当てないでください。

Pythonは、一般的なユーザー向けの形式タグを公開します。`BBOX`、`MLA`、`ARGMAX`、および`DETESSDEQUANT`などの、より低レベルのC++タグは、通常、テンソルの意味メタデータ、ペイロードタグ、または診断を通じて表示され、割り当て可能な`pyneat.Format`値としては表示されません。

一般的なタグ：

| タグ | 一般的なペイロード | 意味 |
|---|---|---|
| `RGB` | 画像 | パックされたRGB、各チャネルあたり8ビット。 |
| `BGR` | 画像 | パックされたBGR、各チャネルあたり8ビット。OpenCVはデフォルトでこれを使用します。 |
| `GRAY8` | 画像 | 8ビットグレースケール。 |
| `NV12` | 画像/ビデオ | Y平面とインターリーブされたUV平面。幅と高さは偶数である必要があります。 |
| `I420` | 画像/ビデオ | Y、U、およびV平面。幅と高さは偶数である必要があります。 |
| `H264` | エンコード済み | H.264アクセスユニット/NALストリーム。`AVC`はエイリアスです。 |
| `H265` | エンコード済み | H.265 / HEVCアクセスユニット/NALストリーム。`HEVC`はエイリアスです。 |
| `ENCODED` | エンコード済み | ジェネリックなエンコード済みペイロード。caps文字列は、専用の形式タグがないコーデックを識別します。 |
| `FP32` | テンソル | Float32テンソルペイロード。 |
| `INT8` | テンソル | 符号付きINT8テンソルペイロード。 |
| `UINT8` | テンソル | 符号なしUINT8テンソルペイロード。 |
| `BF16` | テンソル | BF16テンソルペイロード。 |
| `BBOX` | 検出 | パックされたバウンディングボックスペイロード。 |
| `ByteStream` | テンソルの意味 | 下流の契約によって解釈される不透明なバイトストリーム。 |

## ペイロードファミリー

`PayloadType`は、グラフの境界を越える広範なファミリーを選択します。

| ペイロードファミリー | 内部/メディアの意味 | 一般的なメタデータ |
|---|---|---|
| `Image` | デコードされたピクセル | ピクセル形式、幅、高さ、レイアウト、画像セマンティックメタデータ |
| `Tensor` | モデルまたはアプリのテンソル | dtype、形状、レイアウト、テンソルセマンティックメタデータ |
| `Encoded` | H.264、H.265、またはJPEGなどのエンコードされたメディア | caps文字列、コーデック形式、タイムスタンプ |
| `Auto` | 可能な場合は推論 | テンソル/サンプルメタデータだけで十分な場合にのみ使用 |

テキスト、オーディオ、バイトストリーム、および不透明なバイトペイロードは、テンソルの意味または特殊な仕様を使用します。これらは、このリリースでレビューされた公開APIの`PayloadType`列挙型の個別の値ではありません。

## 生の画像マッピング

| フォーマット | ペイロードタイプ | テンソルレイアウト / シェイプ | 備考 |
| --- | --- | --- | --- |
| `RGB` | `Image` | `HWC`, `[H, W, 3]` | 密にパックされたピクセル。 |
| `BGR` | `Image` | `HWC`, `[H, W, 3]` | `cv2.imread` または OpenCV BGR フレームで使用。 |
| `GRAY8` | `Image` | `HW`, `[H, W]` | 単一チャンネルのグレースケール。 |
| `NV12` | `Image` | `HW`, `[H, W]` + プレーンメタデータ | Y + UV プレーンの複合。 |
| `I420` | `Image` | `HW`, `[H, W]` + プレーンメタデータ | Y + U + V プレーンの複合。 |

パックされたフォーマットの場合、奥行きはチャンネル数です。テンソルペイロードの場合、奥行きは選択されたレイアウトとシェイプから導き出されます。

## フォーマット、レイアウト、軸セマンティクスを組み合わせて読み取る

単一のフィールドを単独で読み取らないでください。

| フィールド | 何を教えてくれるか |
| --- | --- |
| `PixelFormat` / 画像フォーマットメタデータ | RGB、BGR、GRAY8、NV12、または I420 などのピクセルチャンネルをどのように解釈するか。 |
| `TensorLayout` | テンソルの次元がどのように順序付けされているか（例：HWC、CHW、HW）。 |
| `TensorAxisSemantic` | テンソルがより豊富なセマンティックメタデータを持つ場合に、軸が何を意味するか。 |
| `TensorDType` | 各要素がどのように格納されているか（例：UInt8、INT8、FP32、BF16）。 |
| `ByteFormat` / バイトストリームメタデータ | 次の段階で不透明なバイトをどのように解釈するか。 |

バイトは意味を持ちません。バッファーを再解釈する前に、メタデータフィールドを組み合わせて使用してください。

## InputOptions フォーマットの例

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::InputOptions input;
input.payload_type = simaai::neat::PayloadType::Image;
input.format = simaai::neat::FormatTag::BGR;
input.width = 640;
input.height = 480;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
input_options = pyneat.InputOptions()
input_options.payload_type = pyneat.PayloadType.Image
input_options.format = pyneat.Format.BGR
input_options.width = 640
input_options.height = 480
```

</CodeTab>
</CodeTabs>

境界に必要なフィールドのみを設定します。テンソルまたはサンプルにすでに十分なメタデータが含まれている場合は、重複した推測を避けてください。

H.264とH.265には、それぞれ専用の`H264`と`H265`タグがあります。入力境界に一致するタグを設定します。メディアタイプはそこから解決されます。専用のタグを持たないコーデックの場合のみ、`ENCODED`を明示的なcaps文字列とともに使用します。

## 高度な画像/ビデオ出力アダプター

通常のモデル出力には、`nodes.output(...)`を使用し、`pull_tensors(...)`でテンソルを取得します。画像またはビデオ出力をCPUで扱いやすい`UInt8`テンソルに変換、サイズ変更、またはレート調整する必要がある場合にのみ、`OutputTensorOptions`を使用します。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::OutputTensorOptions output;
output.format = simaai::neat::FormatTag::BGR;
output.target_width = 640;
output.target_height = 480;

graph.add_output_tensor(output);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
output = pyneat.OutputTensorOptions()
output.format = pyneat.Format.BGR
output.target_width = 640
output.target_height = 480

graph.add_output_tensor(output)
```

</CodeTab>
</CodeTabs>

`add_output_tensor(...)` は、デフォルトの出力データ型である `TensorDType::UInt8` を受け入れます。モデルのテンソルや、完全な `Sample` エンベロープが必要な出力に対しては、通常の `nodes.output(...)` パスを維持してください。別のデータ型が必要な場合は、明示的なグラフまたはアプリケーション側の変換を追加します。

## サンプルペイロードタグ

`Sample::payload_tag` は、下流のコンシューマーにとって推奨されるラベルです。これは、非推奨の `Sample::format` フィールドに取って代わります。

エンコードされたメディアまたはグラフ境界のネゴシエーションをデバッグする際には、`payload_tag`、`payload_type`、`media_type`、および `caps_string` を組み合わせて使用します。

## 前処理メタデータと ROI ブレッドクラム

検出、デコード、レンダリング、および ROI ワークフローでは、モデル空間の座標をソースフレームの座標にマッピングするために、前処理メタデータが必要です。

そのメタデータには、以下を含めることができます。

- ターゲットの幅と高さ。
- スケールされたコンテンツの幅と高さ。
- リサイズまたはレターボックスモード。
- パディング値とジオメトリ。
- 入力と出力のカラーフォーマット。
- 軸の並べ替え。
- 正規化、量子化、およびテッセレーションフラグ。
- ROI ウィンドウ、ソース画像のサイズ、ROI バッチサイズ、および ROI ごとのアフィン変換。

ボックスまたはマスクが間違った場所に配置されている場合は、しきい値の変更前に、前処理メタデータがデコードまたはレンダリング段階に到達したかどうかを確認してください。ROI リストの前処理の詳細については、[ROIリストを前処理する](/reference/preproc_roi) を参照してください。

## 関連項目

- [テンソルとサンプル](/develop-apps/development-workflow/core_types)
- [dtype の規約](/develop-apps/advanced-concepts/dtype_contract)
- [ノード](/develop-apps/development-workflow/node)
