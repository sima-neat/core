---
title: "フレームごとの属性"
description: "選択した複数の HTTP ヘッダーをキャプチャし、それらが含まれるデコードされたフレームから読み出して取得する。"
sidebar_position: 4
slug: /develop-apps/advanced-concepts/frame_attributes
---

# フレームごとの属性

一部のカメラは、フレームを伝送する際に、そのフレームに関する情報を記述します。具体的には、シーケンス番号、キャプチャ時のタイムスタンプ、チャンネル名などです。Neat は、これらの値の中から選択されたものを、デコード処理を通じて保持し、それらが属するフレームの `Sample` に渡します。

`Sample::attributes` は、単純な文字列から文字列へのマッピングです。Neat は、これをコピーし、フレームに関連付けたまま、宛先バッファが再利用されるときにクリアします。値の解析、マージ、または再解釈は行いません。キーと値は有効な UTF-8 であり、GStreamer の文字列表現では保持できない埋め込み NUL バイトを含んではいけません。

## 保証されること

> 選択されたマルチパートのヘッダーは、デフォルトの `HttpMjpegDecodedInput` パス、キュー/ブランチ、および Core Sample から GStreamer へのマテリアライズされた境界を通じて、デコードされたフレームに関連付けられたままになります。

これが、今回のリリースにおけるすべての保証です。それ以外の事項は、[サポートされていないパス](#unsupported-paths) にリストされており、Neat は、属性を静かに破棄するのではなく、グラフの構築に失敗します。

## キャプチャの有効化

キャプチャはデフォルトで無効になっています。有効にしたいヘッダーの名前を指定することで、有効になります。

```cpp
#include "nodes/groups/HttpMjpegDecodedInput.h"

simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
opt.url = "http://camera.local/stream";
opt.header_capture.headers = {"Image-Index", "Image-Time"};

auto source = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
```

もう一度読み直すと：

```cpp
simaai::neat::Sample sample;
if (run.pull(1000, sample) == simaai::neat::PullStatus::Ok) {
  const auto it = sample.attributes.find("image-index");
  if (it != sample.attributes.end()) {
    // it->second is the value this frame was sent with.
  }
}
```

Pythonでは、同じ表面が使用され、`attributes`は動的なマッピングであり、アイテムの割り当ては、
基盤となる`Sample`に到達し、辞書を割り当てることで内容が置き換えられます。

```python
import pyneat

opt = pyneat.HttpMjpegDecodedInputOptions()
opt.url = "http://camera.local/stream"
opt.header_capture.headers = ["Image-Index", "Image-Time"]
source = pyneat.groups.http_mjpeg_decoded_input(opt)

# ... later, on a pulled sample:
index = sample.attributes.get("image-index")

sample.attributes["image-index"] = "42"     # reaches the Sample
sample.attributes = {"image-time": "..."}   # replaces the whole map
```

## ヘッダーのルール

設定されたリストは**許可リスト**です。空のリストは、キャプチャを完全に無効にし、既存のトポロジーと動作を変更しません。

| ルール | 動作 |
|---|---|
| 大文字と小文字 | 設定された名前と出力されるキーは、ASCII小文字に正規化されます。大文字と小文字を区別しないマッチングが行われます。属性は小文字のキーで読み戻されます。 |
| 許可リスト内の重複 | 正規化後に結合されます。 |
| 1つのパート内で繰り返されるヘッダー | 最後の値が優先されます。 |
| パートにヘッダーが存在しない場合 | キーは省略されます。前のフレームから継承されることはありません。 |
| ヘッダーが存在するが空の場合 | 空の文字列として保持されます。 |
| 空白 | 周囲のSP/HTABのみが削除されます。それ以外の場合、値は再解釈されません。 |
| MIMEタイプ | 存在する`Content-Type`は`image/jpeg`である必要があります（パラメータは許可されます）。存在しない場合、JPEGペイロードチェックによってパートのタイプが決定されます。 |
| JPEGペイロード | パートには、SOIからEOIまでの完全なJPEGが1つだけ含まれている必要があります。切り捨てられた、空の、または連結された画像は、ストリームでエラーになります。 |
| 無効な入力 | 無効なヘッダー名、折りたたまれたヘッダー行、およびCR/LF/NULの挿入は拒否されます。ストリームはエラーを報告し、安全な形式に正規化されることはありません。 |

「存在しない」と「空」を、空の文字列をテストするのではなく、`count()`/ `get()`を使用して区別します。

### 制限

これらのいずれかが超過すると、切り捨てではなく解析が失敗します。

| 制限 | 値 |
|---|---|
| `kMultipartHeaderCaptureMaxHeaders` | 64個の選択されたヘッダー名 |
| `kMultipartHeaderCaptureMaxNameBytes` | 1つの名前あたり128バイト |
| `kMultipartHeaderCaptureMaxLineBytes` | 1つのヘッダー行あたり8 KiB |
| `kMultipartHeaderCaptureMaxBlockBytes` | 1つのパートヘッダーブロックあたり64 KiB |
| マルチパートJPEGボディ | 1つのMIMEパートあたり64 MiB |

不正な許可リストは、`std::invalid_argument`を使用して構築時に拒否されます。

## サポートされていないパス

キャプチャが有効になっている間、`HttpMjpegDecodedInput`は、`use_videoconvert`、`use_videoscale`、`use_videorate`、または`extra_fragment`を含むグラフを構築することを拒否します。これらの要素を通じての保持は証明されておらず、静かに途中で消えてしまうメタデータよりも、明確な構築エラーの方が優れています。

属性は、いくつかの入力から新しい論理サンプルを作成するノード（モデル、結合、集約器）についても定義されていません。これらのノードは属性をマージしません。

## 関連付けが維持される仕組み

キャプチャが有効になっているグラフは、パートの境界とパートヘッダーを1つのステートマシンで解析する、プライベートなインプロセス要素を使用するため、パートのヘッダーはそのバイトを伝送するバッファーに直接添付されます。そのため、ドリフトする可能性のあるサイドチャネルはありません。この要素は、完全で解析済みのJPEGフレームを出力するため、`jpegparse`はキャプチャが有効になっているパスに挿入されません。選択された属性の添付に失敗した場合、フレームは配信されず、ストリームはエラーを報告します。

デコード処理を通じて、プラグインは受け入れられた各エンコードされた画像の属性をスナップショットとして保存し、デコーダーがそれと関連付けることで、デコードされた出力にそれらを復元します。これは、次に到着する画像に適用されるのではなく、特定の出力に適用されます。受け入れられたすべての画像は、正確に1つの最終結果に到達するため、並べ替え、画像の削除、および出力プールの再利用によって、値を別のフレームに移動させることはできません。このメカニズムは、コーデックやトランスポートに依存しないため、後で他のエンコードされたソースにも拡張でき、再設計する必要はありません。

## 互換性

`Sample`とソースオプション構造体に、追加のフィールドが追加されました。フィールド名または集約初期化を使用してこれらを使用するソースコードは、引き続きコンパイルできます。

これらのパブリック構造体のバイナリレイアウトが変更されたため、**すでにビルドされたコンシューマーは再ビルドする必要があります**。NeatのABI/SOVERSIONは**4**のままです。0.4.0はまだリリースされていないため、すべてのABI-4コンポーネントをまとめて再ビルドおよびリリースし、ABIを更新することはありません。

## 後で別のソースを追加する場合

デコーダーのパスは汎用的です。新しいエンコードされたソースは、デコーダーに渡すバッファーにネストされた属性構造をアタッチするだけで済みます。デコーダー内またはサンプル境界内には、トランスポートに固有のものは何もありません。各新しいソースが引き続き所有するのは、独自の抽出ルールと、保証するグラフの形状に関する独自の明示的な記述です。
