# 010 1つのサンプルで複数の入力を送信する

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | None |
| Labels | multi-input, samples, sync |

## Concept

複数の名前付きテンソルを1つの`Sample`にまとめて、単一の推論イベントとして送信します。これは、ステレオ画像、画像＋メタデータ、センサーフュージョンなど、複数の入力を受け取るモデルのパターンです。

## Walkthrough

多くの実際のアプリケーションでは、1回の推論処理で複数の入力を使用します。 Neat これを**バンドルサンプル**として表現します。これは単一のものです。 `Sample` 誰の `fields` リストには複数の名前付きテンソルペイロードが含まれており、それぞれにアドレスが割り当てられています。 `port_name`ランタイムは、指定されたフィールドをまとめて1つの論理的なイベントとして保持します。 `left` そして `right` (または画像とメタデータ) がパイプライン全体を通して整合性を保ちます。

この章では、テンソルを入力として受け取り、テンソルを出力するグラフを構築し、2つの名前付きの浮動小数点テンソルをまとめて、パイプラインを通して処理し、名前付きフィールドを読み出します。最終的には、複数のフィールドを持つサンプルを構築し、両方のフィールドがポート名を含めて変更されずに処理されたことを確認します。

### テンソル入力の設定 {#step-configure-tensor-input}

このグラフはデコードされた画像ではなく、生のテンソルを消費するため、入力の形式はテンソルペイロードとして定義されます。`FP32`、 `width`/`height`/`depth`)というよりは、ピクセル形式です。これにより、入力ノードはテンソルバッファーを直接受け入れるようになります。

**C++:** `in.payload_type = PayloadType::Tensor` を設定します。

**Python:** `inp.payload_type = pyneat.PayloadType.Tensor` と `inp.format = pyneat.Format.FP32` を設定します。

### グラフと初期実行を構築する {#step-build-seed-run}

当社は同じ最小限の `Input -> Output` 第004章のトポロジーから `build()` それを `Run`. `build()` 交渉によって決定された形状を固定するには、代表的なサンプルが必要です。そのため、実際のフィールドで使用するのと同じ形状の単一のシードテンソル（すべてゼロ）を渡します。シードは形状の交渉のためだけに使用され、実際のデータは次に処理されます。

### バンドルを構築します {#step-make-bundle}

それでは、複数の入力をまとめてイベントを作成します。各入力には、次のように名前が付けられます。 `make_tensor_sample(port_name, tensor)`そして、それらの名前付きフィールドは、モデルがポートを通じて扱う対象となります。 `left` ～で満たされています `1.0` そして `right` ～とともに `2.0` そのため、移動中にそれらを区別することができます。

**C++:** `make_bundle_sample({...})` 指定されたフィールドをまとめて処理します。 `Sample` 誰の `kind` は `Bundle`**Python:** 名前付きのサンプルの一覧が、直接渡されます。 `push(...)`; pyneat が、バンドルをまとめる処理を行います。

### バンドルをプッシュし、その内容を読み出します。 {#step-push-and-read}

最後に、バンドルを送信して結果を確認します。出力自体もバンドルです。 `Sample`なので、私たちは読みました。 `out.fields` 単一のテンソルとして扱うのではなく、 `out.fields.size()` ～すべきである `2`、そして各フィールドは次のような情報を含みます。 `port_name` およびテンソルペイロード。

**C++:** `run.run(Sample{bundle}, timeout_ms)` 1つを返します `Sample`論理的な結果には複数のフィールドがあるため、返された `Sample` それ自体が `Bundle` — そこで確認します `out.kind == SampleKind::Bundle` そして繰り返す `out.fields`ではありません `front()` （これは「バンドル内の最初のフィールド」という意味になります）。

**Python:** `run.push(fields)` それから `run.pull(timeout_ms=...)` 出力サンプルを返します。反復処理を行います。 `out.fields` そして、それぞれを読みます `field.port_name` そして `field.tensor`.

## Run

「**Python**」と「**C++（事前にビルドされたもの）**」のコマンドを、以下の場所から実行します。Neat root をインストールします（ディレクトリには、次のものが含まれます）。 `share/` そして `lib/`); **ソースコードからビルドする**ためのコマンドを**リポジトリのルートディレクトリ**から実行します。この章ではモデルアーカイブは必要ありません。

**Python:**
```bash
python3 share/sima-neat/tutorials/010_feed_multi_input_model/feed_multi_input_model.py \
  --width 64 --height 48
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_010_feed_multi_input_model
./build/tutorials-standalone/tutorial_010_feed_multi_input_model \
  --width 64 --height 48
```

期待される出力（C++）：

```text
bundle_fields=2
  field=left has_tensor=yes
  field=right has_tensor=yes
[OK] 010_feed_multi_input_model
```

（Pythonのビルドでは、`port=left has_tensor=True` の行と同じフィールド数が表示されます。）この章のC++ソースコードをカスタムの `CMakeLists.txt` を使って独自のプロジェクトに組み込むには（追加のフォルダーは不要です）、ランディングページの[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## In Practice

この2つのフィールドを使用したデモを超えて、バンドルパターンをどのように適用するか。

### 名前付けとルーティング

- `port_name` 配線契約とは、複数の入力を持つモデルが各フィールドをどのように処理するかを定めたものです。モデルで宣言された入力ポートと名前を一致させてください。

- 出力バンドルはフィールド構造を保持するため、位置ではなく名前で結果と入力を対応させることができます。

### 出力バンドルの確認

- 常に分岐処理を行います。 `kind` まず、複数のフィールドを持つ結果は `SampleKind.Bundle`そして、それを単一のテンソルとして読み込もうとすると、正しく処理できません。
- 各フィールドにおけるテンソルの存在を確認してください。`field.tensor is not None` / `field.tensor.has_value()`ペイロードにアクセスする前に — フィールドにはテンソルではなく、メタデータが含まれている場合があります。

## ソースファイル
- C++: `tutorials/010_feed_multi_input_model/feed_multi_input_model.cpp`
- Python： `tutorials/010_feed_multi_input_model/feed_multi_input_model.py`
