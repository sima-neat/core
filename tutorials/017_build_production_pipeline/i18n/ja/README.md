# 017 本番環境で利用可能なパイプラインを構築する

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | resnet_50 |
| Labels | production, reliability, deployment |

## Concept

これまでの章で学んだパターン（明示的なモデルオプション、明示的なルートオプション、明示的な実行オプション、および1つの非同期プッシュ/プルループ）から、本番環境で使用できる実行ループを組み立てます。これは完全な製品フレームワークではなく、適応可能な信頼性の高い基本構造です。

## Walkthrough

これは集大成の章です。これまでの内容は、それぞれ独立した概念でしたが、ここではそれらが統合され、実際のデプロイメントコードに適用できる単一の設計図となります。このテンプレートの主な目的は、デフォルト設定では暗黙的になってしまう3つの要素を明示的にすることです。それは、モデルの入力範囲（これにより、契約違反は実行中にではなく、ビルド時に検出される）、ステージの名前（これにより、複数のモデルが同じプロセスを共有する場合でも、診断結果が読みやすくなる）、およびキューポリシー（これにより、負荷時の動作が予測可能になり、不可解な状態になるのを防ぐ）です。

全体の流れは次のとおりです。実行オプションを設定し、モデルを構成およびロードし、ランナーを構築し、次に、制限された非同期ループで実行します。最終的に、あなたは、同じアプリケーション内の複数のモデルで標準化できる、プロダクションのデフォルト設定と、正常な出力をカウントするプッシュ/プルループを備えた非同期パイプラインを実行する`Runner`を持つことになります。これはランタイムのスケルトンです。

### 実行オプションを設定する {#step-configure-run-options}

これらは、プロダクションランタイムのデフォルト設定です。`queue_depth = 8`は、小さな制限されたバッファーを提供します。`overflow_policy = Block`は、プロデューサーがフレームをサイレントに破棄するのではなく、待機するようにします（損失が問題になる場合は、安全な選択肢です）。`output_memory = Owned`は、返されたテンソルがプルの後も存続することを保証します。これらの設定を明示的に行う（デフォルトに依存するのではなく）ことで、負荷時の動作が予測可能になります。

### モデルを構成およびロードする {#step-configure-model}

ここでは、モデルの入力契約を明示的にします。`preprocess.input_max_width/height/depth`をフレームの寸法に設定すると、入力が一致しない場合、明確な契約違反エラーが発生し、ビルド時に失敗します。これにより、後で混乱を招くランタイムエラーが発生するのを防ぎます。`name_suffix = "_prod"`は、このモデルのステージにタグを付けて、マルチモデルアプリケーション全体で診断結果を特定できるようにします。次に、アーカイブパスとこれらのオプションから`Model`を構築します。

**C++:** `Model::Options`は、モデルが期待する前処理（`InputKind::Image`、RGBカラー変換、および`has_explicit_stats = true`によるImageNet正規化）も明示的に指定します。これは、C++パスが前処理を事前に宣言し、アーカイブのデフォルトに依存しないためです。

**Python:** `ModelOptions`は、`mopt.preprocess.*`の下で、画像の前処理、入力範囲、ImageNet正規化、およびサフィックスを設定します。

### ランナーを構築する {#step-build-runner}

`ModelRouteOptions` (C++ `Model::RouteOptions`) は、ルートに含める境界を決定します。ここでは、`include_input` と `include_output` の両方が true に設定されており、ルートの要素がモデルの命名規則と一致するように、同じ `_prod` サフィックスが使用されます。次に、`model.build(sample, route_options, run_options)` を呼び出します。これは、`Model` を直接実行可能な `Runner` に接続し、ルートと実行オプションを基盤となるパイプラインに転送する、単一の呼び出しパスです。この代表的なサンプルにより、ビルドプロセスでネゴシエートされた形状を固定できます。

**C++:** サンプルは、`Tensor::from_cv_mat(rgb, ..., TensorMemory::EV74)` を使用して構築された `TensorList` であり、入力データをデバイスに適したメモリに配置します。

**Python:** サンプルは、`Tensor.from_numpy(...)` から作成された 1 つの `Tensor` を含むリストです。

### プロダクションループの実行 {#step-run-loop}

これは、実際のサービスが実行するループです。各イテレーションで、入力を `push(...)` し、ブール値をチェックします。これにより、拒否されたプッシュ（`Block`、一時的な状態）が誤ってカウントされるのではなく、適切に処理されます。次に、有限のタイムアウトで `pull(...)` を実行し、成功した出力をカウントします。ループの最後に、`close()` を呼び出して、ランナーをクリーンにシャットダウンします。このプッシュ-ブール / タイムアウト付きプル / 明示的なクローズのパターンは、信頼性の高い非同期の基本構造です。実際の入力と出力処理に置き換えても、構造は変わりません。

## Run

この章では、モデルアーカイブ（`resnet_50`）が必要です。**Neat インストールルート**（`share/` と `lib/` を含むディレクトリ）から、**Python** および **C++（事前にビルドされたもの）** のコマンドを実行します。**ソースコードからビルドする** コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/017_build_production_pipeline/build_production_pipeline.py \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_017_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_017_build_production_pipeline
./build/tutorials-standalone/tutorial_017_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

期待される出力：

```text
outputs=4
[OK] 017_build_production_pipeline
```

（Python ビルドは、`iters=4 ok=4` を出力します。）

この章の C++ ソースコードを、カスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## ソースファイル
- C++: `tutorials/017_build_production_pipeline/build_production_pipeline.cpp`
- Python: `tutorials/017_build_production_pipeline/build_production_pipeline.py`
