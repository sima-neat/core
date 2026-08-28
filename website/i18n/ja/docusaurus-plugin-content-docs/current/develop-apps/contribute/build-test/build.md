---
title: "構築する"
description: "build.sh を使用して、ソースコードから SiMa.ai Neat をビルドします。"
sidebar_position: 1
slug: /develop-apps/contribute/build
---

# Neat のビルド

このガイドでは、Neat のソースコードからのビルドについて説明します。
事前にビルドされたパッケージのインストールについては、[Neat Library](/getting-started/neat-library/) を参照してください。

`build.sh` は、サポートされているビルドのエントリーポイントです。依存関係のチェック、オプションの依存関係の同期、CMake による設定/ビルド、オプションのドキュメント生成、インストール時の整合性チェック、パッケージングなどを処理します。

## ビルド環境

`build.sh` は、アクティブな環境を自動的に検出します。

- Modalix DevKit のネイティブ環境
- Neat SDK 環境（クロスコンパイル）

どちらの環境でも、同じ `build.sh` コマンドを実行できます。

### クロスコンパイルの前提条件

クロスコンパイルは、DevKit 上で直接ビルドするよりも通常は高速ですが、その後、ビルドされたアーティファクトを DevKit に転送する必要があります。クロスコンパイルには、Neat SDK が必要です。

まず、ホストマシンに `sima-cli` をインストールし、次に SDK をインストールします。

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
sima-cli install sdk
```

`sima-cli` から指示があったら、SDKオプションを選択してください。

次に、SDKを起動します。

```bash
sima-cli sdk elxr
```

次に、SDK内に`sima-cli`をインストールし、その後、SDKのパッチをインストールします。

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
source ~/.bash_profile
sima-cli install tools/sdk-patch
```

- SDK のインストールは、Windows および Ubuntu でサポートされています。
- Modalix DevKit 上でネイティブにビルドする場合、SDK のインストール/パッチ手順は不要です。

## ビルドオプション

サポートされている `build.sh` オプション：

- `--dev-only`: コアライブラリとヘッダーのみをビルドします（デフォルト）。
- `--all`: ライブラリ + テスト + チュートリアル + Python ホイールをビルドします。ドキュメントと依存関係を有効にします。
- `--python`: 選択したターゲットに加えて、Python バインディング (`pyneat`) をビルドします。
- `--install-neat-internals`, `--install-deps`: ビルド前に、依存関係のアーティファクトをダウンロードしてインストールします。
- `--doc`: ドキュメントのみをビルドします。
- `--install`: ビルド/パッケージング後、生成されたアーティファクトを現在の環境にインストールします。ペアリングされた Neat SDK モードでは、これはペアリングされた DevKit 上にも対応するアーティファクトをデプロイおよびインストールします。
- `--no-dist`: ディストリビューションパッケージングをスキップします。
- `--clean`: 構成前に `build/` を削除します。
- `--no-doc`: ドキュメントのビルドをスキップします（`--all` を使用している場合でも）。
- `--no-node`: Node.js のインストールをスキップします（Node が存在しない場合、ドキュメントのビルドが失敗する可能性があります）。
- `--install-deps-only`: システム依存関係と依存関係ヘッダーをインストールしてから終了します。

## コンパイラキャッシュ

`build.sh` は `sccache` を自動的に有効にし、そのキャッシュは `--clean` 後も利用可能です。ローカルビルドでは、ユーザーローカルのディスクキャッシュが使用されます。Vulcan は、`develop` と `main` にそれぞれ個別の保護されたキャッシュを提供します。フィーチャーブランチは、最も近い保護されたベースから分離された書き込み可能なキャッシュをシードし、ブランチが削除されるまでそれを保持します。

ローカルコントロール、クラウドアクセスルール、キャッシュネームスペース、統計、検証、およびトラブルシューティングについては、[Neat sccache チートシート](/develop-apps/contribute/sccache) を参照してください。

## 一般的なビルド

コアライブラリのみ（デフォルト）：

```bash
./build.sh
```

完全ビルド（ライブラリ、テスト、チュートリアル、ドキュメント、ホイール、パッケージング）：

```bash
./build.sh --all
```

コアライブラリとPythonバインディング：

```bash
./build.sh --dev-only --python
```

ドキュメントのみ：

このコマンドは macOS でも動作します。

```bash
./build.sh --doc
```

ドキュメントのビルドプロセスでは、`build/autodoc/insight/neat_insight/openapi.json` でダウンロードされた OpenAPI 仕様に基づいて、Insight API のリファレンスが生成されます。ローカル開発環境では、`INSIGHT_OPENAPI_SPEC` を使用して、このデフォルト設定を上書きできます。

```bash
INSIGHT_OPENAPI_SPEC=../insight/neat_insight/openapi.json ./build.sh --doc
```

相対パスは、Coreリポジトリのルートから解決され、Docusaurusジェネレーターが実行される前に絶対パスに変換されます。選択したファイルが存在しない場合、Insight API生成ステップはスキップされ、そのパスが報告されます。

完全なビルドをクリーンにする：

```bash
./build.sh --all --clean
```

コアをビルドせずに依存関係をインストールします。

```bash
./build.sh --install-deps-only
```

## 出力

- ビルドツリー：`build/`
- Docusaurusサイトの出力（ドキュメントのビルドが実行された場合）：`website/build/`
- インストール時の整合性チェックのプレフィックス：ビルド中に表示される一意のテンポラリディレクトリ（`${TMPDIR:-/tmp}/sima-neat-install-test.XXXXXX`）。成功した場合は削除され、失敗した場合は検査のために保持されます。
- Neatパッケージのアーティファクト（`*.deb`）は、Linuxの完全なビルドで生成されます（ただし、`--no-dist`が使用されている場合は除きます）。
- Extrasパッケージ（`*extras.tar.gz`）は、Linuxの完全なビルドで生成されます（ただし、`--no-dist`が使用されている場合は除きます）。
- Pythonホイール（`dist/*.whl`）は、Pythonビルドが有効になっている場合に生成されます。

Pythonホイールは、メインのCMakeビルドによって生成された`_pyneat_core`拡張をパッケージ化します。ホイールの作成は、2番目のCMakeツリーを構成またはコンパイルしないため、ライブラリ、DEB、Extrasアーカイブ、およびホイールは、1つのコンパイルを共有します。

## ビルドプロファイルとCMakeオプション

フレームワークの最上位の`CMakeLists.txt`は、何がビルドされ、どのようにビルドされるかを制御するいくつかのオプションを公開します。以下に、重要なオプションを示します。

### ビルドプロファイル

フレームワークは、次の3つの名前付きプロファイルをサポートしています。

| プロファイル | 使用例 | ビルドされるもの |
|---|---|---|
| **Production** | 顧客向けのビルド | すべてのパブリックノード、モデルアーカイブのロード、Modalixバックエンド、最適化 |
| **Developer** | フレームワークエンジニア | Productionセット + デバッグノード + 拡張診断 + テスト |
| **Sandbox** | マルチテナントのデプロイ | Productionセット + 強化されたモデルアーカイブのセキュリティデフォルト |

構成時に`-DSIMA_NEAT_PROFILE=Production|Developer|Sandbox`を使用して選択するか、`CMakeLists.txt`でデフォルトを受け入れます。

### 共通のCMakeオプション

| オプション | デフォルト | 効果 |
|---|---|---|
| `SIMA_NEAT_BUILD_TESTS` | `ON`（Developer） | gtestスイートをビルドします。Productionビルドでの高速なCIのために無効にします。 |
| `SIMA_NEAT_BUILD_TUTORIALS` | `OFF` | チュートリアルのバイナリをビルドします。 |
| `SIMA_NEAT_BUILD_PYTHON` | `ON` | `pyneat` nanobindモジュールをビルドします。 |
| `SIMA_NEAT_BUILD_INTERNALS` | `OFF`（パブリック） | 内部のreach-through層（`core/src/pipeline/internal/sima/`）をビルドします。 |
| `SIMA_NEAT_ENABLE_TVM_FALLBACK` | `ON` | MLAが処理できないオペレーションに対して、TVMベースのフォールバックカーネルをコンパイルします。 |
| `SIMA_NEAT_ENABLE_RTSP` | `ON` | RTSPソース/シンクノードをビルドします。 |
| `SIMA_NEAT_DEBUG_PLUGINS` | `OFF` | GStreamerプラグインのデバッグをstdoutに出力します。 |
| `SIMA_NEAT_USE_SYSTEM_GSTREAMER` | `ON`（ホスト）/ `OFF`（クロス） | バンドルする代わりに、システムのGStreamerにリンクします。 |
| `SIMA_NEAT_WARN_AS_ERROR` | `OFF` | コンパイル時の警告をエラーとして扱うようにします。CIに推奨されます。 |

### ツールチェーンの調整

Modalixへのクロスコンパイルの場合：

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/modalix.cmake \
  -DSIMA_NEAT_PROFILE=Production
```

ホスト側での開発について：

```bash
cmake -B build -DSIMA_NEAT_PROFILE=Developer
```

ツリー構造で公開されている要素を列挙するには：

```bash
cmake -L -B build       # list all cache variables
cmake -LA -B build      # include advanced
```

最上位の `CMakeLists.txt` は、オプション名の定義元となります。
