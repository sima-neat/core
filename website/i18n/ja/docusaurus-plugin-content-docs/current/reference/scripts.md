---
title: "使用するツールとスクリプトの一覧"
description: "`core/scripts/`と`core/tools/`には何が含まれており、それぞれをいつ使用すべきか。"
sidebar_position: 90
---

# 使用するツールとスクリプトの一覧

このフレームワークには、2つのヘルパーディレクトリが含まれています。このページはその概要です。

## `core/tools/` — ドキュメント作成およびビルドを支援するツール


| スクリプト | 目的 |
|---|---|
| `generate_api_docs.sh` | Doxygen XMLに対してdoxygen2docusaurusを実行し、C++ APIリファレンスサイト用のMarkdownを生成します。公開ヘッダーを編集した後で実行してください。|

| `generate_python_api_docs.py` | `pyneat`モジュールのドキュメンテーション文字列から、Python APIリファレンスのMarkdownを生成します。|
| `generate_tutorial_docs.py` |（チュートリアルは段階的に廃止される予定であり、このスクリプトも使用されなくなります。）|
| `postprocess_d2d_links.py` | 生成後に、DoxygenからDocusaurusへのリンクのURLスラグを修正します。`generate_api_docs.sh`によって自動的に呼び出されます。|

| `strip_empty_programlisting.py` | 空の `<programlisting>` 要素によって doxygen2docusaurus が混乱するのを防ぐための回避策。|
| `compute_version.sh` | は、`deps/manifest.json` の `package-version` と、ブランチビルドにおける Git メタデータから、フレームワークのパッケージバージョン文字列を計算します。CI およびパッケージングで使用されます。|
| `expand_code_tabs.py` | チュートリアルのソースコード内の多言語対応タブを展開します。|

| `run_clean_env.sh` | 継承された `LD_*` や `PATH` による問題を回避するため、クリーンなシェル環境でコマンドを実行します。|
| `tutorial_quality_lint.py` / `tutorial_scorecard.py` | チュートリアルの Markdown ファイルをチェックし、評価します。（チュートリアルとの連携を廃止します。）|

公開ヘッダーを編集する際の一般的な手順：

```bash
cd core
doxygen docs/doxygen/Doxyfile      # regenerate XML
bash tools/generate_api_docs.sh    # regenerate Markdown
cd website && yarn start           # preview the site
```

## `core/scripts/` — リポジトリレベルのチェックと開発支援ツール


| スクリプト | 目的 |
|---|---|
| `check_format.sh` | C++コードに対してclang-formatを実行し、変更があった場合はエラーとする。|

| `check_cmake_format.sh` / `check_cmake_style.py` | を実行し、`CMakeLists.txt` ファイルに対して cmake-format / lint を適用します。|

| `check_duplicate_includes.{sh,py}` | ヘッダーファイルで、`#include` 行の重複を検出します。|

| `check_internal_headers.sh` | `core/src/pipeline/internal/sima/` パイプラインの内部層が、パブリック/内部の境界を遵守していることを確認します。|
| `run_cpp_tidy.sh` | コードベース全体に対して clang-tidy を実行します。|
| `route_refactor_validation.sh` | これは、特定のルートプランナーに対する回帰テストであり（CIによって実行されます）、検証を行います。|
| `install_neat_plugins.sh` | フレームワークの GStreamer プラグインを、システムのプラグインディレクトリにインストールします。|
| `install_codex_skill.sh` | Codex CLIのNEATスキルをインストールします（開発者の利便性のため）。|
| `fix_devkit_runtime.sh` | 新しい開発キットのランタイムライブラリとパスを修正し、コプロセッサを再起動します。`simaai-appcomplex.service`が実行されている間は、M4のみが起動します。|
| `sync_neatdecoder.sh` / `use_neatdecoder.sh` | 組み込み版と外部版のデコーダービルドを切り替えます。|

### `core/scripts/ci/`, `core/scripts/dev/`, `core/scripts/release/`

これらのサブディレクトリには、それぞれのワークフローに属するスクリプトが格納されています。CI（継続的インテグレーション）では、`ci/` セットのスクリプトが実行され、開発者は `dev/` のスクリプトを必要に応じて実行し、リリースエンジニアリング部門は `release/` のスクリプトを実行します。アプリケーションコードからこれらのスクリプトに依存しないようにしてください。

## クリーンなチェックアウトからドキュメント生成ツールを実行する。

```bash
sudo apt-get install -y doxygen   # if not installed
cd core
doxygen docs/doxygen/Doxyfile      # generates docs/doxygen/out/xml/
bash tools/generate_api_docs.sh    # populates docs/reference/cppapi/
python3 tools/generate_python_api_docs.py   # populates docs/reference/pythonapi/
cd website && yarn install && yarn start    # serve at http://localhost:3000/
```

## 関連資料

- 「ツールとスクリプト」— デザインの詳細解説の第55項。
- リポジトリ `core/AGENTS.md` には、pre-commitで実行する必要があるツールに関するコントリビューター契約が記載されています。
