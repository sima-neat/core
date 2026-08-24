---
title: "Pythonバインディング（pyneat）"
description: "コントリビューターとして、pyneatのバインディングをビルド、テスト、パッケージ化します。"
sidebar_position: 3
slug: /develop-apps/contribute/python_bindings
---

# Pythonバインディング (`pyneat`)

このページは、`pyneat` のコントリビューターおよびメンテナー向けです。

`pyneat` は、SiMa.ai Neat の Python バインディングレイヤーであり、`nanobind` を使用して構築され、`scikit-build-core` でパッケージ化されています。

生成された API ドキュメントについては、[Python API リファレンス](/reference/pythonapi/modules/pyneat) を参照してください。

## 前提条件

`pyneat` は、C++ ライブラリと同じネイティブ依存関係にリンクします。これには、以下が含まれます。

- GStreamer 開発/ランタイム パッケージ
- OpenCV 開発/ランタイム パッケージ
- C++ ツールチェーン (`cmake`、コンパイラ、`pkg-config`)

ホストのセットアップについては、[構築する](/develop-apps/contribute/build) を参照してください。

## ソースコードからのインストール

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

開発用に編集可能なインストール方法：

```bash
python -m pip install -e .[dev]
```

## テストを実行する

```bash
pytest -q python/tests
```

## パッケージング

リポジトリのルートにある `pyproject.toml` は、`pyneat` の wheel/sdist ビルド設定を定義します。
