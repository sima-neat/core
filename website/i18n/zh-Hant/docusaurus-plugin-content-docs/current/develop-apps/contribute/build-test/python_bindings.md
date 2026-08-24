---
title: "Python 綁定 (pyneat)"
description: "作為貢獻者，建立、測試和封裝 pyneat 繫結。"
sidebar_position: 3
slug: /develop-apps/contribute/python_bindings
---

# Python 繫結 (`pyneat`)

此頁面適用於 `pyneat` 的貢獻者和維護者。

`pyneat` 是 SiMa.ai Neat 的 Python 繫結層，使用 `nanobind` 建立，並使用 `scikit-build-core` 封裝。

請參閱 [Python API 參考檔案](/reference/pythonapi/modules/pyneat) 以取得產生的 API 檔案。

## 先決條件

`pyneat` 連結到與 C++ 函式庫相同的原生依賴項，包括：

- GStreamer 開發/執行階段套件
- OpenCV 開發/執行階段套件
- C++ 工具鏈 (`cmake`、編譯器、`pkg-config`)

請參閱 [建立](/develop-apps/contribute/build) 以取得主機設定指南。

## 從原始碼安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

可編輯的安裝設定，用於開發：

```bash
python -m pip install -e .[dev]
```

## 執行測試

```bash
pytest -q python/tests
```

## 封裝

位於程式碼庫根目錄中的 `pyproject.toml` 定義了用於 `pyneat` 的 wheel/sdist 建置設定。
