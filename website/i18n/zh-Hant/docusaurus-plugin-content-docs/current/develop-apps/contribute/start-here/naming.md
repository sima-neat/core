---
title: "命名合約"
description: "針對 SiMa.ai 和 Neat，制定標準化的產品、API 和類型命名規則。"
sidebar_position: 2
slug: /develop-apps/contribute/naming
---

# 命名慣例

本文定義了 SiMa.ai Neat 程式碼庫的標準命名慣例。

## 標準名稱

- 產品名稱：`SiMa.ai Neat`
- CMake 專案：`SimaNeat`
- C++ 命名空間：`simaai::neat`
- 核心執行階段類型：`Model`、`Graph`、`Run`

## 公開的標頭檔

使用 `include/` 下的標頭檔作為公開的原始碼參考。

範例：

```cpp
#include "model/Model.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
```

## 舊版別名

舊版名稱僅作為相容性參考和移轉指南而提供。

- `PipelineSession` -> `Graph`
- `PipelineRun` -> `Run`
- `NeatModel` -> `Model`
- `InputAppSrc` -> `Input`
- `OutputAppSink` -> `Output`

請勿使用舊版符號來建立新的公開檔案/範例。

## 政策

- 所有面向使用者的新檔案都必須使用標準名稱。
- 舊版術語僅允許在移轉說明、相容性註解或明確標記的棄用部分中使用。
- CI 會透過 `scripts/ci/check_naming_and_conflicts.sh` 來強制執行此規定。
