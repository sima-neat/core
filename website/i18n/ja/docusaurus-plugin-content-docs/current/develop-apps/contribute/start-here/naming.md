---
title: "契約書の名称"
description: "SiMa.aiおよびNeatにおける、標準化された製品名、API名、および型名の命名規則。"
sidebar_position: 2
slug: /develop-apps/contribute/naming
---

# 命名規則

このドキュメントは、SiMa.ai Neat コードベースの標準的な命名規則を定義します。

## 標準名

- 製品名：`SiMa.ai Neat`
- CMake プロジェクト：`SimaNeat`
- C++ 名前空間：`simaai::neat`
- コア ランタイム型：`Model`、`Graph`、`Run`

## パブリック インクルード

`include/` の下にあるヘッダーを、パブリックな信頼できる情報源として使用します。

例：

```cpp
#include "model/Model.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
```

## 従来のエイリアス

従来の名称は、互換性のための参照および移行ガイドとしてのみサポートされます。

- `PipelineSession` -> `Graph`
- `PipelineRun` -> `Run`
- `NeatModel` -> `Model`
- `InputAppSrc` -> `Input`
- `OutputAppSink` -> `Output`

従来のシンボルを使用した新しい公開ドキュメント/サンプルを作成しないでください。

## ポリシー

- 新しいユーザー向けドキュメントでは、標準的な名称を使用する必要があります。
- 従来の用語は、移行に関する注記、互換性に関するコメント、または明確に区別された非推奨セクションでのみ使用できます。
- CI（継続的インテグレーション）は、`scripts/ci/check_naming_and_conflicts.sh` を使用してこれを強制します。
