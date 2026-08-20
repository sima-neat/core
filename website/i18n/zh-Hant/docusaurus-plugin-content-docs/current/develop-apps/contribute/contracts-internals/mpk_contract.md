---
title: "MPK 合約"
description: "模型封存檔導入合約、驗證和安全規則。"
sidebar_position: 1
slug: /develop-apps/contribute/mpk_contract
---

# MPK 合約

本文檔定義了模型檔案封存導入和 MPK 合約解析的權威合約。

## 範圍

此合約適用於：

- `src/model/ModelPack.cpp`
- `src/model/ModelArchiveLoader.cpp`
- `src/model/internal/ModelArchiveLoader.h`
- `src/pipeline/internal/sima/MpkContract.cpp`

## 接受的檔案封存格式

接受的封存檔副檔名：僅限完全小寫的 `.tar.gz`。在進行 tar 檢查之前，會拒絕 `.mpk`、`.tgz`、`.tar` 和未壓縮的 `.gz`。

檔案封存要求：

- 檔案封存必須可以作為 tar 串流讀取。
- 檔案封存大小必須小於或等於載入器設定的 `max_archive_bytes`。
- 檔案封存中的項目數量必須小於或等於載入器設定的 `max_entries`。
- 每個項目的有效負載大小必須小於或等於載入器設定的 `max_entry_bytes`。

## 允許的佈局

僅接受普通檔案進行解壓縮。允許目錄條目，但會忽略。

允許解壓縮的檔案類型：

- JSON 設定檔案 (`*.json`) -> 解壓縮到 `etc/` 目錄下
- 共享物件 (`*.so`) -> 解壓縮到 `lib/` 目錄下
- ELF 二進位檔案 (`*.elf`) -> 解壓縮到 `share/` 目錄下

所需的封存檔內容：

- MPK 推理合約 (`mpk.json` 或 `*_mpk.json`)
- 執行階段所需的載入器端階段/設定 JSON
- 至少一個模型二進位成品 (`*.elf` 或 `*.so`)

## 解壓縮安全規則

解壓縮操作必須採取「安全預設」策略。

拒絕的檔案路徑格式：

- 絕對路徑（例如 `/etc/passwd`）
- 跨目錄路徑（`..`）
- Windows 磁碟機字首（`C:`）
- 混合分隔符跨目錄路徑（`..\\`、`..//`）
- 無效的 UTF-8 檔案路徑位元組
- Unicode 斜線/反斜線混淆字元（例如 `U+FF0F`、`U+2215`、`U+FF3C`）
- 用於類似跨目錄路徑的 Unicode 點混淆字元（例如 `U+FF0E`、`U+2024`、`U+FE52`）

拒絕的條目類型：

- 符號連結條目
- 硬連結條目
- 裝置條目
- FIFO 條目

解壓縮行為：

- 永遠不要將檔案封存路徑直接寫入檔案系統輸出路徑。
- 首先正規化並驗證檔案封存條目路徑。
- 將批准的條目按內容串流解壓縮到受控的臨時根目錄。
- 永遠不要允許在解壓縮根目錄之外進行寫入。
- 拒絕重複的正規化 tar 路徑，並將其視為 `invalid_archive`。
- 拒絕 tar 標頭/校驗和損壞，並將其視為 `invalid_archive`。

## JSON 和序列驗證

`pipeline_sequence.json` 必須滿足以下條件：

- JSON 物件，其中包含非空的 `pipelines` 陣列。
- 第一個管線物件必須包含非空的 `sequence` 陣列。
- 每個階段條目必須包含：
  - `sequence_id`（整數）
  - `name`（非空字串）
  - `pluginId`（非空字串）
  - `configPath`（非空字串）
  - `processor`（非空字串）
  - `kernel`（非空字串）
- 拒絕重複的階段名稱。
- 拒絕重複的 JSON 鍵。
- 拒絕 JSON 巢狀層級超過載入器 `max_json_depth` 的情況。
- 拒絕不受支援的 `kernel` 值。
- `input` 中的階段依賴關係僅能參考：
  - `decoder`，或
  - 在穩定排序後，參考較早的階段名稱。

## 錯誤分類

所有模型檔案或 MPK 合約導入失敗都必須對應到以下其中一個類別：

- `invalid_archive`
- `path_traversal`
- `schema_error`
- `unsupported_version`
- `size_limit_exceeded`

公開的錯誤訊息必須包含分類法金鑰，以便測試可以驗證確定性的分類。

## 確定性要求

- 序列排序在重複執行時必須具有確定性。
- 位於 `tests/assets/mpk` 下的測試檔案必須能夠逐位元地重現。
- 在 `test-assets/model-archive/fixtures_manifest.json` 下產生的測試檔案清單檢查總和，是建立建構樹安全測試檔案的可靠來源。

## 測試對應要求

每個負面模型檔案/MPK 合約測試都必須驗證上述其中一個分類法金鑰。
