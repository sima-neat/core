---
title: "教學資源與模型封存檔"
description: "用於 Neat 教程和測試的原始碼樹資源查找、模型封存檔覆寫以及偵錯環境變數。"
sidebar_position: 7
slug: /reference/tutorial-assets
---

# 教學資源與模型封存檔

大多數教學使用者的需求僅限於 `--model <path>` 和 `--image <path>`。本頁面
適用於從原始碼樹執行、進行測試以及進行可重複的除錯工作，以便您可以
控制模型封存檔和範例素材的來源。

請勿將此內容納入初學者教學中。它很有用，但並非第一步。

## 模型封存檔覆寫

當儲存庫內的測試或輔助指令碼需要模型封存檔的路徑，但又不想每次都修改指令時，請使用這些環境變數：

| 變數 | 用法 |
| --- | --- |
| `SIMA_RESNET50_TAR` | 針對 ResNet-50 教學和測試，提供每個模型的個別覆寫設定。|
| `SIMA_MODEL_TAR` | 這是用於模型測試和範例的共享預設模型封存檔。|
| 當腳本支援時，針對 YOLO 樣式的偵測教學和測試，可以針對每個模型進行覆寫設定。| `SIMA_YOLO_TAR` |

教學指令仍然接受明確的參數。對於單次執行，建議使用 `--model <path>`；如果多個指令需要共享同一個成品，則使用環境變數。

<ShellCommand prompt="sdk-or-devkit">
export SIMA_RESNET50_TAR=/path/to/resnet_50.tar.gz
export SIMA_YOLO_TAR=/path/to/yolo_v8s.tar.gz
</ShellCommand>

## 常見的程式碼庫路徑

儲存庫內的測試和範例通常會搜尋以下目錄： `tmp/` 位於儲存庫的根目錄中。最常見的路徑如下：

| 資產 | 常用的程式碼目錄路徑 |
| --- | --- |
| ResNet-50 模型封存檔 | `tmp/resnet_50.tar.gz` |
| YOLOv8s 模型封存檔 | `tmp/yolo_v8s.tar.gz` |
| COCO 範例圖片 | `tmp/coco_sample.jpg` |

已安裝的教學程式可以從「extras」資料夾中執行。在這種情況下，如果模型和圖片的路徑不在 `/tmp` 之下，請明確地傳遞這些路徑。

## 模型提取控制項

Neat 在執行模型封存檔之前會先解壓縮它們。這些變數主要用於除錯期間的檢查和清理控制：

| 變數 | 效果 |
| --- | --- |
| `SIMA_MPK_EXTRACT_ROOT=<dir>` | 設定提取的模型資料的基本目錄。|
| `SIMA_MPK_CLEANUP_EXTRACTED=0` | 在程序結束後，會保留程序本地的 `proc_*` 提取根目錄，以便進行檢查。其他程序不會自動發現或重複使用它。|
| `SIMA_MPK_EXTRACT_GC_STALE_PROC=0` | 在啟動時停用清理過時的 `proc_*` 目錄。|

在您需要檢查產生的設定檔或比較不同執行次數產生的成品時，請使用它們。在一般教學過程中，請保留預設設定。

## 範例圖片覆寫

某些來源樹測試會使用 COCO 範例圖片，如果缺少圖片，則會自動下載。
當您的環境需要使用本機鏡像時，請覆寫 URL：

<ShellCommand prompt="sdk-or-devkit">
export SIMA_COCO_URL=https://example.com/path/to/coco_sample.jpg
</ShellCommand>

對於單次教學示範，請優先使用 `--image <path>`。當測試套件或 CI 作業需要從受控位置下載時，請使用 `SIMA_COCO_URL`。

## 對缺失的資源進行分類和優先排序。

如果教學課程輸出 `SKIP: missing ...` 或無法開啟模型：

1. 請檢查訊息中顯示的路徑。
2. 明確地將資源傳遞給教學模式。
3. 如果您執行 source-tree 測試，請設定對應的環境變數。
4. 如果該素材來自 Model Zoo，請確認已下載的檔案名稱。
   將教學指引指向實際的 `.tar.gz` 檔案。

在一般執行時，請勿手動解壓縮模型封存檔。Neat 預期會提供已編譯的封存檔路徑，並自行處理解壓縮。
