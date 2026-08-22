---
title: "sccache 快速參考指南"
description: "在本機和 Vulcan 中使用並排除 Neat 編譯器快取的錯誤。"
sidebar_position: 2
slug: /develop-apps/contribute/sccache
---

# Neat `sccache` 快速參考

Neat 使用 [`sccache`](https://github.com/mozilla/sccache) 作為 C 和 C++ 編譯器的啟動器。它會快取編譯器的輸出，而不是最終的 Neat 套件、測試結果、依賴項下載或 Docker 映像。

對於大多數開發人員來說，無需安裝或設定任何內容：正常使用 `build.sh`，並檢查在最後列印的快取統計資訊。

## 簡要說明

| 建置 | 快取層級 | 寫入位置 | 是否能保留在 `--clean` 之後 |
|---|---|---|---|
| 本機 | 使用者本機磁碟 | 本機磁碟 | 是 |
| Vulcan `develop` 或 `main` | 執行器本機磁碟，然後是受保護分支的 S3 | 本機磁碟及其受保護的 S3 命名空間 | S3 可以 |
| Vulcan 功能分支推送 | 執行器本機磁碟，然後是分支 S3 | 本機磁碟及其隔離的分支命名空間 | 直到分支刪除 |
| Vulcan 標籤或非直接引用 | 執行器本機磁碟，然後是最近的受保護 S3 | 僅限本機磁碟 | 沒有持續的執行器狀態 |

支援的進入點始終是：

```bash
./build.sh <options>
```

請勿將 `sccache` 用作編譯器的替代方案，或手動新增啟動程式選項。`build.sh` 提供了兩種 CMake 啟動程式：

```text
CMAKE_C_COMPILER_LAUNCHER
CMAKE_CXX_COMPILER_LAUNCHER
```

## 本地建置

### 一般使用

已啟用 `auto` 模式下的快取功能：

```bash
./build.sh --dev-only
./build.sh --all --clean
```

預設的快取位置和限制如下：

```text
~/.cache/sima-neat/sccache
10 GiB
```

快取資料位於 `build/` 之外。刪除 `build/` 或執行 `--clean` 不會刪除快取的編譯器輸出。

如果 `sccache` 不在 `PATH` 中，`build.sh` 會將指定的版本下載到：

```text
${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/tools/sccache/<version>/
```

此封存檔會與 `scripts/configure_sccache.sh` 中記錄的 SHA-256 進行比對。支援在 arm64 和 x86-64 架構上的 Linux 和 macOS 系統。

### 常見控制項

```bash
# Explicitly require sccache. Fail the build if it cannot be configured.
SIMANEAT_SCCACHE=on ./build.sh --all

# Disable caching for a reproducibility comparison.
SIMANEAT_SCCACHE=off ./build.sh --all --clean

# Put the local cache on a larger or faster volume.
SCCACHE_DIR=/mnt/nvme/sccache ./build.sh --all

# Change the local cache limit.
SCCACHE_CACHE_SIZE=20G ./build.sh --all
```

`SIMANEAT_SCCACHE=auto` 是預設設定。在此模式下，如果啟動程序失敗，則會產生警告，且建置程序會繼續進行，而不會使用快取。`on` 會使該失敗變成致命錯誤。

### 檢查或清除本機快取

使用 `build.sh` 選取的相同二進位檔案，或在 `PATH` 中啟用 `sccache` 時使用。

```bash
sccache --show-stats
sccache --zero-stats
sccache --show-adv-stats
```

為了釋放空間，請停止伺服器，並僅移除已設定的快取目錄：

```bash
sccache --stop-server
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/sccache"
```

在刪除之前，請確認已解決的檔案路徑。請勿刪除整個使用者快取目錄。

## Vulcan Cloud Builds

Vulcan 提供相同的本機磁碟快取，以及加密的 S3 層級：

```text
s3://sima-neat-compiler-cache-production/
  core/
    sccache-v1/
      <architecture>/
        <sdk-cache>/
          <build-mode>/
            develop/
              branches/<encoded-feature-branch>/
            main/
              branches/<encoded-feature-branch>/
```

例如：

```text
core/sccache-v1/arm64/sdk-develop/standard/develop/
```

命名空間會刻意包含：

- `sccache-v1`：快取結構描述，允許進行有意的全域重設
- 架構：可防止 arm64 和 x86-64 編譯器輸出混合
- SDK 快取識別碼：可防止不相容的 SDK/工具鏈輸出混合
- 建置模式：可將標準和模糊測試工具分開
- 受保護的基本分支：可防止 `develop` 和 `main` 寫入同一個命名空間

S3 儲存桶是私有的，使用其自己的 KMS 金鑰進行加密，並且與成品儲存桶分開。它沒有 CloudFront 發布，因為編譯器快取物件是私有的且可丟棄的。物件會在 45 天後自動過期。

### 分支存取權

| Git 參考 | OIDC 角色 | S3 模式 |
|---|---|---|
| 精確的 `refs/heads/develop` | 受保護的寫入者 | `READ_WRITE`，位於 `develop/` 中 |
| 精確的 `refs/heads/main` | 受保護的寫入者 | `READ_WRITE`，位於 `main/` 中 |
| 直接的功能分支推送 | 分支寫入者 | `READ_WRITE`，位於 `<base>/branches/<branch>/` 以下 |
| 標籤或非直接參考 | 讀取者 | 從選定的受保護基準讀取，為 `READ_ONLY` |

在第一次建置時，功能分支會將快取從最接近的受保護 Git 祖先（`develop` 或 `main`）複製到其自己的命名空間中。然後，建置會僅讀取和寫入該分支的命名空間。後續的建置會重複使用它，直到 GitHub 的分支刪除事件移除該分支的所有架構、SDK 和建置模式命名空間為止。功能分支無法寫入任何受保護的快取。

自動祖先檢測會比較與 `develop` 和 `main` 的合併基準距離。可重複使用或手動啟動的工作流程可以在需要明確基準時設定 `cache_base_branch=develop|main`。AWS 憑證是短期的 GitHub OIDC 憑證；沒有長期有效的 AWS 金鑰儲存在 GitHub 或 SDK 容器中。

預期在第一次可寫入的受保護分支建置之前，會有一個空的受保護命名空間。功能分支仍然可以填充其自己的命名空間，但如果其選定的受保護基準為空，則不會收到任何初始快取命中。

Vulcan 會在設定 CMake 之前明確地探測 `sccache` 的啟動。如果 S3、KMS、網路或臨時憑證阻止快取伺服器啟動，則工作流程會列印警告，並在沒有 `sccache` 的情況下進行編譯。因此，遠端快取可用性是一種最佳化，並且不能阻止編譯。由於 Vulcan 執行器是暫時性的，因此不會使用執行器本地快取作為後備。本地開發人員建置會保留其正常的持續磁碟快取。

## 讀取建置統計資訊

每個快取建置都會以類似以下的輸出結束：

```text
Compile requests                    623
Cache hits                          619
Cache misses                          4
Cache hits rate                   99.36 %
Cache timeouts                        0
Cache read errors                     0
Cache write errors                    0
Compilations                          4
```

以下是對重要欄位的解釋：

| 欄位 | 意義 |
|---|---|
| 編譯請求 | `sccache` 看到的編譯器調用次數 |
| 快取命中 | 請求已從快取中還原，無需執行編譯器 |
| 快取未命中 | 需要編譯的請求 |
| 編譯 | 實際執行的編譯器程序 |
| 無法快取的調用 | `sccache` 有意繞過的調用 |
| 讀/寫錯誤 | 快取後端失敗；當在本地或可寫的建置中出現非零值時，請進行調查 |
| 快取位置 | 正在使用的後端，例如本地磁碟或多層級快取 |

對於新的工具鏈、SDK 快取、架構、建置模式或實質性變更的原始碼樹的第一次建置，較低的命中率是正常的。使用相同提交和設定的第二次建置來評估快取。

當任何層級為唯讀時，`sccache` v0.16 可以將嘗試的寫入操作報告為寫入錯誤，即使唯讀建置成功。這適用於標籤和其他非直接上下文。直接的功能分支推送應報告 `READ_WRITE`；調查這些建置中的寫入錯誤。

## 快速驗證

### 驗證本地重用

執行相同的乾淨建置兩次：

```bash
SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist

SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist
```

第二次執行時，命中率應該會顯著提高。具體數值會因原始碼的變更、編譯器探測、產生的檔案以及所選取的建置方式而有所不同。

### 驗證 Vulcan 設定

在 GitHub Actions 的建置記錄中，請尋找以下內容：

```text
sccache enabled: sccache <version>
sccache local cache: <path> (<limit>)
sccache remote cache: s3://<bucket>/<prefix> (READ_ONLY|READ_WRITE)
```

然後，驗證最終的統計資料中不包含任何意外的讀取、寫入、逾時或快取錯誤。

## 疑難排解

### `sccache` 未啟用

- 確認建置程序使用 `build.sh`。
- 檢查 `SIMANEAT_SCCACHE` 是否未設定為 `off`。
- 重新執行，並設定 `SIMANEAT_SCCACHE=on`，以便將啟動錯誤視為致命錯誤。
- 確認 `curl`、`tar`，以及 `sha256sum` 或 `shasum` 是否可用。

### 第二次本機建置仍然遺漏

- 確認兩個建置程序都使用相同的編譯器、SDK、建置模式和旗標。
- 確認 `SCCACHE_DIR` 解析為相同的持續目錄。
- 尋找包含時間戳記或變更的絕對路徑的產生輸入。
- 檢查 `Non-cacheable calls` 和 `Unsupported compiler calls`。
- 確認快取未因 `SCCACHE_CACHE_SIZE` 而被清除。

### Vulcan 顯示零個遠端快取命中

- 確認選取的 `develop` 或 `main` 基線已填入資料。
- 對於功能分支，確認日誌報告了預期的基本分支及其編碼的分支特定字首。
- 比較架構、SDK 快取識別碼和建置模式。
- 確認日誌顯示預期的儲存桶和字首。
- 將「冷」命名空間視為正常情況；比較兩個相同的建置程序。

### S3 啟動失敗，出現 `AccessDenied` 錯誤

快取角色需要針對 `.sccache_check` 探測，使用以字首為範圍的 `s3:ListBucket`，以及物件權限：

- 讀取者：`GetObject`
- 寫入者：`GetObject` 和 `PutObject`

這兩個角色還需要對應的 KMS 權限。不要將 S3 或 KMS 權限新增到 EC2 執行程式角色中作為一種暫時解決方案；請修正 Vulcan 中的 GitHub OIDC 快取角色。

### 在診斷編譯器失敗時繞過快取

```bash
SIMANEAT_SCCACHE=off ./build.sh --all --clean
```

如果問題仍然存在，則並非由快取編譯器輸出所引起。

## 所有權與權威來源

| 考量點 | 來源 |
|---|---|
| 本機啟動、版本、校驗總和、快取預設值 | `scripts/configure_sccache.sh` |
| CMake 啟動器整合與統計資料 | `build.sh` |
| 分支角色選擇與快取命名空間 | `.github/workflows/vulcan-ci.yml` |
| 可重複使用的 Vulcan 工作流程輸入與 OIDC 設定 | `sima-neat/.github` |
| S3、KMS、生命週期與快取 IAM 角色 | `sima-neat/vulcan` |

當變更快取結構、工具鏈相容性邊界或存取模型時，請同時更新所有受影響的儲存庫和本頁。
