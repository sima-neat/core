---
title: "DevKit Sync"
description: "與 Modalix DevKit 共用 SDK 工作區，並在硬體上執行指令。"
sidebar_position: 3
---

:::tip 僅供參考
在安裝 SDK 時，設定 DevKit Sync 是可選的，而且安裝程式已經會提示您進行設定。請將此頁面作為參考，以了解 DevKit Sync 的運作方式，或者當您稍後想要將 SDK 配對變更為不同的 DevKit 時。
:::

DevKit Sync 將 Neat 開發環境（以下簡稱 Neat SDK）與同一個網路上的 Modalix DevKit 連接起來。它會在主機、Neat SDK 容器和 DevKit 之間公開一個共享的工作空間，並且提供 `dk` 輔助工具，以便在硬體上執行透過 SDK 建立的指令。

![主機、容器與 DevKit 之間的工作區對應關係](@site/../docs/images/elxr-sdk-workspaces.svg)

相同的工作區會掛載到 Neat SDK 容器和 DevKit 中，作為 `/workspace`，因此，您可以從每個環境中查看建置成品、日誌、追蹤資料和模型檔案。

## 設定 DevKit Sync

如果您在安裝過程中跳過了 DevKit 配對步驟，或者之後需要更改配對設定，請從主機執行此設定指令：

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

設定期間：

- 如果有多個 SDK 映像檔，請選擇已安裝的 `sdk:v2.1-latest` 映像檔。
- 除非您需要不同的路徑，否則請接受預設的 `/workspace` DevKit 掛載路徑。
- 當系統提示設定主機上的 NFS 伺服器時，請輸入主機管理員密碼。
- 當系統提示時，請輸入 DevKit 使用者的憑證。預設使用者為 `sima`，預設密碼為 `edgeai`。
- 從 SDK 2.1.2.2 開始，如果主機和 DevKit 之間無法設定 NFS，則設定程序可以回退到使用 SSH 上的 rsync。

設定成功後，您應該會看到類似以下的輸出結果：

```text
============================================================
  DevKit Connected
============================================================
  DevKit target : sima@192.168.91.221:22
  Mounted path  : /workspace
  Host export   : 192.168.74.48:/Users/joey/workspace

  You can now run DevKit binaries from this SDK shell:
    dk /workspace/<path-to-arm64-binary> [args...]
============================================================
```

從 SDK 殼層內部，使用 `dk status` 來確認已配對的 DevKit，以及目前使用的工作區同步方法：

<ShellCommand prompt="sdk">
dk status
</ShellCommand>

## 從 SDK 更新配對設定

當 Neat SDK 已經安裝完成，且您需要配對不同的 DevKit，或是在您的 DevKit IP 位址變更後，需要更新配對資訊時，請按照以下步驟操作。

從 Neat SDK 容器內部執行：

<ShellCommand prompt="sdk">
source devkit.sh {devkit-ip}
</ShellCommand>

請將 `{devkit-ip}` 替換為您想要使用的 DevKit 的 IP 位址。

範例：

<ShellCommand prompt="sdk">
source devkit.sh 192.168.91.221
</ShellCommand>

## 在沒有 DevKit Sync 的情況下設定 SDK

如果無法從 Neat SDK 主機存取 DevKit，您仍然可以設定 SDK 工作區，而無需進行配對：

<ShellCommand prompt="host">
sima-cli sdk setup
</ShellCommand>

您仍然可以在 Neat SDK 容器中建立二進位檔案，但您必須手動將它們傳輸到 DevKit 中進行測試。請確保 DevKit 正在執行相容的 Neat Library 版本。

## 使用 DevKit Sync 進行檔案共享

DevKit Sync 將三個環境連接起來：

1. 主機
2. Neat SDK 容器
3. DevKit

`sima-cli sdk setup --devkit {devkit-ip}` 會設定 NFS，以便在所有三個環境中都能使用相同的工作區：

- 主機工作區資料夾透過主機 NFS 進行匯出。
- 此資料夾會掛載到 Neat SDK 容器中，作為 `/workspace`。
- 相同內容會透過 NFS 出現在 DevKit 上，作為 `/workspace`。
- 掛載的資料夾名稱預設為 `/workspace`，並且可以在設定期間進行更改。

此設定提供一個直接的流程，用於處理建置後的成品：

- 在 Neat SDK 中產生的成品，無需額外的部署步驟，即可立即在 DevKit 上查看。
- 代理程式可以存取日誌、輸出、追蹤以及應用程式在 DevKit 上執行期間產生的其他暫時檔案。
- 開發人員和代理程式可以從同一個工作區環境中檢視相同的檔案。

透過 Insight，您可以透過網頁瀏覽器檢視工作區。某些 SiMa.ai 專用的模型封存檔，例如 `*.tar.gz` 模型成品，會自動進行最佳化，以便更輕鬆地進行檢查。

## Rsync 備援方案

從 SDK 2.1.2.2 開始，當 NFS 設定失敗時，DevKit Sync 可以使用透過 SSH 的 rsync 作為備用方案。這在網路或主機上很有用，因為在這些環境中，可以透過 SSH 連接到 DevKit，但主機上的 NFS 匯出無法由 DevKit 進行掛載。

當啟用 rsync 備援機制時：

- 主機和 SDK 容器仍然使用本機的 `/workspace` 目錄。
- DevKit 使用同步的遠端工作區，通常為 `/workspace-rsync`。
- `dk status` 報告 `Sync method : rsync`，並顯示本機和遠端工作區的路徑。
- `dk <file> [args...]` 將路徑從 SDK 工作區映射到 DevKit rsync 工作區，然後再遠端執行命令。
- 在 `dk` 執行檔案之前，它會自動同步包含該檔案的頂層工作區資料夾。例如，`dk apps/demo.py` 在執行 DevKit 端的複製之前，會同步 `/workspace/apps` 資料夾。

檢查目前的配對方式和同步方法：

<ShellCommand prompt="sdk">
dk status
</ShellCommand>

手動同步目前的工作區範圍：

<ShellCommand prompt="sdk">
dk sync
</ShellCommand>

同步特定檔案或資料夾：

<ShellCommand prompt="sdk">
dk sync /workspace/apps
</ShellCommand>

同步整個工作區：

<ShellCommand prompt="sdk">
dk sync --all
</ShellCommand>

當啟用 rsync 備援機制時，請將一個 `dk` 指令所需的檔案保存在同一個頂層工作區資料夾中。如果從 `/workspace/apps` 啟動一個指令，指向 `/workspace/models` 的參數將不在自動同步的範圍內，因此應使用 `dk sync
/workspace/models` 進行單獨同步，或者應該組織專案，使所需的檔案都位於同一個頂層資料夾中。

## 在 DevKit 上使用 dk 執行

SDK 包含 `dk` 輔助工具，也稱為 `devkit-run`，用於在 SDK 殼層內執行配對的 DevKit 上的 ARM64 可執行檔。

當您執行 `dk` 時，SDK 會在配對的 DevKit 上執行該指令，並轉換路徑，以便從容器傳遞的檔案參數能在 DevKit 上正確解析。

<ShellCommand prompt="sdk">
dk <file> [args...]
</ShellCommand>

在 SDK 工作區中編譯完 C++ 應用程式後，請在 DevKit 上執行產生的 ARM64 可執行檔：

<ShellCommand prompt="sdk">
dk build/sima_neat_hello
</ShellCommand>

在將 Python 腳本建立或複製到 SDK 工作區後，請在配對的 DevKit 上執行它：

<ShellCommand prompt="sdk">
dk hello_neat.py
</ShellCommand>

對於 Python 腳本，`dk` 會在配對的 DevKit 上執行腳本，並使用
DevKit 的 PyNeat 執行階段環境。SDK 仍然是一個有用的統一工作區和協調環境，但僅使用 Python 的工作流程不需要 C++ 交叉編譯工具鏈。

:::note `dk` 的來源
`dk` 是一個在 SDK 容器內的 `~/devkit-sync.rc` 中定義的 shell 函數。
shell 會透過 `~/.bashrc` 載入它，因此它可以在互動式會話中使用。
:::

## 下一步

若要安裝或更新函式庫/執行階段本身，請繼續前往 [Neat Library](/getting-started/neat-library/)。
