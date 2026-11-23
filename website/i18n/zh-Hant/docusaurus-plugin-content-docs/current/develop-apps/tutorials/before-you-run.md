---
title: "教學設定"
description: "選擇一個執行環境，下載教學資料，並準備模型封存檔。"
sidebar_position: 2
slug: /tutorials/before-you-run
---

# 教程設定

在開始教程之前，請先完成一次此設定。選擇與教程類別相符的環境；Neat Library 和 PCIe 套件不能互換。

## 1. 選擇您的環境

| 教程類別 | 執行於 | Python 環境 |
|---|---|---|
| 模型與推論、圖與管線、相機與串流、生成式 AI | Modalix DevKit 或教程指定的環境 | `~/pyneat` |
| PCIe 協同處理 | 連接到 Modalix PCIe 卡的主機 | `~/pyneatpcie` |

PCIe 教程在主機上執行，而不是在 SDK 容器內或直接在卡上執行。

## 2. 設定 Neat Library 教程

請確保已安裝 [Neat Library](/getting-started/neat-library/install-or-update/)，然後從您希望放置教程套件的目錄中執行以下指令：

<ShellCommand prompt="sdk|devkit">
sima-cli neat install core@v%%release.core%% -t extras
cd sima-neat-*-Linux-extras
</ShellCommand>

對於直接在 DevKit 上執行的 Python 教程，請啟用 PyNeat，並驗證導入是否成功：

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 -c "import pyneat; print('pyneat ready')"
</ShellCommand>

## 3. 設定 PCIe 教學

首先，請 [安裝並驗證 PCIe 主機套件](/getting-started/neat-library/pcie-host/)。
然後，下載適用於在主機上執行的 Ubuntu 版本的教學套件。
從您想要放置套件的目錄中執行指令。

在 Windows 上，主機套件已包含教學課程。請依照[在 Windows 上執行應用程式](/getting-started/neat-library/pcie-host/#run-applications-on-windows)的說明進行設定。

**Ubuntu 22.04：**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04：**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

驗證 PCIe PyNeat：

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. 準備模型封存檔

從 [Model Browser](/tools/model-browser/) 下載教學課程使用的模型，並從每個 ZIP 檔案中解壓縮封存檔：

<ShellCommand prompt="sdk|devkit|pcie-host">
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
unzip models/resnet_50/w8a8/*.zip resnet_50_w8a8_mpk.tar.gz
unzip models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
</ShellCommand>

Neat Library 的教學課程接受 `--model`，因此您可以直接傳遞解壓縮後的封存檔。PCIe 教學課程會在 PCIe 附加資料的根目錄中使用固定的檔案名稱：

| PCIe 教學課程 | 必要的模型檔案 |
|---|---|
| 透過 PCIe 執行您的第一個模型 | `yolo_v8s_mpk.tar.gz` |
| 執行 PCIe 非同步推論 | `yolo_v8s_mpk.tar.gz` |
| 執行多個模型 | `resnet_50_mpk.tar.gz` 和 `yolo_v8s_mpk.tar.gz` |

對於 PCIe 教學課程，請在 PCIe 附加資料的根目錄中執行這些命令，並重新命名封存檔：

<ShellCommand prompt="pcie-host">
mv resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

在 Windows 上，請在您的教學課程副本中執行下列 PowerShell 命令：

```powershell
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
Expand-Archive models\resnet_50\w8a8\*.zip -DestinationPath models\resnet_50\w8a8
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8
Copy-Item models\resnet_50\w8a8\resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

## 5. 驗證路徑和預期輸出

從提取的額外檔案的根目錄執行教學指令。確認其中包含建置輔助工具、預先建置的 C++ 程式碼，以及教學原始碼：

<ShellCommand prompt="sdk|pcie-host">
test -x build.sh
ls lib/*/tutorials/
ls share/*/tutorials/
</ShellCommand>

- 預先建置的 C++ 程式位於 `lib/<package>/tutorials/`。
- C++ 和 Python 原始碼位於 `share/<package>/tutorials/`。
- `./build.sh --list-targets` 列出您可以重新建置的 C++ 程式。
- 成功的 C++ 教程會以 `[OK]` 結束；Python 教程會列印出簡潔的結果，例如 `top1=...`、`completed=...` 或 `detections=...`。

如果某個教程報告缺少檔案，請先檢查目前目錄和模型檔案名稱。 如需更多協助，請參閱 [疑難排解](/reference/troubleshooting/)（疑難排解）。
