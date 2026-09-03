---
title: "AppComplex 工作區封裝"
description: "建立並安裝一個具有閘道功能的應用程式複合工作區服務套件。"
sidebar_position: 3
slug: /develop-apps/contribute/appcomplex_workspace_packaging
---

# AppComplex 工作區封裝

本指南將 `tmp/core/sima-ai-appcomplex` 封裝到一個隔離的系統套件中，除非您明確要求，否則它不會取代現有的 `simaai-appcomplex.service`。

## 將安裝的內容

- `/opt/simaai/appcomplex-workspace/` 下的二進位檔和函式庫
- Systemd 服務單元：`simaai-appcomplex-workspace.service`
- 設定檔：`/etc/default/simaai-appcomplex-workspace`

工作區單元的預設設定為隔離的端點：

- 控制插座：`/tmp/mlactrl_workspace`
- SHM 物件：`/mlashmdata_workspace`
- MLA 初始化閘道：`APP_COMPLEX_RUN_INIT=0`（跳過並行運行的初始化）

## 建立套件

```bash
./scripts/release/build_appcomplex_workspace_deb.sh
```

這個指令碼會將產生的 `.deb` 路徑列印到 `build/packages/` 中。

## 安裝（預設啟用）

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb>
```

預設安裝行為：

- 不會停止/停用 `simaai-appcomplex.service`
- 不會自動啟用/啟動 `simaai-appcomplex-workspace.service`

## 手動啟用工作區服務

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now simaai-appcomplex-workspace.service
```

如果您需要在工作區啟動前重新初始化 MLA（用於切換模式），請設定：

```bash
sudo sed -i 's/^APP_COMPLEX_RUN_INIT=.*/APP_COMPLEX_RUN_INIT=1/' /etc/default/simaai-appcomplex-workspace
```

## 可選的切換作業（僅限明確指定的情況）

若要要求停止舊服務並啟動工作區服務，請執行以下操作：

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb> --activate --switch-system
```

或者更新 `/etc/default/simaai-appcomplex-workspace`：

- `APP_COMPLEX_ACTIVATE_ON_INSTALL=1`
- `APP_COMPLEX_SWITCH_SYSTEM_SERVICE=1`

然後執行：

```bash
sudo dpkg-reconfigure simaai-appcomplex-workspace
```
