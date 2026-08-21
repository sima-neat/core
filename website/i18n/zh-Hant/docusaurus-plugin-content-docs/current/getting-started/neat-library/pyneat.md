---
title: "安裝 PyNeat。"
description: "在自訂的 Python 虛擬環境中安裝 PyNeat 輪包。"
sidebar_position: 4
---

:::note 僅限於 DevKit，跳過 SDK 安裝。
這些步驟會在 DevKit（或作為執行階段的主機）上設定 PyNeat。如果您正在 Neat SDK 內工作，請跳過此頁面。
:::

:::tip PyNeat 已經與 Neat Library 一起安裝好了。
PyNeat 與 Neat Library 捆綁在一起，當您安裝 Neat Library 時，它會自動安裝。

預設情況下，它會安裝在 `~/pyneat` 的虛擬環境中。除非您想將 PyNeat 安裝在自訂的虛擬環境中（例如，在 DevKit 上建立一個獨立的 venv 或 conda 環境），否則您可以跳過此頁面。
:::

在 DevKit 上執行以下步驟。此指令不會安裝或更新 `.deb` 執行階段套件，因此請在已安裝相應的 Neat Library 執行階段的環境中執行。

## 下載 Wheel

<ShellCommand prompt="devkit">
sima-cli neat install core -t pyneat
</ShellCommand>

若要下載特定 Neat Library 版本對應的 wheel 檔案，請包含版本號。

若要安裝特定版本：

<ShellCommand prompt="devkit">
sima-cli neat install core@v0.4.0 -t pyneat
</ShellCommand>

## 建立一個 Python 環境

建立並啟動一個虛擬環境，使用該環境的 `python3`：

<ShellCommand prompt="devkit">
python3 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## 安裝 Wheel

<ShellCommand prompt="devkit">
pip install ./pyneat-*.whl
</ShellCommand>

如需了解支援的 Neat Library、SDK 和 DevKit 軟體組合，請參閱
[相容性指南](/getting-started/compatibility/)。

## 下一步

繼續使用 [Neat 指令列介面](/getting-started/neat-library/neat-cli/) 來檢查已安裝的環境。
