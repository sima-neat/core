---
title: "AppComplex ワークスペースのパッケージ化"
description: "ゲート付きのアプリケーション複合ワークスペースサービスパッケージを構築およびインストールします。"
sidebar_position: 3
slug: /develop-apps/contribute/appcomplex_workspace_packaging
---

# AppComplex ワークスペースのパッケージング

このガイドでは、`tmp/core/sima-ai-appcomplex` を、既存の `simaai-appcomplex.service` を明示的に要求しない限り、置き換えを行わない、隔離されたシステムパッケージとしてパッケージングします。

## インストールされるもの

- `/opt/simaai/appcomplex-workspace/` 内のバイナリとライブラリ
- Systemd ユニット: `simaai-appcomplex-workspace.service`
- 構成ファイル: `/etc/default/simaai-appcomplex-workspace`

ワークスペースユニットは、デフォルトで隔離されたエンドポイントを使用します。

- コントロールソケット: `/tmp/mlactrl_workspace`
- SHM オブジェクト: `/mlashmdata_workspace`
- MLA 初期化ゲート: `APP_COMPLEX_RUN_INIT=0` (並列実行の場合は初期化をスキップ)

## パッケージの作成

```bash
./scripts/release/build_appcomplex_workspace_deb.sh
```

このスクリプトは、生成された `.deb` のパスを `build/packages/` に出力します。

## インストール（デフォルトでは制限付き）

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb>
```

デフォルトのインストール時の動作：

- `simaai-appcomplex.service` を停止または無効にしません。
- `simaai-appcomplex-workspace.service` を自動的に有効化または起動しません。

## ワークスペースサービスを手動で有効にする

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now simaai-appcomplex-workspace.service
```

ワークスペースの起動前に MLA の再初期化が必要な場合（切り替えモードの場合）、以下を設定してください。

```bash
sudo sed -i 's/^APP_COMPLEX_RUN_INIT=.*/APP_COMPLEX_RUN_INIT=1/' /etc/default/simaai-appcomplex-workspace
```

## オプションの切り替え（明示的な指定のみ）

古いサービスを停止し、ワークスペースサービスを有効にするには、以下のようにリクエストしてください。

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb> --activate --switch-system
```

または、`/etc/default/simaai-appcomplex-workspace` を更新します。

- `APP_COMPLEX_ACTIVATE_ON_INSTALL=1`
- `APP_COMPLEX_SWITCH_SYSTEM_SERVICE=1`

その後、次のコマンドを実行します。

```bash
sudo dpkg-reconfigure simaai-appcomplex-workspace
```
