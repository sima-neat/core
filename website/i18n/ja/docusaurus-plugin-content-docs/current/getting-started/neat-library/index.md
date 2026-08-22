---
title: "Neatライブラリ"
description: "DevKitまたはNeat SDKでSiMa.ai Neatライブラリをインストール・更新する"
sidebar_position: 1
---

Neatライブラリは、Modalix向けのモデル実行とアプリケーション開発に使用するC++およびPython APIです。

:::tip Neat SDKを使用している場合、このセクションは省略できます
Neat SDKコンテナにはNeatライブラリがインストール済みです。セットアップ時にDevKitをペアリングすると、対応するNeatライブラリと付属のPyNeat環境もDevKitへ自動的にインストールされます。DevKitをペアリングしたかどうかにかかわらず、Neat SDKをインストール済みであればNeatライブラリも利用できるため、このセクションを省略して先へ進めます。

SDKとは別にNeatライブラリのみを更新する場合、またはNeat SDKを使わずにDevKitや通常のホストへ直接セットアップする場合にだけ、このセクションを使用してください。
:::

SDKコンテナを再インストールせず、互換性のある新しいバージョンが必要な場合は、Neatライブラリだけを更新できます。サポートされる組み合わせについては、[互換性ガイド](/getting-started/compatibility/)を参照してください。

<div class="overview-section-label">次のステップ</div>

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>ライブラリのセットアップ</h2>
    <p>Neatライブラリのインストール、PCIeホストのセットアップ、Pythonバインディングの利用、コマンドラインからの環境確認を行います。</p>
    <ul class="overview-link-list neat-library-setup-grid">
      <li><a class="overview-link-card" href="/getting-started/neat-library/install-or-update/"><strong>Neatライブラリのインストールまたは更新</strong><span>sima-neatパッケージ、GStreamerコンポーネント、標準のPyNeat環境をインストールします。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/pcie-host/"><strong>PCIeホストのインストール</strong><span>接続したModalix PCIeカードでコプロセッシング・アプリケーションを実行するためのPCIeホストパッケージをインストールします。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/pyneat/"><strong>PyNeatのインストール</strong><span>独自のPython仮想環境でPyNeat wheelを使用します。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/neat-cli/"><strong>Neat CLI</strong><span>インストール済みのバージョンやSDKの状態を確認し、コンポーネントを更新します。</span></a></li>
    </ul>
  </section>
</div>

## 次のステップ

[Neatライブラリのインストールまたは更新](/getting-started/neat-library/install-or-update/)から始めてください。
