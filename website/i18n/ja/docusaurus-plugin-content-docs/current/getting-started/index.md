---
title: "Palette Neat"
sidebar_position: 1
---

# Palette Neat

Palette Neat は、Modalix 上で AI アプリケーションを構築するための SiMa.ai
ソフトウェア開発ツールキットです。開発環境、ランタイムライブラリ、モデルツール、
DevKit 検証ワークフローが含まれています。これらのコンポーネントにより、モデルの準備、
アプリケーションの構築、Modalix ハードウェアでの結果の検証という一連の開発工程を
サポートします。

この概要では、Palette Neat の主要な構成要素を理解し、セットアップ、モデル準備、
アプリケーション開発に適した経路を選択できます。

![Neat SDK、Neat Library、Model Compiler、LLiMa、Modalix DevKit で構成される Palette Neat ソフトウェアスタック](@site/../docs/images/neat-software-stack-animated.svg)

<div class="overview-section-label">開発者向けガイド</div>

:::tip Palette Neat を初めて使用する場合
Palette Neat でアプリケーションを開発する方法は 2 つあります。

- モデルのコンパイルや大規模な C++ コードのクロスコンパイルを予定しているなど、
  より高性能な開発環境が必要な場合は、**[Neat SDK を使用する](/getting-started/dev-environment/)**を選択します。
- 特にモデルのコンパイルを行わず、開発環境の構成要素を減らしたい場合は、
  **[DevKit 上で直接開発する](/getting-started/neat-library/)**を選択します。

SDK の経路を選択する場合は、ホストが[ホスト要件](/getting-started/dev-environment/#host-requirements)を
満たしていることを確認してから SDK をインストールしてください。SDK のインストールでは
互換性のある既定値が適用されるため、正確なバージョンを固定する場合、コンポーネントを
個別にアップグレードする場合、またはバージョンの不一致をトラブルシューティングする場合にのみ、
互換性リファレンスが必要です。

**SDK のみの最短経路（DevKit なし）：** SDK をインストールしたら、
[モデルのコンパイル](/compile-a-model/)へ直接進みます。SDK の構成、DevKit Sync、
Neat Library、PyNeat は、DevKit を接続するまで省略できる任意の手順です。
:::

<div class="overview-section-label">コマンドの読み方</div>

このドキュメントのコマンドブロックには、実行する環境がラベルと色分けで示されています。どこで入力すればよいか迷う必要はありません。

| プロンプト | 実行する場所 |
| --- | --- |
| `host$` | SDK の外側、自分のマシン上。 |
| `sdk$` | Neat SDK コンテナのシェル内。 |
| `devkit$` | Modalix DevKit 上。 |
| `pcie-host$` | Modalix PCIe カードを搭載したホストマシン上。 |

`sdk or devkit$` のように複数の環境がラベル付けされたブロックは、どちらの環境でも同じように実行できます。コマンドのパスが相対パスの場合は、実行するディレクトリもブロックに示されます。

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>はじめに</h2>
    <p>ローカル開発とハードウェア検証のために、ホストマシン、Neat SDK、DevKit を準備します。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/"><strong>Neat SDK</strong><span>高性能なホストベースのワークフロー、モデルコンパイル、C++ ビルド、DevKit 検証に SDK を使用します。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/"><strong>Neat Library</strong><span>DevKit 上で少ない構成要素でアプリケーションを試作する場合は、ランタイムと PyNeat を直接インストールします。</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/compatibility/"><strong>互換性ガイド</strong><span>サポートされるバージョンの組み合わせを確認するためのリファレンスです。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>モデルの準備</h2>
    <p>学習済みモデルを Modalix ハードウェアで実行できるデプロイ可能なアーティファクトに変換します。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/compile-a-model/"><strong>モデルのコンパイル</strong><span>事前学習済み ONNX ビジョンモデルまたは GenAI モデルを Modalix 向けにコンパイルします。</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-zoo/"><strong>コンパイル済みモデルの使用</strong><span>すぐに実行できるモデルアーティファクトで迅速に開始します。</span></a></li>
      <li><a class="overview-link-card" href="/genai-llima/"><strong>LLiMa による GenAI</strong><span>Modalix 上で GenAI モデルをコンパイル、テスト、ベンチマークします。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>アプリケーションの構築</h2>
    <p>Neat Library を使用してモデルを実行し、本番アプリケーションのパイプラインを構成します。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Hello Neat!</strong><span>最初の Neat アプリケーションを実行して、開発ワークフローを確認します。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/"><strong>アプリケーション開発</strong><span>C++ または PyNeat を使用して、Neat Library で AI アプリケーションを構築します。</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>チュートリアル</strong><span>実際の Neat アプリケーションパターンをガイド付きの例で学習します。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>ツールとリファレンス</h2>
    <p>詳細が必要な場合は、補助ツールとリファレンス資料を利用します。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/tools/"><strong>ツール</strong><span>sima-cli、Model Zoo、Insight。</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>リファレンス</strong><span>API、トラブルシューティング、環境変数、データ形式、リリースノートを参照します。</span></a></li>
      <li><a class="overview-link-card" href="/reference/troubleshooting/"><strong>トラブルシューティング</strong><span>セットアップ、ランタイム、パイプラインの問題に対する解決策を確認します。</span></a></li>
    </ul>
  </section>
</div>
