---
title: "アプリを開発する"
description: "Modalix を使用して、SiMa.ai と Neat を活用し、AIアプリケーションを構築および実行します。"
sidebar_position: 1
---

# SiMa.aiとNeatを使ってアプリを開発しましょう。

<LanguageContent lang="cpp">

<div className="overview-workflow-image overview-workflow-image-light">

![SiMa.ai Neat で構成した C++ アプリケーションのワークフロー](@site/../docs/develop-apps/images/neat-overview-workflow-cpp.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![SiMa.ai Neat で構成した C++ アプリケーションのワークフロー](@site/../docs/develop-apps/images/neat-overview-workflow-cpp-dark.svg)

</div>

</LanguageContent>

<LanguageContent lang="py">

<div className="overview-workflow-image overview-workflow-image-light">

![SiMa.ai Neat で構成した Python アプリケーションのワークフロー](@site/../docs/develop-apps/images/neat-overview-workflow-python.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![SiMa.ai Neat で構成した Python アプリケーションのワークフロー](@site/../docs/develop-apps/images/neat-overview-workflow-python-dark.svg)

</div>

</LanguageContent>

## 何 SiMa.ai Neat ～ですか

SiMa.ai Neat は、SiMa.ai プラットフォーム上で AI アプリケーションを構築および実行するためのアプリケーション開発フレームワークです。コンパイルされたモデルアーカイブ（`.tar.gz`）をロードおよび実行するための Python および C++ API を提供し、Modalix 処理リソースを使用するエンドツーエンドのアプリケーションを構築し、ランタイム実行を管理します。

より広範な SiMa.ai ソフトウェアスタックにおいて、SiMa.ai Neat はアプリケーション層に位置します。SiMa.ai ランタイムスタックを基盤とし、その下に GStreamer を使用するため、開発者は、より低レベルのランタイムコンポーネントを手動で接続するのではなく、アプリケーションロジックに集中できます。

推論への最短経路として、コンパイルされたモデルアーカイブを`Model` としてロードし、直接実行します。アプリケーションが複数の入力、処理ステージ、モデル、または出力を必要とする場合は、これらのコンポーネントを`Graph` として構成し、`Run` に組み込みます。同じパブリック API は、従来の開発とエージェントベースの開発の両方をサポートするため、チームはどちらのワークフローを使用してでも、アプリケーションをレビュー、拡張、および保守できます。

### 最適なデプロイモデルを選択してください。

- **Modalix DevKit 上で実行** — アプリケーションと Neat グラフが直接実行されます。
  デバイスを使用します。`Model`、`Graph`、`Node`、および`Run`とともに、`simaai::neat`または`pyneat`を使用してください。[実行 / 推論](/develop-apps/development-workflow/overview/)から開始します。
- **共同処理のために、Modalix PCIeカードを使用してください** — アプリケーションは、
  ホストマシンからテンソルまたは画像をカードに送信し、モデルを実行します。`simaai::neat::pcie`または`pyneatpcie`を使用してください。[PCIeによる共同処理](/develop-apps/development-workflow/pcie-model/)から開始します。

### C++ または PyNeat

直接実行されるアプリケーションの場合 Modalix DevKit, SiMa.ai Neat 2つの言語インターフェースを通じて、同じ基本的なワークフローを提供するため、お客様のアプリケーションに最適なものを選択できます。

- **PyNeat** — Pythonバインディング（`pyneat`）。迅速な反復処理、ノートブック、データサイエンスのワークフロー、およびDevKit上でPythonアプリケーションを直接実行する場合に最適です。
- **C++** — ネイティブな `simaai::neat` API。大規模なアプリケーション、既存のC++コードベースとの緊密な統合、およびクロスコンパイルされたホストからDevKitへのワークフローに最適です。

どちらも同じコンパイルされたモデルのアーティファクトとModalixランタイムを使用します。以下に示す概念とページは、どちらにも適用されます。PCIeコプロセッシングは、`simaai::neat::pcie`と`pyneatpcie`を通じて、個別のC++およびPythonインターフェースを提供します。

## アプリケーションを開発してください。 <span className="neat-heading-highlight">SiMa.ai Neat あなたのために地図を作成します。</span>

Modalixは、アプリケーションコア、画像処理、機械学習の高速化、ビデオエンジン、共有メモリ、および高速I/Oを1つのSoCに統合しています。PythonおよびC++ APIを通じて、SiMa.ai Neatは、システム内のアプリケーションに関連する処理リソース全体でアプリケーションを構築するための、単一のプログラミングモデルを提供します。

カメラまたはネットワークストリームから、処理および推論を経て最終結果に至るまでのエンドツーエンドのフローを構築します。SiMa.ai Neatは、ランタイムパイプラインを構築し、適用可能な場合は高速化された実装を選択し、Modalix全体での実行とデータ移動を調整します。あなたはアプリケーションに集中し、SiMa.ai Neatが基盤となるハードウェアとランタイムの複雑さを処理します。

<div className="overview-workflow-image modalix-application-map-desktop">

![MLSoC Modalix のフロアプランに配置された SiMa.ai Neat アプリケーション](@site/../docs/images/neat-modalix-floorplan-animated.svg)

</div>

<div className="overview-workflow-image modalix-application-map-mobile">

![MLSoC Modalix のフロアプランに配置された SiMa.ai Neat アプリケーションのモバイル表示](@site/../docs/images/neat-modalix-floorplan-mobile-animated.svg)

</div>

<p className="overview-figure-caption"><strong>例示的なマッピング：</strong> 選択するルートは、アプリケーション、モデル、および利用可能なハードウェアアクセラレーションによって異なります。詳細については、こちらをご覧ください。 <a href="/develop-apps/advanced-concepts/processor_backends/">プロセッサのバックエンド</a> 技術的なマッピングのため。</p>

## あなたのアプリケーションについて説明してください。 <span className="agentic-heading-highlight">～を持つエージェント Neat スキルがそれを発展させます。</span>

SiMa.aiとNeatは、Neat SDK（Neat開発環境）に含まれるスキルを通じて、すぐに利用できるエージェントベースのアプリケーション開発をサポートします。これらのスキルにより、コーディングエージェントは、公開されているPythonおよびC++ APIを使用し、確立されたアプリケーションパターンに従い、Modalixの開発および検証ワークフローで作業するためのコンテキストを得ることができます。

推奨されるエージェントベースのパスでは、アプリケーションを作成し、ペアリングされたModalix DevKit上で実行し、結果と診断を調べ、実装を改良することができます。従来の開発は、同じAPIを通じて直接制御するための並行パスとして残ります。どちらの方法でも、標準的で検証可能なSiMa.ai Neatアプリケーションが生成されるため、エージェントによって開発されたコードを確認または変更し、アプリケーションの進化に合わせて2つのワークフローを切り替えることができます。エージェントベースの開発を有効にするには、[Neat SDK をセットアップします。](/getting-started/dev-environment/)を参照してください。

<div className="overview-workflow-image agentic-visual-desktop">

![コーディングエージェントが SiMa.ai Neat アプリケーションを作成、実行、診断、改善する流れ](@site/../docs/images/agentic-development-loop-animated.svg)

</div>

<div className="overview-workflow-image agentic-visual-mobile">

![SiMa.ai Neat のエージェント開発ループのモバイル表示](@site/../docs/images/agentic-development-loop-mobile-animated.svg)

</div>

## 要件

アプリケーションを開発する前に、まず「はじめてのステップ」の手順を完了させてください。

- **デプロイモデルのインストール** — Modalix DevKit の場合は、Neat をインストールしてください。
  Neat SDK に含まれるライブラリを使用するか、デバイスに直接インストールします。PCIe によるコプロセッシングを行う場合は、ホストマシンに `core/pciehost` をインストールし、Modalix PCIe カードに互換性のある Neat Library をインストールします。
- **モデルのアーティファクト** — Model Zoo から事前にコンパイルされたモデルを使用するか、独自のモデルを Modalix で使用できる形式のアーカイブにコンパイルします。
- **ランタイムターゲット** — DevKit 上でネイティブアプリケーションを実行するか、またはアプリケーションをビルドして実行します。
  ホストマシン上で直接コプロセッシングアプリケーションを実行します。Neat SDKでネイティブC++アプリケーションをクロスコンパイルする際に、DevKitをペアリングおよび同期してください。

「Hello Neat!」のページでは、最初の推論を実行する方法を説明し、「開発ワークフロー」のページでは、主要な概念をより詳細に解説し、チュートリアルでは、実際のアプリケーションパターンに適用する方法を示します。

完全なアプリケーションを調べて、変更して実行するには、[アプリケーションの例](https://developer.sima.ai/examples)を参照してください。

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>ここから始めましょう。</h2>
    <p>まず、動作環境を構築し、そこから主要な機能を段階的に追加していく。 SiMa.ai Neat アプリケーションのワークフロー。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>こんにちは、Neat！</strong><span>最小限の Neat アプリケーションを実行し、開発サイクルが正常に機能することを確認します。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/development-workflow/overview/"><strong>開発ワークフロー</strong><span>`Model`、`Graph`、および`Run`のワークフローについて、さらに詳しく学びましょう。</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>チュートリアル</strong><span>実際の事例を参考に、手順に従って進めてください。 SiMa.ai Neat アプリケーションのパターン。</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-explore">
    <h2>さらに構築する</h2>
    <p>より高度なアプリケーションを開発したり、APIの機能を詳しく調べたりする準備が整ったら、これらのセクションを参照してください。</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/"><strong>高度な概念</strong><span>グラフ、形式、メモリ、スレッド、およびランタイムの動作を理解する。</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>参考文献</strong><span>C++、Python、Model Compiler、トラブルシューティング、および関連資料をご覧ください。</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>貢献する</strong><span>アーキテクチャ、ソースコードのビルド、テストの要件、およびリポジトリの規則について理解する。</span></a></li>
    </ul>
  </section>
</div>
